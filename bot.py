import os
import re
import sqlite3
import feedparser
import requests
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from huggingface_hub import InferenceClient

# ===================== КОНФИГУРАЦИЯ =====================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHANNEL_ID = os.getenv("TG_CHAT_ID")
TG_GROUP_ID = os.getenv("TG_GROUP_ID")
CREATOR_ID = int(os.getenv("CREATOR_ID", "0"))
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

DB_FILE = "bot_memory.db"
OFFSET_FILE = "last_update_id.txt"
IZHEVSK_TZ = ZoneInfo("Europe/Samara")
COMMENTS_CHAT_ID = TG_GROUP_ID if TG_GROUP_ID else TG_CHANNEL_ID

SOURCE_PRIORITY = {
    "Habr": 1, "iXBT": 2, "3DNews": 3, "SecurityLab": 4,
    "vc.ru": 5, "Kod": 6, "CNews": 7, "Ferra": 8, "Overclockers": 9,
}

KEYWORDS = [
    'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'golang', 'rust',
    'разработка', 'программирование', 'code', 'developer', 'алгоритм', 'фреймворк',
    'нейросеть', 'искусственный интеллект', 'ai', 'machine learning', 'data science'
]

GREETINGS = [
    "Привет, друзья! 👋", "Здравствуйте, уважаемые подписчики! 💻",
    "Всем привет! 🤗", "Доброго времени суток! 🌞", "Приветствую, IT-энтузиасты! 🚀"
]

CLOSINGS = [
    "А что вы думаете по этому поводу? Делитесь мнением в комментариях! 💬",
    "Как вам такие новости? Напишите в обсуждении! 👇",
    "Интересно? Ставьте реакции и пишите свои мысли! 😊",
    "Следите за обновлениями, впереди ещё много интересного! 🔥"
]

SYSTEM_PROMPT = """
Ты — опытный Python-разработчик и автор IT-канала. Твой стиль — живой, с юмором, иногда с лёгкой иронией. Ты обожаешь программирование, нейросети и всё, что связано с технологиями. Ты не любишь цифровой контроль и мессенджер Max (от VK), но не выражаешь это открыто — только лёгкий сарказм в новостях на эту тему. 

**ВАЖНО:** Не копируй текст статьи дословно! Перескажи суть своими словами, как будто ты сам видел эту новость и делишься впечатлением. Избегай фраз «в статье говорится», «автор пишет» — просто расскажи, что произошло.

Твоя задача: на основе предоставленных новостей написать пост для Telegram-канала.

Требования:
- Кратко перескажи каждую новость (2-3 предложения), своими словами, не повторяя заголовок.
- Если новость про Python, нейросети или программирование — добавь немного энтузиазма (эмодзи, восклицания).
- Если новость про Max — можешь легонько подколоть (например: «Max опять сбоит? Ну, бывает...»), но без прямой агрессии.
- Стиль: дружелюбный, профессиональный, с эмодзи.
- Пиши на русском языке.
- В конце поста добавь ссылки на источники.
"""

FALLBACK_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "IlyaGusev/saiga_llama3_8b",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-2-1",
    "CompVis/stable-diffusion-v1-4",
]

# ===================== БАЗА ДАННЫХ =====================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS posts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message_id INTEGER,
                  content TEXT,
                  created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS published_news
                 (link TEXT PRIMARY KEY, title TEXT, published_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS processed_comments
                 (comment_id INTEGER PRIMARY KEY, post_id INTEGER, replied BOOLEAN)''')
    conn.commit()
    conn.close()

def save_post(message_id: int, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    created_at = datetime.now().isoformat()
    c.execute("INSERT INTO posts (message_id, content, created_at) VALUES (?, ?, ?)",
              (message_id, content, created_at))
    conn.commit()
    conn.close()

def get_last_post():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, message_id, content, created_at FROM posts ORDER BY created_at DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "message_id": row[1], "content": row[2], "created_at": row[3]}
    return None

def is_comment_processed(comment_id: int) -> bool:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM processed_comments WHERE comment_id = ?", (comment_id,))
    return c.fetchone() is not None

def mark_comment_processed(comment_id: int, post_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO processed_comments (comment_id, post_id, replied) VALUES (?, ?, ?)",
              (comment_id, post_id, True))
    conn.commit()
    conn.close()

def is_news_already_published(link: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM published_news WHERE link = ?", (link,))
    result = c.fetchone() is not None
    conn.close()
    return result

def save_published_news(link: str, title: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    published_at = datetime.now().isoformat()
    c.execute("INSERT OR IGNORE INTO published_news VALUES (?, ?, ?)",
              (link, title, published_at))
    conn.commit()
    conn.close()

def clean_old_news(days: int = 7):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM published_news WHERE published_at < ?", (cutoff,))
    conn.commit()
    conn.close()

def get_recent_posts(limit: int = 10):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, message_id, content, created_at FROM posts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "message_id": r[1], "content": r[2][:100], "created_at": r[3]} for r in rows]

def get_last_offset():
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_last_offset(offset):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))

# ===================== НОВОСТИ =====================
def calculate_priority(news_item):
    source = news_item.get('source', '')
    source_priority = SOURCE_PRIORITY.get(source, 10)
    text = (news_item['title'] + ' ' + news_item['summary']).lower()
    keyword_boost = 0
    for kw in KEYWORDS:
        if kw in text:
            keyword_boost -= 1
    return source_priority + keyword_boost

def fetch_fresh_news(limit: int = 5):
    print("📰 Запрашиваю свежие новости...")
    news_sources = [
        {"name": "Habr", "url": "https://habr.com/ru/rss/feed/posts/?fl=ru"},
        {"name": "3DNews", "url": "https://3dnews.ru/news/all/"},
        {"name": "CNews", "url": "https://www.cnews.ru/rss/news/"},
        {"name": "iXBT", "url": "https://www.ixbt.com/news.rss"},
        {"name": "Ferra", "url": "https://ferra.ru/rss/news/"},
        {"name": "SecurityLab", "url": "https://www.securitylab.ru/export/rss/"},
        {"name": "vc.ru", "url": "https://vc.ru/rss/"},
        {"name": "Kod", "url": "https://kod.ru/rss/"},
        {"name": "Overclockers", "url": "https://overclockers.ru/rss/news.rss"},
    ]
    all_candidates = []
    for source in news_sources:
        try:
            print(f"    - {source['name']}...")
            feed = feedparser.parse(source['url'])
            for entry in feed.entries[:20]:
                link = entry.get('link')
                if not link or is_news_already_published(link):
                    continue
                title = entry.get('title', 'Без заголовка')
                summary = re.sub('<[^<]+?>', '', entry.get('summary', ''))[:500]
                all_candidates.append({
                    'source': source['name'],
                    'title': title,
                    'link': link,
                    'summary': summary
                })
        except Exception as e:
            print(f"    ⚠️ Ошибка {source['name']}: {e}")
    if not all_candidates:
        return []
    all_candidates.sort(key=calculate_priority)
    return all_candidates[:limit]

# ===================== ПОИСК И ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ =====================
def search_pexels_image(query: str) -> bytes | None:
    if not PEXELS_API_KEY:
        return None
    print(f"🔍 Ищу изображение на Pexels...")
    search_query = ' '.join(query.split()[:5])
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_query, "per_page": 5, "orientation": "landscape"}
    try:
        resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            photos = resp.json().get("photos", [])
            if photos:
                img_url = photos[0]["src"]["large"]
                img_resp = requests.get(img_url, timeout=30)
                if img_resp.status_code == 200:
                    return img_resp.content
    except Exception as e:
        print(f"⚠️ Pexels ошибка: {e}")
    return None

def generate_image(prompt: str) -> bytes | None:
    print("🎨 Резервная генерация изображения...")
    if not HF_API_TOKEN:
        return None
    enhanced_prompt = f"Create a realistic cover image for news: {prompt}. No text."
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    for model in IMAGE_MODELS:
        try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            resp = requests.post(url, headers=headers, json={"inputs": enhanced_prompt}, timeout=60)
            if resp.status_code == 200:
                return resp.content
        except:
            continue
    return None

# ===================== ОТПРАВКА С ОБРЕЗКОЙ ПОДПИСИ =====================
def send_telegram_photo(chat_id: int, photo_bytes: bytes, caption: str) -> bool:
    MAX_CAPTION = 1000  # запас под лимит Telegram 1024
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("image.png", photo_bytes, "image/png")}
    if len(caption) <= MAX_CAPTION:
        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
        try:
            resp = requests.post(url, files=files, data=data, timeout=30)
            return resp.status_code == 200
        except:
            return False
    else:
        # Обрезаем подпись и добавляем пометку о продолжении
        short = caption[:MAX_CAPTION] + "\n\n… (продолжение в следующем сообщении)"
        data = {"chat_id": chat_id, "caption": short, "parse_mode": "HTML"}
        try:
            resp_photo = requests.post(url, files=files, data=data, timeout=30)
            if resp_photo.status_code != 200:
                return False
            # Отправляем остаток текста отдельным сообщением
            remainder = caption[MAX_CAPTION:]
            msg_url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
            resp_msg = requests.post(msg_url, json={"chat_id": chat_id, "text": remainder, "parse_mode": "HTML"}, timeout=30)
            return resp_msg.status_code == 200
        except:
            return False

def send_telegram_message(chat_id: int, text: str) -> bool:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
        return resp.status_code == 200
    except:
        return False

# ===================== ГЕНЕРАЦИЯ ТЕКСТА =====================
def generate_post(news_list):
    print("🤖 Генерирую пост...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: нет токена HF"
    random_greeting = random.choice(GREETINGS)
    random_closing = random.choice(CLOSINGS)
    news_context = "\n\n".join([
        f"ИСТОЧНИК: {n['source']}\nЗАГОЛОВОК: {n['title']}\nОПИСАНИЕ: {n['summary']}\nССЫЛКА: {n['link']}"
        for n in news_list
    ])
    prompt = f"""{SYSTEM_PROMPT}

Вот свежие новости:

{news_context}

Напиши пост. Начни: "{random_greeting}". Перескажи новости. В конце: "{random_closing}"
"""
    for model in FALLBACK_MODELS:
        try:
            client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.8,
            )
            result = completion.choices[0].message.content.strip()
            if result:
                return result
        except:
            continue
    return "❌ Не удалось сгенерировать пост."

def generate_reply(comment_text: str, post_content: str) -> str:
    prompt = f"""
Ты — автор IT-канала. Подписчик: "{comment_text}"
Пост был о: {post_content[:500]}
Ответь дружелюбно, с юмором (2-3 предложения). Если про Max — легкая ирония.
"""
    for model in FALLBACK_MODELS:
        try:
            client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.8,
            )
            result = completion.choices[0].message.content.strip()
            if result:
                return result
        except:
            continue
    return None

# ===================== КОММЕНТАРИИ И СООБЩЕНИЯ СОЗДАТЕЛЯ =====================
def check_and_reply_to_comments():
    print("💬 Проверяю комментарии...")
    last_post = get_last_post()
    if not last_post:
        return
    offset = get_last_offset()
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 30, "allowed_updates": ["message"]}
    try:
        resp = requests.get(url, params=params, timeout=35)
        data = resp.json()
        if not data.get("ok"):
            return
        max_id = offset - 1
        for update in data.get("result", []):
            update_id = update.get("update_id")
            if update_id > max_id:
                max_id = update_id
            msg = update.get("message")
            if not msg:
                continue
            chat_id = msg.get("chat", {}).get("id")
            if COMMENTS_CHAT_ID and str(chat_id) != str(COMMENTS_CHAT_ID):
                continue
            reply_to = msg.get("reply_to_message")
            if not reply_to or reply_to.get("message_id") != last_post['message_id']:
                continue
            comment_id = msg.get("message_id")
            if is_comment_processed(comment_id):
                continue
            text = msg.get("text", "")
            if not text:
                continue
            print(f"📝 Комментарий: {text[:50]}...")
            reply = generate_reply(text, last_post['content'])
            if reply:
                send_telegram_message(chat_id, reply)
                mark_comment_processed(comment_id, last_post['id'])
        if max_id >= offset:
            save_last_offset(max_id + 1)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def check_creator_messages():
    print("👤 Проверяю сообщения создателя...")
    offset = get_last_offset()
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 30}
    try:
        resp = requests.get(url, params=params, timeout=35)
        data = resp.json()
        if not data.get("ok"):
            return
        max_id = offset - 1
        for update in data.get("result", []):
            update_id = update.get("update_id")
            if update_id > max_id:
                max_id = update_id
            msg = update.get("message")
            if not msg:
                continue
            if msg.get("from", {}).get("id") == CREATOR_ID:
                text = msg.get("text", "")
                if text.startswith("/generate"):
                    publish_new_post()
                elif text.startswith("/stats"):
                    posts = get_recent_posts(10)
                    stats = f"📊 Постов: {len(posts)}"
                    send_telegram_message(CREATOR_ID, stats)
                else:
                    send_telegram_message(CREATOR_ID, f"✅ Получено: {text}")
        if max_id >= offset:
            save_last_offset(max_id + 1)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

# ===================== ПУБЛИКАЦИЯ =====================
def publish_new_post():
    print("📝 Публикация нового поста...")
    news_list = fetch_fresh_news(limit=5)
    if not news_list:
        send_telegram_message(CREATOR_ID, "❌ Нет новых новостей.")
        return
    top_news = news_list[0]
    post_content = generate_post(news_list)
    if not post_content or post_content.startswith("❌"):
        send_telegram_message(CREATOR_ID, f"❌ Ошибка генерации: {post_content}")
        return
    # Ищем или генерируем картинку
    image = search_pexels_image(top_news['title'])
    if not image:
        image = generate_image(f"{top_news['title']} {top_news['summary']}")
    if image:
        success = send_telegram_photo(TG_CHANNEL_ID, image, post_content)
    else:
        success = send_telegram_message(TG_CHANNEL_ID, post_content)
    if success:
        save_post(0, post_content)
        for news in news_list:
            save_published_news(news['link'], news['title'])
        send_telegram_message(CREATOR_ID, "✅ Пост опубликован!")
    else:
        send_telegram_message(CREATOR_ID, "❌ Ошибка публикации.")

# ===================== MAIN =====================
def main():
    print("🚀 Запуск бота...")
    init_db()
    clean_old_news()
    if TEST_MODE:
        print("🧪 ТЕСТОВЫЙ РЕЖИМ")
        publish_new_post()
        check_and_reply_to_comments()
        check_creator_messages()
        return
    now = datetime.now(IZHEVSK_TZ)
    hour = now.hour
    print(f"🕐 {now.strftime('%Y-%m-%d %H:%M:%S')}")
    if hour in [9, 12, 18]:
        publish_new_post()
    else:
        check_and_reply_to_comments()
        check_creator_messages()
    print("✅ Готово")

if __name__ == "__main__":
    main()
