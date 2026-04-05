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
TG_CHANNEL_ID = os.getenv("TG_CHAT_ID")          # ID канала
TG_GROUP_ID = os.getenv("TG_GROUP_ID")           # ID группы для комментариев
CREATOR_ID = int(os.getenv("CREATOR_ID", "0"))

TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

DB_FILE = "bot_memory.db"
OFFSET_FILE = "last_update_id.txt"
IZHEVSK_TZ = ZoneInfo("Europe/Samara")           # Ижевск UTC+4

COMMENTS_CHAT_ID = TG_GROUP_ID if TG_GROUP_ID else TG_CHANNEL_ID

# Приоритет источников (чем меньше число, тем выше приоритет)
SOURCE_PRIORITY = {
    "Habr": 1,
    "iXBT": 2,
    "3DNews": 3,
    "SecurityLab": 4,
    "vc.ru": 5,
    "Kod": 6,
    "CNews": 7,
    "Ferra": 8,
    "Overclockers": 9,
}

# Ключевые слова для повышения приоритета
KEYWORDS = [
    'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'golang', 'rust',
    'разработка', 'программирование', 'code', 'developer', 'алгоритм', 'фреймворк',
    'нейросеть', 'искусственный интеллект', 'ai', 'machine learning', 'data science'
]

GREETINGS = [
    "Привет, друзья! 👋",
    "Здравствуйте, уважаемые подписчики! 💻",
    "Всем привет! 🤗",
    "Доброго времени суток! 🌞",
    "Приветствую, IT-энтузиасты! 🚀"
]

CLOSINGS = [
    "А что вы думаете по этому поводу? Делитесь мнением в комментариях! 💬",
    "Как вам такие новости? Напишите в обсуждении! 👇",
    "Интересно? Ставьте реакции и пишите свои мысли! 😊",
    "Следите за обновлениями, впереди ещё много интересного! 🔥"
]

# Системный промпт с характером
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

# --- СПИСОК МОДЕЛЕЙ ДЛЯ FALLBACK (в порядке приоритета) ---
# Здесь перечислены модели, которые будут использоваться, если основная недоступна
FALLBACK_MODELS = [
    "deepseek-ai/DeepSeek-V3",          # Основная мощная модель
    "Qwen/Qwen2.5-7B-Instruct",         # Хорошая альтернатива с поддержкой русского
    "meta-llama/Llama-3.1-8B-Instruct", # Популярная модель от Meta
    "IlyaGusev/saiga_llama3_8b",        # Специализированная русскоязычная модель
    "mistralai/Mistral-7B-Instruct-v0.3" # Ещё один надёжный вариант
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
    result = c.fetchone() is not None
    conn.close()
    return result

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
    exists = c.fetchone() is not None
    conn.close()
    return exists

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

# ===================== OFFSET ДЛЯ getUpdates =====================
def get_last_offset():
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_last_offset(offset):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))

# ===================== НОВОСТИ С ПРИОРИТЕТАМИ =====================
def calculate_priority(news_item):
    """Чем меньше число, тем выше приоритет."""
    source = news_item.get('source', '')
    source_priority = SOURCE_PRIORITY.get(source, 10)
    text = (news_item['title'] + ' ' + news_item['summary']).lower()
    keyword_boost = 0
    for kw in KEYWORDS:
        if kw in text:
            keyword_boost -= 1
    return source_priority + keyword_boost

def fetch_fresh_news(limit: int = 5):
    """
    Собирает новости из RSS, избегает уже опубликованных,
    сортирует по приоритету, но гарантирует возврат ровно `limit` новостей.
    """
    print("📰 Запрашиваю свежие новости из нескольких источников...")
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
            print(f"    - Загружаю новости из {source['name']}...")
            feed = feedparser.parse(source['url'])
            for entry in feed.entries[:20]:
                link = entry.get('link', '')
                if not link or is_news_already_published(link):
                    continue
                title = entry.get('title', 'Без заголовка')
                summary = entry.get('summary', '')
                summary = re.sub('<[^<]+?>', '', summary)
                if len(summary) > 500:
                    summary = summary[:500] + '...'
                all_candidates.append({
                    'source': source['name'],
                    'title': title,
                    'link': link,
                    'summary': summary
                })
        except Exception as e:
            print(f"    ⚠️ Ошибка при парсинге RSS {source['name']}: {e}")
            continue

    if not all_candidates:
        print("⚠️ Не удалось загрузить новости из доступных источников.")
        return []

    # Сортируем все кандидаты по приоритету
    all_candidates.sort(key=calculate_priority)

    # Берём первые `limit` из отсортированного списка
    top_news = all_candidates[:limit]
    print(f"✅ Отобрано {len(top_news)} новостей для публикации.")
    return top_news

# ===================== ГЕНЕРАЦИЯ ПОСТА С FALLBACK =====================
def generate_post(news_list):
    print("🤖 Генерирую пост...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: Не указан токен Hugging Face."

    random_greeting = random.choice(GREETINGS)
    random_closing = random.choice(CLOSINGS)

    # Формируем контекст новостей
    news_context = "\n\n".join([
        f"ИСТОЧНИК: {n['source']}\nЗАГОЛОВОК: {n['title']}\nОПИСАНИЕ: {n['summary']}\nССЫЛКА: {n['link']}"
        for n in news_list
    ])

    # Промпт с характером
    full_prompt = f"""{SYSTEM_PROMPT}

Вот свежие новости:

{news_context}

Напиши пост. Начни его с приветствия: "{random_greeting}". Затем перескажи новости. В конце добавь вопрос к подписчикам или пожелание, например: "{random_closing}"
"""

    # --- FALLBACK ЛОГИКА: ПЕРЕБИРАЕМ МОДЕЛИ ПО ОЧЕРЕДИ ---
    for model_id in FALLBACK_MODELS:
        try:
            print(f"🔄 Пробую модель: {model_id}")
            client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=800,
                temperature=0.8,
            )
            result = completion.choices[0].message.content.strip()
            if result:
                print(f"✅ Успешно сгенерировано с помощью {model_id}")
                return result
            else:
                print(f"⚠️ Модель {model_id} вернула пустой ответ, пробую следующую...")
        except Exception as e:
            print(f"⚠️ Ошибка с моделью {model_id}: {str(e)[:100]}, пробую следующую...")
            continue

    # Если ни одна модель не сработала
    return "❌ Не удалось сгенерировать пост. Все модели недоступны."

# ===================== ОТВЕТЫ НА КОММЕНТАРИИ С FALLBACK =====================
def generate_reply(comment_text: str, post_content: str) -> str:
    prompt = f"""
Ты — автор IT-канала. Подписчик оставил комментарий к твоему посту.

Твой характер: любишь программирование и нейросети, не любишь Max и цифровой контроль (но не явно, а с лёгкой иронией). Отвечай дружелюбно, с юмором, но профессионально.

Пост был о: {post_content[:500]}

Комментарий подписчика: "{comment_text}"

Напиши ответ (2-3 предложения). Не копируй фразы из комментария, а отвечай по существу. Если комментарий про Max — можешь слегка подколоть. Если про Python или ИИ — порадуйся вместе с подписчиком.
"""
    # --- FALLBACK ЛОГИКА ДЛЯ ОТВЕТОВ ---
    for model_id in FALLBACK_MODELS:
        try:
            print(f"🔄 Пробую модель для ответа: {model_id}")
            client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.8,
            )
            result = completion.choices[0].message.content.strip()
            if result:
                print(f"✅ Ответ сгенерирован с помощью {model_id}")
                return result
            else:
                print(f"⚠️ Модель {model_id} вернула пустой ответ, пробую следующую...")
        except Exception as e:
            print(f"⚠️ Ошибка с моделью {model_id}: {str(e)[:100]}, пробую следующую...")
            continue
    return None

# ===================== ПРОВЕРКА КОММЕНТАРИЕВ =====================
def check_and_reply_to_comments():
    print("💬 Проверяю комментарии в группе обсуждения...")
    last_post = get_last_post()
    if not last_post:
        print("Нет постов для проверки комментариев.")
        return
    post_message_id = last_post['message_id']
    offset = get_last_offset()
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 30, "allowed_updates": ["message"]}
    try:
        resp = requests.get(url, params=params, timeout=35)
        data = resp.json()
        if not data.get("ok"):
            print(f"Ошибка getUpdates: {data}")
            return
        max_update_id = offset - 1
        for update in data.get("result", []):
            update_id = update.get("update_id")
            if update_id > max_update_id:
                max_update_id = update_id
            message = update.get("message")
            if not message:
                continue
            chat_id = message.get("chat", {}).get("id")
            if COMMENTS_CHAT_ID and str(chat_id) != str(COMMENTS_CHAT_ID):
                continue
            reply_to = message.get("reply_to_message")
            if not reply_to or reply_to.get("message_id") != post_message_id:
                continue
            comment_id = message.get("message_id")
            if is_comment_processed(comment_id):
                continue
            comment_text = message.get("text", "")
            username = message.get("from", {}).get("username", "подписчик")
            if not comment_text:
                continue
            print(f"📝 Новый комментарий от @{username}: {comment_text[:50]}...")
            reply_text = generate_reply(comment_text, last_post['content'])
            if reply_text:
                reply_payload = {
                    "chat_id": chat_id,
                    "text": reply_text,
                    "reply_to_message_id": comment_id
                }
                reply_resp = requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", json=reply_payload, timeout=30)
                if reply_resp.status_code == 200:
                    print(f"✅ Ответ отправлен на комментарий {comment_id}")
                    mark_comment_processed(comment_id, last_post['id'])
                else:
                    print(f"❌ Ошибка отправки ответа: {reply_resp.text}")
            else:
                print(f"⚠️ Не удалось сгенерировать ответ на комментарий {comment_id}")
                mark_comment_processed(comment_id, last_post['id'])
        if max_update_id >= offset:
            save_last_offset(max_update_id + 1)
    except Exception as e:
        print(f"❌ Ошибка при проверке комментариев: {e}")

# ===================== ПРОВЕРКА СООБЩЕНИЙ СОЗДАТЕЛЯ =====================
def check_creator_messages():
    print("👤 Проверяю личные сообщения от создателя...")
    offset = get_last_offset()
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 30}
    try:
        resp = requests.get(url, params=params, timeout=35)
        data = resp.json()
        if not data.get("ok"):
            return
        max_update_id = offset - 1
        for update in data.get("result", []):
            update_id = update.get("update_id")
            if update_id > max_update_id:
                max_update_id = update_id
            message = update.get("message")
            if not message:
                continue
            user_id = message.get("from", {}).get("id")
            text = message.get("text", "")
            if user_id == CREATOR_ID and text:
                print(f"📨 Сообщение от создателя: {text}")
                if text.startswith("/generate"):
                    publish_new_post()
                elif text.startswith("/stats"):
                    posts = get_recent_posts(10)
                    stats = f"📊 Постов всего: {len(posts)}\nПоследний: {posts[0]['created_at'][:16] if posts else 'нет'}"
                    send_telegram_message(CREATOR_ID, stats)
                else:
                    send_telegram_message(CREATOR_ID, f"✅ Сообщение получено: '{text}'\nДоступные команды: /generate, /stats")
        if max_update_id >= offset:
            save_last_offset(max_update_id + 1)
    except Exception as e:
        print(f"❌ Ошибка при проверке сообщений создателя: {e}")

def send_telegram_message(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
        if resp.status_code == 200:
            print(f"✅ Сообщение отправлено в чат {chat_id}")
        else:
            print(f"❌ Ошибка отправки: {resp.text}")
    except Exception as e:
        print(f"❌ Исключение: {e}")

# ===================== ПУБЛИКАЦИЯ НОВОГО ПОСТА =====================
def publish_new_post():
    print("📝 Начинаю публикацию нового поста...")
    news_list = fetch_fresh_news(limit=5)
    if not news_list:
        print("❌ Нет новых новостей для публикации.")
        send_telegram_message(CREATOR_ID, "❌ Нет новых новостей для публикации.")
        return

    post_content = generate_post(news_list)
    if post_content and not post_content.startswith("❌"):
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHANNEL_ID, "text": post_content}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                message_id = result.get("result", {}).get("message_id")
                save_post(message_id, post_content)
                for news in news_list:
                    save_published_news(news['link'], news['title'])
                print(f"✅ Пост опубликован! ID: {message_id}")
                send_telegram_message(CREATOR_ID, f"✅ Пост опубликован! ID: {message_id}")
            else:
                print(f"❌ Ошибка публикации: {resp.text}")
                send_telegram_message(CREATOR_ID, f"❌ Ошибка публикации: {resp.text[:100]}")
        except Exception as e:
            print(f"❌ Исключение: {e}")
            send_telegram_message(CREATOR_ID, f"❌ Исключение: {e}")
    else:
        print(f"❌ Не удалось сгенерировать пост: {post_content}")
        send_telegram_message(CREATOR_ID, f"❌ Не удалось сгенерировать пост: {post_content}")

# ===================== ОСНОВНАЯ ЛОГИКА =====================
def main():
    print("🚀 Запуск бота...")
    init_db()
    clean_old_news(days=7)

    if TEST_MODE:
        print("🧪 ТЕСТОВЫЙ РЕЖИМ: публикую пост и отвечаю на комментарии")
        publish_new_post()
        check_and_reply_to_comments()
        check_creator_messages()
        return

    now_izhevsk = datetime.now(IZHEVSK_TZ)
    current_hour = now_izhevsk.hour
    print(f"🕐 Текущее время по Ижевску: {now_izhevsk.strftime('%Y-%m-%d %H:%M:%S')}")

    PUBLISH_HOURS = [9, 12, 18]
    if current_hour in PUBLISH_HOURS:
        print(f"⏰ {current_hour}:00 — публикую пост!")
        publish_new_post()
    else:
        print(f"🕐 {current_hour}:00 — проверяю комментарии и сообщения создателя")
        check_and_reply_to_comments()
        check_creator_messages()

    print("✅ Работа завершена")

if __name__ == "__main__":
    main()
