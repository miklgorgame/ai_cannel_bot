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
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")     # API ключ Pexels (бесплатно)

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

# --- СПИСОК МОДЕЛЕЙ ДЛЯ FALLBACK (ГЕНЕРАЦИЯ ТЕКСТА) ---
FALLBACK_MODELS = [
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "IlyaGusev/saiga_llama3_8b",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

# --- СПИСОК МОДЕЛЕЙ ДЛЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ (РЕЗЕРВ) ---
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
    source = news_item.get('source', '')
    source_priority = SOURCE_PRIORITY.get(source, 10)
    text = (news_item['title'] + ' ' + news_item['summary']).lower()
    keyword_boost = 0
    for kw in KEYWORDS:
        if kw in text:
            keyword_boost -= 1
    return source_priority + keyword_boost

def fetch_fresh_news(limit: int = 5):
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
    all_candidates.sort(key=calculate_priority)
    top_news = all_candidates[:limit]
    print(f"✅ Отобрано {len(top_news)} новостей для публикации.")
    return top_news

# ===================== ПОИСК ИЗОБРАЖЕНИЙ НА PEXELS =====================
def search_pexels_image(query: str) -> bytes | None:
    if not PEXELS_API_KEY:
        print("⚠️ API-ключ Pexels не задан. Поиск изображений недоступен.")
        return None
    print(f"🔍 Ищу изображение на Pexels по запросу: {query[:100]}...")
    search_query = ' '.join(query.split()[:5])
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_query, "per_page": 5, "orientation": "landscape"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            photos = data.get("photos", [])
            if photos:
                photo_url = photos[0]["src"]["large"]
                print(f"✅ Изображение найдено на Pexels. Скачиваю...")
                img_response = requests.get(photo_url, timeout=30)
                if img_response.status_code == 200:
                    return img_response.content
                else:
                    print(f"⚠️ Не удалось скачать изображение: {img_response.status_code}")
            else:
                print("⚠️ По запросу ничего не найдено на Pexels.")
        else:
            print(f"⚠️ Ошибка Pexels API: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Ошибка при поиске на Pexels: {e}")
    return None

# ===================== ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ ЧЕРЕЗ AI (РЕЗЕРВ) =====================
def generate_image(prompt: str) -> bytes | None:
    print("🎨 Начинаю резервную генерацию изображения через AI...")
    if not HF_API_TOKEN:
        print("❌ Нет токена HF для генерации изображения")
        return None
    enhanced_prompt = f"Create a realistic and engaging cover image for a news article. The image should be high quality and thematically match the text: {prompt}. Do not include any text or letters in the image."
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    for model_id in IMAGE_MODELS:
        try:
            print(f"🔄 Пробую модель для изображения: {model_id}")
            payload = {"inputs": enhanced_prompt}
            API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                print(f"✅ Изображение успешно сгенерировано с помощью {model_id}")
                return response.content
            else:
                print(f"⚠️ Модель {model_id} вернула ошибку {response.status_code}, пробую следующую...")
        except Exception as e:
            print(f"⚠️ Ошибка с моделью {model_id}: {str(e)[:100]}, пробую следующую...")
            continue
    print("❌ Не удалось сгенерировать изображение ни одной из моделей.")
    return None

# ===================== ОТПРАВКА ФОТО С УМНОЙ ОБРЕЗКОЙ ПОДПИСИ =====================
def send_telegram_photo(chat_id: int, photo_bytes: bytes, caption: str) -> bool:
    MAX_CAPTION = 1000  # оставляем запас под лимит Telegram (1024)
    url_photo = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("image.png", photo_bytes, "image/png")}
    if len(caption) <= MAX_CAPTION:
        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
        try:
            resp = requests.post(url_photo, files=files, data=data, timeout=30)
            if resp.status_code == 200:
                print("✅ Фото с подписью отправлено")
                return True
            else:
                print(f"❌ Ошибка отправки фото: {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Исключение: {e}")
            return False
    else:
        # Обрезаем подпись
        short_caption = caption[:MAX_CAPTION] + "\n\n… (продолжение в следующем сообщении)"
        data = {"chat_id": chat_id, "caption": short_caption, "parse_mode": "HTML"}
        try:
            resp_photo = requests.post(url_photo, files=files, data=data, timeout=30)
            if resp_photo.status_code != 200:
                print(f"❌ Ошибка отправки фото: {resp_photo.text}")
                return False
            print("✅ Фото с обрезанной подписью отправлено")
            # Отправляем остаток
            remainder = caption[MAX_CAPTION:]
            url_msg = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": chat_id, "text": remainder, "parse_mode": "HTML"}
            resp_msg = requests.post(url_msg, json=payload, timeout=30)
            if resp_msg.status_code == 200:
                print("✅ Остаток текста отправлен")
                return True
            else:
                print(f"❌ Ошибка отправки остатка: {resp_msg.text}")
                return False
        except Exception as e:
            print(f"❌ Исключение: {e}")
            return False

def send_telegram_message(chat_id: int, text: str) -> bool:
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)
        if resp.status_code == 200:
            print(f"✅ Сообщение отправлено в чат {chat_id}")
            return True
        else:
            print(f"❌ Ошибка отправки: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False

# ===================== ГЕНЕРАЦИЯ ПОСТА С FALLBACK =====================
def generate_post(news_list):
    print("🤖 Генерирую пост...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: Не указан токен Hugging Face."
    random_greeting = random.choice(GREETINGS)
    random_closing = random.choice(CLOSINGS)
    news_context = "\n\n".join([
        f"ИСТОЧНИК: {n['source']}\nЗАГОЛОВОК: {n['title']}\nОПИСАНИЕ: {n['summary']}\nССЫЛКА: {n['link']}"
        for n in news_list
    ])
    full_prompt = f"""{SYSTEM_PROMPT}

Вот свежие новости:

{news_context}

Напиши пост. Начни его с приветствия: "{random_greeting}". Затем перескажи новости. В конце добавь вопрос к подписчикам или пожелание, например: "{random_closing}"
"""
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
    return "❌ Не удалось сгенерировать пост. Все модели недоступны."

def generate_reply(comment_text: str, post_content: str) -> str:
    prompt = f"""
Ты — автор IT-канала. Подписчик оставил комментарий к твоему посту.

Твой характер: любишь программирование и нейросети, не любишь Max и цифровой контроль (но не явно, а с лёгкой иронией). Отвечай дружелюбно, с юмором, но профессионально.

Пост был о: {post_content[:500]}

Комментарий подписчика: "{comment_text}"

Напиши ответ (2-3 предложения). Не копируй фразы из комментария, а отвечай по существу. Если комментарий про Max — можешь слегка подколоть. Если про Python или ИИ — порадуйся вместе с подписчиком.
"""
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
                reply_payload = {"chat_id": chat_id, "text": reply_text, "reply_to_message_id": comment_id}
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

# ===================== ПУБЛИКАЦИЯ НОВОГО ПОСТА =====================
def publish_new_post():
    print("📝 Начинаю публикацию нового поста...")
    news_list = fetch_fresh_news(limit=5)
    if not news_list:
        print("❌ Нет новых новостей для публикации.")
        send_telegram_message(CREATOR_ID, "❌ Нет новых новостей для публикации.")
        return
    top_news = news_list[0]
    print(f"🏆 Выбрана главная новость: {top_news['title'][:80]}...")
    post_content = generate_post(news_list)
    if not post_content or post_content.startswith("❌"):
        print(f"❌ Не удалось сгенерировать пост: {post_content}")
        send_telegram_message(CREATOR_ID, f"❌ Не удалось сгенерировать пост: {post_content}")
        return
    # Ищем картинку
    image_bytes = search_pexels_image(top_news['title'])
    if not image_bytes:
        image_prompt = f"{top_news['title']}. {top_news['summary']}"
        image_bytes = generate_image(image_prompt)
    if image_bytes:
        success = send_telegram_photo(TG_CHANNEL_ID, image_bytes, post_content)
    else:
        success = send_telegram_message(TG_CHANNEL_ID, post_content)
    if success:
        # Сохраняем пост и новости
        # Для фото message_id можно получить из ответа, но для упрощения сохраняем заглушку
        save_post(0, post_content)
        for news in news_list:
            save_published_news(news['link'], news['title'])
        print("✅ Пост опубликован!")
        send_telegram_message(CREATOR_ID, "✅ Пост опубликован!")
    else:
        print("❌ Ошибка публикации поста.")
        send_telegram_message(CREATOR_ID, "❌ Ошибка публикации поста.")

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
