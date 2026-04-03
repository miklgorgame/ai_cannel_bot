import os
import re
import sqlite3
import feedparser
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from huggingface_hub import InferenceClient

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHANNEL_ID = os.getenv("TG_CHAT_ID")          # ID канала
CREATOR_ID = int(os.getenv("CREATOR_ID", "0"))   # Ваш Telegram ID

# Режим тестирования: при ручном запуске workflow или local TEST_MODE=true
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

DB_FILE = "bot_memory.db"
IZHEVSK_TZ = ZoneInfo("Europe/Samara")           # Ижевск UTC+4

# --- СИСТЕМНЫЙ ПРОМПТ ДЛЯ ГЕНЕРАЦИИ ПОСТА ---
SYSTEM_PROMPT = """
Ты — опытный Python-разработчик и автор IT-канала.
Твоя задача: на основе предоставленных новостей написать пост для Telegram.

Требования:
- Кратко перескажи каждую новость (2-3 предложения), не копируя заголовок.
- Стиль: дружелюбный, профессиональный, живой, с эмодзи.
- Пиши на русском языке.
- В конце добавь ссылки на источники.
"""

# --- БАЗА ДАННЫХ (SQLite) ---
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

# --- ПОЛУЧЕНИЕ НОВОСТЕЙ ИЗ НЕСКОЛЬКИХ ИСТОЧНИКОВ ---
def fetch_fresh_news(limit: int = 5):
    """Получает новости из нескольких RSS-лент, избегая повторной публикации."""
    print("📰 Запрашиваю свежие новости из нескольких источников...")

    # Список источников (можно легко добавлять/удалять)
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

    news_list = []
    for source in news_sources:
        try:
            print(f"    - Загружаю новости из {source['name']}...")
            feed = feedparser.parse(source['url'])
            for entry in feed.entries[:limit]:
                link = entry.get('link', '')
                if not link or is_news_already_published(link):
                    continue
                title = entry.get('title', 'Без заголовка')
                summary = entry.get('summary', '')
                summary = re.sub('<[^<]+?>', '', summary)
                if len(summary) > 500:
                    summary = summary[:500] + '...'
                news_list.append({
                    'source': source['name'],
                    'title': title,
                    'link': link,
                    'summary': summary
                })
                if len(news_list) >= limit:
                    break
            if len(news_list) >= limit:
                break
        except Exception as e:
            print(f"    ⚠️ Ошибка при парсинге RSS {source['name']}: {e}")
            continue

    if not news_list:
        print("⚠️ Не удалось загрузить новости из доступных источников.")
        return []

    # Возвращаем только первые `limit` новостей
    return news_list[:limit]

# --- ГЕНЕРАЦИЯ ПОСТА ЧЕРЕЗ Hugging Face ---
def generate_post(news_list):
    print("🤖 Генерирую пост...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: Не указан токен Hugging Face."

    # Формируем контекст с указанием источника
    news_context = "\n\n".join([
        f"ИСТОЧНИК: {n['source']}\nЗАГОЛОВОК: {n['title']}\nОПИСАНИЕ: {n['summary']}\nССЫЛКА: {n['link']}"
        for i, n in enumerate(news_list)
    ])

    full_prompt = f"{SYSTEM_PROMPT}\n\nВот свежие новости:\n\n{news_context}"

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        return f"❌ Ошибка: {str(e)}"

# --- ОТВЕТЫ НА КОММЕНТАРИИ (не шаблонные) ---
def generate_reply(comment_text: str, post_content: str) -> str:
    prompt = f"""
Ты — автор IT-канала. Подписчик оставил комментарий к твоему посту.

Пост был о: {post_content[:500]}

Комментарий подписчика: "{comment_text}"

Напиши дружелюбный, полезный ответ на этот комментарий (2-3 предложения). 
Будь вежливым, профессиональным. Используй эмодзи. Не отвечай шаблонно — отвечай именно по существу комментария.
"""
    try:
        client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.8,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Ошибка генерации ответа: {e}")
        return None

def check_and_reply_to_comments():
    """Проверяет комментарии к последнему посту и отвечает на новые (один раз)."""
    print("💬 Проверяю комментарии к последнему посту...")
    last_post = get_last_post()
    if not last_post:
        print("Нет постов для проверки комментариев.")
        return

    post_message_id = last_post['message_id']
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if not data.get("ok"):
            print(f"Ошибка getUpdates: {data}")
            return

        for update in data.get("result", []):
            message = update.get("message")
            if not message:
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
                reply_url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
                reply_payload = {
                    "chat_id": TG_CHANNEL_ID,
                    "text": reply_text,
                    "reply_to_message_id": comment_id
                }
                reply_resp = requests.post(reply_url, json=reply_payload, timeout=30)
                if reply_resp.status_code == 200:
                    print(f"✅ Ответ отправлен на комментарий {comment_id}")
                    mark_comment_processed(comment_id, last_post['id'])
                else:
                    print(f"❌ Ошибка отправки ответа: {reply_resp.text}")
            else:
                print(f"⚠️ Не удалось сгенерировать ответ на комментарий {comment_id}")
                mark_comment_processed(comment_id, last_post['id'])  # чтобы не пытаться снова
    except Exception as e:
        print(f"❌ Ошибка при проверке комментариев: {e}")

# --- ПРОВЕРКА ЛИЧНЫХ СООБЩЕНИЙ ОТ СОЗДАТЕЛЯ ---
def check_creator_messages():
    print("👤 Проверяю личные сообщения от создателя...")
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if not data.get("ok"):
            return

        for update in data.get("result", []):
            message = update.get("message")
            if not message:
                continue
            user_id = message.get("from", {}).get("id")
            text = message.get("text", "")
            if user_id == CREATOR_ID and text:
                print(f"📨 Сообщение от создателя: {text}")
                if text.startswith("/generate"):
                    print("🔄 Принудительная публикация поста по команде создателя")
                    publish_new_post()
                elif text.startswith("/stats"):
                    posts = get_recent_posts(10)
                    stats = f"📊 Постов всего: {len(posts)}\nПоследний: {posts[0]['created_at'][:16] if posts else 'нет'}"
                    send_telegram_message(CREATOR_ID, stats)
                else:
                    send_telegram_message(CREATOR_ID, f"✅ Сообщение получено: '{text}'\nДоступные команды: /generate, /stats")
                # После обработки можно удалить апдейт, используя offset, но для простоты пропускаем
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

# --- ПУБЛИКАЦИЯ НОВОГО ПОСТА ---
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

# --- ОСНОВНАЯ ЛОГИКА ---
def main():
    print("🚀 Запуск бота...")
    init_db()
    clean_old_news(days=7)

    # Если тестовый режим (ручной запуск или TEST_MODE=true) – делаем всё: публикуем пост и отвечаем на комментарии
    if TEST_MODE:
        print("🧪 ТЕСТОВЫЙ РЕЖИМ: публикую пост и отвечаю на комментарии")
        publish_new_post()
        check_and_reply_to_comments()
        check_creator_messages()
        return

    # Обычный режим по расписанию
    now_izhevsk = datetime.now(IZHEVSK_TZ)
    current_hour = now_izhevsk.hour
    print(f"🕐 Текущее время по Ижевску: {now_izhevsk.strftime('%Y-%m-%d %H:%M:%S')}")

    if current_hour == 12:
        print("⏰ 12:00 по Ижевску — публикую пост!")
        publish_new_post()
    else:
        print(f"🕐 {current_hour}:00 — проверяю комментарии и сообщения создателя")
        check_and_reply_to_comments()
        check_creator_messages()

    print("✅ Работа завершена")

if __name__ == "__main__":
    main()
