import os
import re
import sqlite3
import feedparser
import requests
from datetime import datetime
from huggingface_hub import InferenceClient

# --- 1. Конфигурация ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")
DB_FILE = "bot_memory.db"

SYSTEM_PROMPT = """
Ты — опытный Python-разработчик и автор IT-канала.
Твоя задача: на основе предоставленных новостей (каждая новость содержит заголовок и краткое описание) написать пост для Telegram.

Требования:
- Не копируй заголовки и описания дословно.
- Сделай краткий пересказ каждой новости (2-3 предложения), передавая главную мысль.
- Стиль: дружелюбный, профессиональный, живой, с эмодзи, без маркдауна.
- Пиши на русском языке.
- В конце поста можно перечислить ссылки на источники.

Пост должен быть целостным и интересным для подписчиков.
"""

# --- 2. Память (SQLite) ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversation_history
                 (user_id TEXT, timestamp TEXT, role TEXT, content TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_memory
                 (user_id TEXT PRIMARY KEY, memory TEXT)''')
    conn.commit()
    conn.close()

def add_to_history(user_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO conversation_history VALUES (?, ?, ?, ?)",
              (user_id, timestamp, role, content))
    conn.commit()
    conn.close()

def get_history(user_id: str, limit: int = 10):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM conversation_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
              (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]

def get_user_memory(user_id: str) -> str:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT memory FROM user_memory WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

# --- 3. Получение новостей с Хабра ---
def fetch_news() -> str:
    print("📰 Запрашиваю свежие новости с Хабра...")
    rss_url = 'https://habr.com/ru/rss/best/daily/?fl=ru'
    news_items = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:5]:
            title = entry.get('title', 'Без заголовка')
            link = entry.get('link', '#')
            summary = entry.get('summary', '')
            # Очищаем HTML-теги из описания
            summary = re.sub('<[^<]+?>', '', summary)
            # Обрезаем слишком длинное описание
            if len(summary) > 500:
                summary = summary[:500] + '...'
            news_items.append(f"ЗАГОЛОВОК: {title}\nОПИСАНИЕ: {summary}\nССЫЛКА: {link}")
    except Exception as e:
        print(f"⚠️ Ошибка при парсинге RSS: {e}")
        return "Не удалось загрузить новости."

    if not news_items:
        return "Нет свежих новостей."

    return "\n\n".join(news_items)

# --- 4. Генерация поста через Hugging Face ---
def generate_post(news_context: str, user_id: str = None) -> str:
    print("🤖 Генерирую пост с кратким пересказом новостей...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: Не указан токен Hugging Face."

    history_context = ""
    if user_id:
        history = get_history(user_id)
        if history:
            history_context = "\nИстория диалога:\n"
            for role, content in history:
                history_context += f"{role}: {content}\n"

    user_memory = get_user_memory(user_id) if user_id else ""
    memory_context = f"\nИнформация о пользователе: {user_memory}\n" if user_memory else ""

    full_prompt = f"""{SYSTEM_PROMPT}

{memory_context}{history_context}

Вот свежие новости IT (каждая новость содержит заголовок, описание и ссылку):

{news_context}

Напиши пост для Telegram-канала, где для каждой новости сделай краткий пересказ своими словами (2-3 предложения), не копируя заголовок. В конце поста добавь ссылки на источники.
"""

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=800,
            temperature=0.7,
        )
        post_text = completion.choices[0].message.content.strip()
        if post_text:
            if user_id:
                add_to_history(user_id, "assistant", post_text)
            return post_text
        else:
            return "⚠️ Модель вернула пустой ответ."
    except Exception as e:
        return f"❌ Ошибка генерации: {str(e)}"

# --- 5. Отправка в Telegram ---
def escape_markdown(text: str) -> str:
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def send_to_telegram(text: str, is_error: bool = False) -> bool:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("❌ Нет токена или chat_id")
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    if is_error:
        payload = {"chat_id": TG_CHAT_ID, "text": text}
    else:
        safe = escape_markdown(text)
        payload = {"chat_id": TG_CHAT_ID, "text": safe, "parse_mode": "MarkdownV2"}
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            print("✅ Сообщение отправлено в Telegram")
            return True
        if not is_error and "can't parse entities" in resp.text:
            payload2 = {"chat_id": TG_CHAT_ID, "text": text}
            resp2 = requests.post(url, json=payload2, timeout=30)
            if resp2.status_code == 200:
                print("✅ Отправлено без форматирования")
                return True
        print(f"❌ Ошибка Telegram: {resp.text}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

# --- 6. Основной запуск ---
if __name__ == "__main__":
    print("🚀 Запуск генерации поста с кратким пересказом статей...")
    init_db()

    news = fetch_news()
    if not news or news.startswith("Не удалось"):
        send_to_telegram("❌ Не удалось загрузить новости.", is_error=True)
        exit(1)

    user_id = str(TG_CHAT_ID) if TG_CHAT_ID else None
    if user_id:
        add_to_history(user_id, "user", news)

    post = generate_post(news, user_id)

    if post and not post.startswith("❌"):
        send_to_telegram(post)
    else:
        send_to_telegram(post or "❌ Не удалось сгенерировать пост.", is_error=True)
