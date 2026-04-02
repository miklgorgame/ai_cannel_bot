import os
import re
import json
import sqlite3
import feedparser
import requests
from datetime import datetime
from huggingface_hub import InferenceClient

# --- 1. Конфигурация и глобальные переменные ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# Путь к файлу базы данных SQLite
DB_FILE = "bot_memory.db"

# Системный промпт (задаёт личность и правила для бота)
SYSTEM_PROMPT = """
Ты — опытный Python-разработчик и автор IT-канала.
Твоя задача: написать пост на основе предоставленных новостей, используя свой внутренний опыт.
Стиль: дружелюбный, профессиональный, живой, с эмодзи, без маркдауна.
Пиши на русском языке.
Пост должен быть целостным и интересным для подписчиков.
"""

# --- 2. Функции для работы с базой данных (память) ---
def init_db():
    """Создаёт таблицы в SQLite, если они ещё не существуют."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Таблица для хранения истории диалогов
    c.execute('''CREATE TABLE IF NOT EXISTS conversation_history
                 (user_id TEXT, timestamp TEXT, role TEXT, content TEXT)''')
    # Таблица для хранения "запомненной" информации о пользователях
    c.execute('''CREATE TABLE IF NOT EXISTS user_memory
                 (user_id TEXT PRIMARY KEY, memory TEXT)''')
    conn.commit()
    conn.close()

def add_to_history(user_id: str, role: str, content: str):
    """Добавляет сообщение в историю диалога."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO conversation_history VALUES (?, ?, ?, ?)",
              (user_id, timestamp, role, content))
    conn.commit()
    conn.close()

def get_history(user_id: str, limit: int = 10) -> list:
    """Возвращает последние 'limit' сообщений из истории."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM conversation_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
              (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]  # Возвращаем в хронологическом порядке

def save_user_memory(user_id: str, memory: str):
    """Сохраняет информацию о пользователе (например, его имя)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_memory VALUES (?, ?)", (user_id, memory))
    conn.commit()
    conn.close()

def get_user_memory(user_id: str) -> str:
    """Возвращает сохранённую информацию о пользователе."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT memory FROM user_memory WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

# --- 3. Функции для работы с новостями ---
def fetch_news() -> str:
    """Парсит RSS Хабра и возвращает строку с последними новостями."""
    print("📰 Запрашиваю свежие новости с Хабра...")
    news_sources = [
        'https://habr.com/ru/rss/best/daily/?fl=ru',  # Лучшее за день
        'https://habr.com/ru/rss/feed/posts/?fl=ru'   # Новые посты
    ]
    news_items = []
    for rss_url in news_sources:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:  # Берём первые 5 записей из каждой ленты
                title = entry.get('title', 'Нет заголовка')
                link = entry.get('link', '#')
                summary = entry.get('summary', 'Нет описания')
                # Очищаем HTML-теги из summary
                summary = re.sub('<[^<]+?>', '', summary)
                news_items.append(f"• **{title}**\n  {summary[:200]}...\n  Подробнее: {link}")
        except Exception as e:
            print(f"⚠️ Ошибка при парсинге {rss_url}: {e}")

    if not news_items:
        return "Не удалось загрузить новости."

    return "\n\n".join(news_items)

# --- 4. Функции для работы с Hugging Face Inference API ---
def generate_post(news_context: str, user_id: str = None) -> str:
    """Генерирует пост на основе новостей с учётом контекста диалога."""
    print("🤖 Генерирую пост с помощью AI...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: Не указан токен Hugging Face."

    # Загружаем историю диалога, если передан user_id
    history_context = ""
    if user_id:
        history = get_history(user_id)
        if history:
            history_context = "\nИстория нашего диалога:\n"
            for role, content in history:
                history_context += f"{role}: {content}\n"

    # Загружаем сохранённую память о пользователе
    user_memory = get_user_memory(user_id) if user_id else ""
    memory_context = f"\nИнформация о пользователе: {user_memory}\n" if user_memory else ""

    # Формируем полный промпт
    full_prompt = f"{SYSTEM_PROMPT}\n\n{memory_context}{history_context}\n\nАктуальные новости IT:\n{news_context}\n\nНапиши интересный пост для Telegram-канала."

    try:
        client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        post_text = completion.choices[0].message.content.strip()
        if post_text:
            # Сохраняем сгенерированный пост в историю (если есть user_id)
            if user_id:
                add_to_history(user_id, "assistant", post_text)
            return post_text
        else:
            return "⚠️ Модель вернула пустой ответ."
    except Exception as e:
        return f"❌ Ошибка генерации: {str(e)}"

# --- 5. Функции для отправки сообщений в Telegram ---
def escape_markdown(text: str) -> str:
    """Экранирует спецсимволы для Telegram MarkdownV2."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def send_to_telegram(text: str, is_error: bool = False) -> bool:
    """Отправляет сообщение в Telegram."""
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
            # fallback без форматирования
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

# --- 6. Основная логика работы бота ---
if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")
    # Инициализируем базу данных
    init_db()

    # Получаем новости
    news = fetch_news()
    if not news:
        send_to_telegram("❌ Не удалось загрузить новости.", is_error=True)
        exit(1)

    # Сохраняем новости в историю (как сообщение от пользователя)
    # Для этого нужно знать user_id. Для каналов его можно получить из TG_CHAT_ID
    user_id = str(TG_CHAT_ID) if TG_CHAT_ID else None
    if user_id:
        add_to_history(user_id, "user", news)

    # Генерируем пост
    post = generate_post(news, user_id)

    # Отправляем результат
    if post:
        if send_to_telegram(post):
            print("[PUBLISH_SUCCESS] Пост отправлен в Telegram")
        else:
            send_to_telegram("⚠️ Пост создан, но не отправлен", is_error=True)
    else:
        send_to_telegram("❌ Не удалось сгенерировать пост.", is_error=True)
