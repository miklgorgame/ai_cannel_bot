import os
import re
import sqlite3
import feedparser
import requests
import random
import time
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import InferenceClient
from telegram import Bot
from telegram.error import TelegramError
from groq import Groq
from openai import OpenAI
import json

# ===================== НАСТРОЙКА ЛОГИРОВАНИЯ =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== КОНФИГУРАЦИЯ =====================
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHANNEL_ID = os.getenv("TG_CHAT_ID")
TG_GROUP_ID = os.getenv("TG_GROUP_ID")
CREATOR_ID = int(os.getenv("CREATOR_ID", "0"))
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWELVER_API_KEY = os.getenv("TWELVER_API_KEY")
QUIZ_PROBABILITY = 0.4

TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

DB_FILE = "bot_memory.db"
OFFSET_COMMENTS_FILE = "offset_comments.txt"
OFFSET_CREATOR_FILE = "offset_creator.txt"
IZHEVSK_TZ = ZoneInfo("Europe/Samara")

COMMENTS_CHAT_ID = int(TG_GROUP_ID) if TG_GROUP_ID else (int(TG_CHANNEL_ID) if TG_CHANNEL_ID else None)

MAX_NEWS_AGE_DAYS = 2

SOURCE_PRIORITY = {
    "Habr": 1, "iXBT": 2, "3DNews": 3, "SecurityLab": 1,
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

# ===================== SYSTEM_PROMPT =====================
SYSTEM_PROMPT = """
Ты - опытный Python-разработчик и автор IT-канала. Твой стиль - живой, с юмором, иногда с лёгкой иронией. Ты обожаешь программирование, нейросети и всё, что связано с технологиями. Ты не любишь цифровой контроль и мессенджер Max (от VK), но не выражаешь это открыто - только лёгкий сарказм в новостях на эту тему. 

**ВАЖНО:** Не копируй текст статьи дословно! Перескажи суть своими словами, как будто ты сам видел эту новость и делишься впечатлением. Избегай фраз «в статье говорится», «автор пишет» - просто расскажи, что произошло.

Твоя задача: на основе предоставленных новостей написать пост для Telegram-канала.

Требования:
- Используй Markdown.
- Кратко перескажи каждую новость (2-3 предложения), своими словами, не повторяя заголовок.
- Если новость про Python, нейросети или программирование - добавь немного энтузиазма (эмодзи: 😎🤔😏🙄😴🤑🤠🦾👁👀👩🏻‍💻💻, восклицания).
- Если новость про Max - можешь легонько подколоть, но без прямой агрессии.
- Стиль: дружелюбный, профессиональный, с эмодзи.
- Пиши на русском языке.
- В конце поста добавь ссылки на источники.
- В некоторых интригующих местах используй скрытый текст ||...||

**ПРАВИЛА ФОРМАТИРОВАНИЯ TELEGRAM:**

В Telegram доступны:
- **жирный текст** - обрамляется `**`
- __курсив__ - обрамляется `__`
- ~~зачёркнутый~~ - обрамляется `~~`
- ||спойлер|| - обрамляется `||`
- `код` - обрамляется обратными кавычками
- [ссылка](http://example.com/)

**ВАЖНО:** Точки (.) и восклицательные знаки (!) экранировать НЕ НУЖНО. Экранируются только служебные символы разметки внутри форматирования.

**ЗАПОМНИ:** Ты пишешь пост по этим правилам. Точки и восклицания оставляешь как есть.
"""

# ===================== МОДЕЛИ =====================
HF_MODELS_LEGACY = [
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "IlyaGusev/saiga_llama3_8b",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

HF_MODELS_FREE = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-2b-it"
]

FALLBACK_MODELS = HF_MODELS_LEGACY + HF_MODELS_FREE

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

IMAGE_MODELS = [
    "cutycat2000/InterDiffusion-2.5",
    "cutycat2000x/InterDiffusion-3.5",
    "cutycat2000x/InterDiffusion-4.04",
]

# ===================== HTTP СЕССИЯ =====================
def create_retry_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

http_session = create_retry_session()

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
    logger.info("✅ База данных инициализирована.")

def save_post(message_id: int, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    created_at = datetime.now().isoformat()
    c.execute("INSERT INTO posts (message_id, content, created_at) VALUES (?, ?, ?)",
              (message_id, content, created_at))
    conn.commit()
    conn.close()
    logger.info(f"💾 Пост сохранён в БД: message_id={message_id}")

def get_last_posts(limit: int = 5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, message_id, content, created_at FROM posts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "message_id": r[1], "content": r[2], "created_at": r[3]} for r in rows]

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
    logger.info(f"✅ Комментарий {comment_id} помечен как обработанный.")

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
    logger.info(f"🧹 Удалены старые новости старше {days} дней.")

def get_recent_posts(limit: int = 10):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, message_id, content, created_at FROM posts ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "message_id": r[1], "content": r[2][:100], "created_at": r[3]} for r in rows]

# ===================== УПРАВЛЕНИЕ OFFSET =====================
def get_comments_offset():
    if os.path.exists(OFFSET_COMMENTS_FILE):
        with open(OFFSET_COMMENTS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_comments_offset(offset):
    with open(OFFSET_COMMENTS_FILE, "w") as f:
        f.write(str(offset))
    logger.info(f"💾 Сохранён offset комментариев: {offset}")

def get_creator_offset():
    if os.path.exists(OFFSET_CREATOR_FILE):
        with open(OFFSET_CREATOR_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_creator_offset(offset):
    with open(OFFSET_CREATOR_FILE, "w") as f:
        f.write(str(offset))
    logger.info(f"💾 Сохранён offset создателя: {offset}")

# ===================== УДАЛЕНИЕ WEBHOOK =====================
def delete_webhook():
    if not TG_BOT_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/deleteWebhook"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("✅ Webhook удален.")
        else:
            logger.warning(f"⚠️ Не удалось удалить webhook: {resp.text}")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка при удалении webhook: {e}")

# ===================== НОВОСТИ =====================
def parse_rss_date(entry):
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        return datetime.fromtimestamp(time.mktime(entry.published_parsed))
    if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
        return datetime.fromtimestamp(time.mktime(entry.updated_parsed))
    return None

def is_fresh_news(entry):
    pub_date = parse_rss_date(entry)
    if pub_date is None:
        return False
    now = datetime.now()
    age = now - pub_date
    return age.days <= MAX_NEWS_AGE_DAYS

def calculate_priority(news_item):
    source = news_item.get('source', '')
    source_priority = SOURCE_PRIORITY.get(source, 10)
    text = (news_item['title'] + ' ' + news_item['summary']).lower()
    keyword_boost = 0
    for kw in KEYWORDS:
        if kw in text:
            keyword_boost -= 1
    age_hours = news_item.get('age_hours', 0)
    freshness_priority = age_hours * 0.5
    return source_priority + keyword_boost + freshness_priority

def fetch_fresh_news(limit: int = 5):
    logger.info("📰 Запрашиваю свежие новости...")
    news_sources = [
        {"name": "Habr", "url": "https://habr.com/ru/rss/articles/?fl=ru"},
        {"name": "SecurityLab", "url": "https://www.securitylab.ru/_services/export/rss/vulnerabilities/"},
        {"name": "3DNews", "url": "https://3dnews.ru/news/rss/"},
        {"name": "CNews", "url": "https://www.cnews.ru/inc/rss/news.xml"},
        {"name": "iXBT", "url": "https://www.ixbt.com/export/news.rss"},
        {"name": "Ferra", "url": "https://www.ferra.ru/exports/rss.xml"},
        {"name": "vc.ru", "url": "https://vc.ru/rss/"},
        {"name": "Kod", "url": "https://kod.ru/rss/"},
        {"name": "Overclockers", "url": "https://overclockers.ru/rss/news.rss"},
    ]
    all_candidates = []
    for source in news_sources:
        try:
            logger.info(f"    - {source['name']}...")
            feed = feedparser.parse(source['url'])
            for entry in feed.entries[:30]:
                if not is_fresh_news(entry):
                    continue
                link = entry.get('link')
                if not link or is_news_already_published(link):
                    continue
                title = entry.get('title', 'Без заголовка')
                summary = re.sub('<[^<]+?>', '', entry.get('summary', ''))[:500]
                pub_date = parse_rss_date(entry)
                if pub_date is None:
                    continue
                age_hours = (datetime.now() - pub_date).total_seconds() / 3600
                all_candidates.append({
                    'source': source['name'],
                    'title': title,
                    'link': link,
                    'summary': summary,
                    'pub_date': pub_date,
                    'age_hours': age_hours,
                })
        except Exception as e:
            logger.warning(f"    ⚠️ Ошибка {source['name']}: {e}")
    if not all_candidates:
        logger.warning("⚠️ Новостей нет.")
        return []
    all_candidates.sort(key=calculate_priority)
    top_news = all_candidates[:limit]
    logger.info(f"✅ Отобрано {len(top_news)} новостей.")
    return top_news

# ===================== ПОИСК И ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ =====================
def search_pexels_image(query: str) -> bytes | None:
    if not PEXELS_API_KEY:
        logger.warning("⚠️ PEXELS_API_KEY не задан.")
        return None
    logger.info(f"🔍 Ищу изображение на Pexels: {query[:80]}...")
    search_query = ' '.join(query.split()[:5])
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": search_query, "per_page": 5, "orientation": "landscape"}
    try:
        resp = http_session.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            photos = resp.json().get("photos", [])
            if photos:
                img_url = photos[0]["src"]["large"]
                img_resp = http_session.get(img_url, timeout=30)
                if img_resp.status_code == 200:
                    logger.info("✅ Изображение найдено.")
                    return img_resp.content
        else:
            logger.warning(f"⚠️ Ошибка Pexels: {resp.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Исключение: {e}")
    return None

def generate_image(prompt: str) -> bytes | None:
    logger.info("🎨 Генерация изображения через AI...")
    if not HF_API_TOKEN:
        return None
    enhanced_prompt = f"Create a realistic cover image for news: {prompt}. No text."
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    for model in IMAGE_MODELS:
        try:
            logger.info(f"🔄 Пробую модель {model}...")
            url = f"https://api-inference.huggingface.co/models/{model}"
            resp = requests.post(url, headers=headers, json={"inputs": enhanced_prompt}, timeout=60)
            if resp.status_code == 200:
                logger.info(f"✅ Изображение сгенерировано.")
                return resp.content
        except Exception as e:
            logger.warning(f"⚠️ Ошибка: {e}")
    return None

# ===================== ОТПРАВКА В TELEGRAM =====================
async def send_telegram_photo(chat_id: int, photo_bytes: bytes, caption: str, bot: Bot) -> tuple[bool, int | None]:
    MAX_CAPTION = 1024
    if len(caption) <= MAX_CAPTION:
        try:
            msg = await bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=caption, parse_mode="MarkdownV2")
            return True, msg.message_id
        except TelegramError as e:
            if "can't parse entities" in str(e).lower():
                logger.warning("Ошибка Markdown, пробую без форматирования")
                try:
                    msg = await bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=caption)
                    return True, msg.message_id
                except Exception as e2:
                    logger.error(f"Ошибка: {e2}")
                    return False, None
            else:
                logger.error(f"Ошибка: {e}")
                return False, None
    else:
        logger.info(f"Подпись длинная, отправляю отдельно.")
        try:
            msg_photo = await bot.send_photo(chat_id=chat_id, photo=photo_bytes)
            success, _ = await send_telegram_message(chat_id, caption, bot)
            return success, msg_photo.message_id
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return False, None

async def send_telegram_message(chat_id: int, text: str, bot: Bot) -> tuple[bool, int | None]:
    try:
        msg = await bot.send_message(chat_id=chat_id, text=text, parse_mode="MarkdownV2")
        return True, msg.message_id
    except TelegramError as e:
        if "can't parse entities" in str(e).lower():
            logger.warning("Ошибка Markdown, отправляю без форматирования")
            try:
                msg = await bot.send_message(chat_id=chat_id, text=text)
                return True, msg.message_id
            except Exception as e2:
                logger.error(f"Ошибка: {e2}")
                return False, None
        else:
            logger.error(f"Ошибка: {e}")
            return False, None

# ===================== ГЕНЕРАЦИЯ ТЕКСТА =====================
def twelver_generate(prompt: str, max_tokens: int = 800) -> str | None:
    api_key = os.getenv("TWELVER_API_KEY")
    if not api_key:
        logger.warning("⚠️ TWELVER_API_KEY не найден.")
        return None
    base_url = "https://twelver.ru/api/ai/c7e9d747-ffbe-4027-9c48-8541c5302b72/v1"
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Twelver error: {e}")
        return None

def dandolo_generate(prompt: str, max_tokens: int = 800) -> str | None:
    url = "https://api.dandolo.ai/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Dandolo error: {e}")
    return None

def unofficial_openai_generate(prompt: str, max_tokens: int = 800) -> str | None:
    url = "https://devsdocode-openai.hf.space/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Unofficial error: {e}")
    return None

def algion_generate(prompt: str, max_tokens: int = 800) -> str | None:
    api_key = os.getenv("ALGION_API_KEY")
    if not api_key:
        return None
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.algion.dev/v1")
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Algion error: {e}")
    return None

def devtoolbox_generate(prompt: str, max_tokens: int = 800) -> str | None:
    url = "https://api.devtoolbox.io/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "devtoolbox-7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"DevToolbox error: {e}")
    return None

ai_client = None

def generate_with_groq(prompt: str, max_tokens: int = 800) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        for model in GROQ_MODELS:
            try:
                logger.info(f"🔄 Пробую Groq: {model}")
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.8,
                )
                result = completion.choices[0].message.content.strip()
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Ошибка {model}: {e}")
                continue
    except Exception as e:
        logger.error(f"Groq ошибка: {e}")
    return None

def generate_with_hf(prompt: str, max_tokens: int = 800) -> str | None:
    global ai_client
    if not HF_API_TOKEN:
        return None
    if ai_client is None:
        ai_client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
    for model in FALLBACK_MODELS:
        try:
            logger.info(f"🔄 Пробую HF: {model}")
            completion = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.8,
            )
            result = completion.choices[0].message.content.strip()
            if result:
                return result
        except Exception as e:
            error_msg = str(e)
            if "402" in error_msg:
                logger.warning(f"Модель {model} платная, пропускаем.")
            else:
                logger.warning(f"Ошибка {model}: {e}")
            continue
    return None

PROVIDER_FUNCTIONS = {
    "twelver": twelver_generate,
    "groq": generate_with_groq,
    "huggingface": generate_with_hf,
    "dandolo": dandolo_generate,
    "unofficial": unofficial_openai_generate,
    "algion": algion_generate,
    "devtoolbox": devtoolbox_generate,
}

PROVIDER_ORDER = os.getenv("PROVIDER_ORDER", "twelver,groq,huggingface,dandolo,unofficial,algion,devtoolbox").split(",")

def generate_with_fallback(prompt: str, max_tokens: int = 800) -> str | None:
    for name in PROVIDER_ORDER:
        func = PROVIDER_FUNCTIONS.get(name.strip())
        if not func:
            continue
        logger.info(f"🔄 Пробую провайдер: {name}")
        try:
            result = func(prompt, max_tokens)
            if result:
                logger.info(f"✅ Успех через {name}")
                return result
        except Exception as e:
            logger.warning(f"❌ {name}: {e}")
    logger.error("❌ Все провайдеры недоступны.")
    return None

def generate_quiz_question(news_item: dict) -> dict | None:
    prompt = f"""
На основе новости составь ВОПРОС и 4 варианта ответа (один правильный) для викторины.
Ответ должен быть СТРОГО в JSON формате:
{{"question": "...", "options": ["...", "..."], "correct_option_id": 0}}
Вопрос должен быть интересным, варианты - правдоподобными.

Новость:
Источник: {news_item['source']}
Заголовок: {news_item['title']}
Описание: {news_item['summary']}
"""
    result = generate_with_fallback(prompt, max_tokens=300)
    if not result:
        return None

    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if not json_match:
        logger.warning(f"Не удалось найти JSON: {result}")
        return None

    try:
        data = json.loads(json_match.group())
        if not all(k in data for k in ('question', 'options', 'correct_option_id')):
            return None
        if len(data['options']) < 2:
            return None
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"Ошибка парсинга JSON: {e}")
        return None

def fix_links(text: str) -> str:
    fixed = re.sub(r'\[([^\]]+)\]\s*\(?\s*(https?://[^\s\)]+)\)?', r'[\1](\2)', text)
    return fixed

def generate_post(news_list):
    logger.info("🤖 Генерирую пост...")
    if not PROVIDER_ORDER:
        return "❌ Ошибка: нет настроенных провайдеров"

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
    result = generate_with_fallback(prompt, max_tokens=800)
    print(result)
    if result:
        result = fix_links(result)
        return result
    return "❌ Не удалось сгенерировать пост."

def generate_reply(comment_text: str, post_content: str) -> str:
    prompt = f"""
Ты - автор IT-канала. Подписчик: "{comment_text}"
Пост был о: {post_content[:500]}
Ответь дружелюбно, с юмором (2-3 предложения). Ты не любишь цифровой контроль и мессенджер Max (от VK), но не выражаешь это открыто - только лёгкий сарказм в ответах на комментарии к этой теме.
"""
    result = generate_with_fallback(prompt, max_tokens=200)
    return result if result else "Спасибо за комментарий! 👍 Обязательно отвечу подробнее позже."

# ===================== ПРОВЕРКА КОММЕНТАРИЕВ =====================
async def check_and_reply_to_comments(bot: Bot):
    logger.info("💬 Проверяю комментарии...")
    offset = get_comments_offset()
    try:
        updates = await bot.get_updates(offset=offset, limit=100, timeout=10, allowed_updates=["message"])
    except Exception as e:
        logger.error(f"Ошибка получения обновлений: {e}")
        return

    max_id = offset - 1
    processed_count = 0
    duplicates_to_delete = []

    for update in updates:
        if update.update_id > max_id:
            max_id = update.update_id

        msg = update.message
        if not msg:
            continue

        if COMMENTS_CHAT_ID and msg.chat_id != COMMENTS_CHAT_ID:
            continue

        if msg.sender_chat and msg.sender_chat.id == int(TG_CHANNEL_ID) and msg.photo:
            duplicates_to_delete.append(msg.message_id)
            logger.info(f"🔍 Дубликат (msg_id={msg.message_id})")
            continue

        reply_to = msg.reply_to_message
        if not reply_to:
            continue

        tg_channel_id = int(TG_CHANNEL_ID)
        is_our_post = False
        if reply_to.sender_chat and reply_to.sender_chat.id == tg_channel_id:
            is_our_post = True
        elif reply_to.from_user and reply_to.from_user.id == bot.id:
            is_our_post = True

        if not is_our_post:
            continue

        post_content = reply_to.caption or reply_to.text or ""
        if not post_content:
            continue

        comment_id = msg.message_id
        if is_comment_processed(comment_id):
            continue

        comment_text = msg.text or msg.caption
        if not comment_text:
            continue

        username = msg.from_user.username or msg.from_user.first_name
        logger.info(f"📝 Комментарий от @{username}: {comment_text[:50]}...")

        reply_text = generate_reply(comment_text, post_content)
        if reply_text:
            try:
                await bot.send_message(
                    chat_id=msg.chat_id,
                    text=reply_text,
                    reply_to_message_id=comment_id,
                    parse_mode="MarkdownV2"
                )
                mark_comment_processed(comment_id, 0)
                processed_count += 1
            except Exception as e:
                if "can't parse entities" in str(e).lower():
                    try:
                        await bot.send_message(
                            chat_id=msg.chat_id,
                            text=reply_text,
                            reply_to_message_id=comment_id
                        )
                        mark_comment_processed(comment_id, 0)
                        processed_count += 1
                    except Exception as e2:
                        logger.error(f"Ошибка: {e2}")

    for dup_id in duplicates_to_delete:
        try:
            await bot.delete_message(chat_id=COMMENTS_CHAT_ID, message_id=dup_id)
            logger.info(f"🗑️ Удалён дубликат {dup_id}")
        except Exception as e:
            logger.warning(f"Не удалось удалить {dup_id}: {e}")

    if max_id >= offset:
        save_comments_offset(max_id + 1)

    logger.info(f"Обработано комментариев: {processed_count}")

# ===================== КОМАНДЫ СОЗДАТЕЛЯ =====================
async def check_creator_messages(bot: Bot):
    logger.info("👤 Проверяю сообщения создателя...")
    offset = get_creator_offset()
    try:
        updates = await bot.get_updates(offset=offset, timeout=30, allowed_updates=["message"])
        max_id = offset - 1
        for update in updates:
            if update.update_id > max_id:
                max_id = update.update_id
            msg = update.message
            if not msg:
                continue
            if msg.from_user.id == CREATOR_ID:
                text = msg.text
                if text.startswith("/generate"):
                    await publish_new_post(bot)
                elif text.startswith("/stats"):
                    posts = get_recent_posts(10)
                    stats = f"📊 Постов: {len(posts)}"
                    await bot.send_message(chat_id=CREATOR_ID, text=stats)
                else:
                    await bot.send_message(chat_id=CREATOR_ID, text=f"✅ Получено: {text}")
        if max_id >= offset:
            save_creator_offset(max_id + 1)
    except Exception as e:
        logger.error(f"Ошибка: {e}")

# ===================== ПУБЛИКАЦИЯ ПОСТА =====================
async def maybe_create_quiz(bot: Bot, news_list: list, main_news: dict):
    import random
    if random.random() > QUIZ_PROBABILITY:
        return
    if not TG_GROUP_ID:
        return

    candidates = [n for n in news_list if n['link'] != main_news['link']]
    if not candidates:
        candidates = news_list

    selected = random.choice(candidates)
    logger.info(f"🎲 Квиз по: {selected['title'][:80]}")

    quiz_data = generate_quiz_question(selected)
    if not quiz_data:
        logger.warning("Не удалось сгенерировать квиз.")
        return

    try:
        await bot.send_poll(
            chat_id=int(TG_CHANNEL_ID),
            question=quiz_data['question'],
            options=quiz_data['options'],
            type="quiz",
            correct_option_id=quiz_data['correct_option_id'],
            is_anonymous=True,
        )
        logger.info("📊 Квиз отправлен.")
    except Exception as e:
        logger.warning(f"Ошибка квиза: {e}")

async def publish_new_post(bot: Bot):
    logger.info("📝 Публикация...")
    try:
        news_list = fetch_fresh_news(limit=5)
        if not news_list:
            await bot.send_message(chat_id=CREATOR_ID, text="❌ Нет новостей.")
            return
        top_news = news_list[0]
        post_content = generate_post(news_list)
        if not post_content or post_content.startswith("❌"):
            await bot.send_message(chat_id=CREATOR_ID, text=f"❌ Ошибка: {post_content}")
            return
        image = search_pexels_image(top_news['title'])
        if not image and HF_API_TOKEN:
            image = generate_image(f"{top_news['title']} {top_news['summary']}")
        if image:
            success, msg_id = await send_telegram_photo(int(TG_CHANNEL_ID), image, post_content, bot)
            if success and msg_id:
                save_post(msg_id, post_content)
                for news in news_list:
                    save_published_news(news['link'], news['title'])
                await bot.send_message(chat_id=CREATOR_ID, text="Пост опубликован")
                await maybe_create_quiz(bot, news_list, top_news)
        else:
            success, msg_id = await send_telegram_message(int(TG_CHANNEL_ID), post_content, bot)
            if success and msg_id:
                save_post(msg_id, post_content)
                for news in news_list:
                    save_published_news(news['link'], news['title'])
                await bot.send_message(chat_id=CREATOR_ID, text="Пост опубликован (без фото)")
                await maybe_create_quiz(bot, news_list, top_news)
    except Exception as e:
        error_msg = f"Критическая ошибка: {str(e)}"
        logger.error(error_msg)
        await bot.send_message(chat_id=CREATOR_ID, text=error_msg)

# ===================== MAIN =====================
async def run_all(bot: Bot):
    logger.info("🔄 Выполняю все задачи")
    await check_and_reply_to_comments(bot)
    await check_creator_messages(bot)
    await publish_new_post(bot)
    await check_and_reply_to_comments(bot)

async def run_check(bot: Bot):
    await check_and_reply_to_comments(bot)
    await check_creator_messages(bot)

def main():
    logger.info("🚀 Запуск бота...")
    delete_webhook()
    init_db()
    clean_old_news()

    bot = Bot(token=TG_BOT_TOKEN)

    if TEST_MODE:
        logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ")
        asyncio.run(run_all(bot))
        return

    now = datetime.now(IZHEVSK_TZ)
    hour = now.hour
    logger.info(f"🕐 Время: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if hour in [9, 10, 12, 14, 16, 18]:
        logger.info(f"⏰ {hour}:00 - публикую пост")
        asyncio.run(run_all(bot))
    else:
        logger.info(f"🕐 {hour}:00 - проверка комментариев")
        asyncio.run(run_check(bot))

    logger.info("✅ Работа завершена")

if __name__ == "__main__":
    main()
