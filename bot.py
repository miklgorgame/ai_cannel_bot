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

TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

DB_FILE = "bot_memory.db"
OFFSET_COMMENTS_FILE = "offset_comments.txt"
OFFSET_CREATOR_FILE = "offset_creator.txt"
IZHEVSK_TZ = ZoneInfo("Europe/Samara")

COMMENTS_CHAT_ID = TG_GROUP_ID if TG_GROUP_ID else TG_CHANNEL_ID
MAX_NEWS_AGE_DAYS = 2

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

# ===================== HTTP СЕССИЯ С RETRY =====================
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
    return c.fetchone() is not None

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

# ===================== OFFSET ДЛЯ КОММЕНТАРИЕВ И КОМАНД =====================
def get_comments_offset():
    if os.path.exists(OFFSET_COMMENTS_FILE):
        with open(OFFSET_COMMENTS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_comments_offset(offset):
    with open(OFFSET_COMMENTS_FILE, "w") as f:
        f.write(str(offset))

def get_creator_offset():
    if os.path.exists(OFFSET_CREATOR_FILE):
        with open(OFFSET_CREATOR_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_creator_offset(offset):
    with open(OFFSET_CREATOR_FILE, "w") as f:
        f.write(str(offset))

# ===================== НОВОСТИ С ПРАВИЛЬНЫМИ RSS =====================
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
    return source_priority + keyword_boost

def fetch_fresh_news(limit: int = 5):
    logger.info("📰 Запрашиваю свежие новости...")
    # Правильные RSS-ссылки
    news_sources = [
        {"name": "Habr", "url": "https://habr.com/ru/rss/articles/?fl=ru"},
        {"name": "3DNews", "url": "https://3dnews.ru/news/rss/"},
        {"name": "CNews", "url": "https://www.cnews.ru/inc/rss/news.xml"},
        {"name": "iXBT", "url": "https://www.ixbt.com/export/news.rss"},
        {"name": "Ferra", "url": "https://www.ferra.ru/exports/rss.xml"},
        {"name": "SecurityLab", "url": "https://www.securitylab.ru/_services/export/rss/vulnerabilities/"},
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
                all_candidates.append({
                    'source': source['name'],
                    'title': title,
                    'link': link,
                    'summary': summary
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
        logger.warning("⚠️ PEXELS_API_KEY не задан, пропускаем поиск.")
        return None
    logger.info(f"🔍 Ищу изображение на Pexels по запросу: {query[:80]}...")
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
                    logger.info("✅ Изображение найдено на Pexels.")
                    return img_resp.content
                else:
                    logger.warning(f"⚠️ Не удалось скачать изображение: {img_resp.status_code}")
            else:
                logger.warning("⚠️ На Pexels ничего не найдено.")
        else:
            logger.warning(f"⚠️ Ошибка Pexels API: {resp.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Исключение при поиске на Pexels: {e}")
    return None

def generate_image(prompt: str) -> bytes | None:
    logger.info("🎨 Резервная генерация изображения через AI...")
    if not HF_API_TOKEN:
        logger.warning("⚠️ Нет токена HF для генерации изображения.")
        return None
    enhanced_prompt = f"Create a realistic cover image for news: {prompt}. No text."
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    for model in IMAGE_MODELS:
        try:
            logger.info(f"🔄 Пробую модель {model}...")
            url = f"https://api-inference.huggingface.co/models/{model}"
            resp = requests.post(url, headers=headers, json={"inputs": enhanced_prompt}, timeout=60)
            if resp.status_code == 200:
                logger.info(f"✅ Изображение сгенерировано через {model}.")
                return resp.content
            else:
                logger.warning(f"⚠️ Модель {model} вернула {resp.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка с моделью {model}: {e}")
    return None

# ===================== АСИНХРОННЫЕ ФУНКЦИИ TELEGRAM =====================
async def send_telegram_photo(chat_id: int, photo_bytes: bytes, caption: str, bot: Bot) -> tuple[bool, int | None]:
    MAX_CAPTION = 1024
    if len(caption) <= MAX_CAPTION:
        try:
            msg = await bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=caption, parse_mode="HTML")
            return True, msg.message_id
        except TelegramError as e:
            logger.error(f"Ошибка отправки фото: {e}")
            return False, None
    else:
        logger.info(f"Подпись длиной {len(caption)} превышает лимит, отправляю фото без подписи, текст отдельно.")
        try:
            msg_photo = await bot.send_photo(chat_id=chat_id, photo=photo_bytes)
            await bot.send_message(chat_id=chat_id, text=caption, parse_mode="HTML")
            return True, msg_photo.message_id
        except TelegramError as e:
            logger.error(f"Ошибка отправки: {e}")
            return False, None

async def send_telegram_message(chat_id: int, text: str, bot: Bot) -> tuple[bool, int | None]:
    try:
        msg = await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
        return True, msg.message_id
    except TelegramError as e:
        logger.error(f"Ошибка отправки сообщения: {e}")
        return False, None

async def delete_duplicate_from_group(photo_message_id: int, post_text: str, bot: Bot):
    if not TG_GROUP_ID:
        return
    logger.info(f"🔍 Ищу дубликат фото (message_id={photo_message_id}) в группе {TG_GROUP_ID}...")
    await asyncio.sleep(5)
    offset = get_comments_offset()
    try:
        updates = await bot.get_updates(offset=offset, timeout=5, allowed_updates=["message"])
        max_id = offset - 1
        for update in updates:
            if update.update_id > max_id:
                max_id = update.update_id
            msg = update.message
            if not msg:
                continue
            if msg.chat_id != int(TG_GROUP_ID):
                continue
            if msg.photo and not msg.reply_to_message:
                caption = msg.caption or ""
                if caption and post_text.startswith(caption[:50]):
                    logger.info(f"Найден дубликат фото в группе, message_id={msg.message_id}")
                    try:
                        await bot.delete_message(chat_id=TG_GROUP_ID, message_id=msg.message_id)
                        logger.info("✅ Дубликат фото удалён из группы обсуждения.")
                    except TelegramError as e:
                        logger.warning(f"Не удалось удалить дубликат: {e}")
                    break
        if max_id >= offset:
            save_comments_offset(max_id + 1)
    except Exception as e:
        logger.warning(f"Ошибка при удалении дубликата: {e}")

# ===================== ГЕНЕРАЦИЯ ТЕКСТА =====================
ai_client = None

def generate_post(news_list):
    global ai_client
    logger.info("🤖 Генерирую пост...")
    if not HF_API_TOKEN:
        return "❌ Ошибка: нет токена HF"
    if ai_client is None:
        ai_client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
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
            logger.info(f"🔄 Пробую модель: {model}")
            completion = ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.8,
            )
            result = completion.choices[0].message.content.strip()
            if result:
                logger.info(f"✅ Успешно сгенерировано с помощью {model}")
                return result
        except Exception as e:
            logger.warning(f"⚠️ Ошибка с моделью {model}: {e}")
            continue
    return "❌ Не удалось сгенерировать пост."

def generate_reply(comment_text: str, post_content: str) -> str:
    global ai_client
    prompt = f"""
Ты — автор IT-канала. Подписчик: "{comment_text}"
Пост был о: {post_content[:500]}
Ответь дружелюбно, с юмором (2-3 предложения). Если про Max — легкая ирония.
"""
    for model in FALLBACK_MODELS:
        try:
            if ai_client is None:
                ai_client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
            completion = ai_client.chat.completions.create(
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

# ===================== КОММЕНТАРИИ И КОМАНДЫ =====================
async def check_and_reply_to_comments(bot: Bot):
    logger.info("💬 Проверяю комментарии (включая старые)...")
    last_post = get_last_post()
    if not last_post:
        logger.info("Нет постов для проверки комментариев.")
        return
    
    # Получаем последние 100 обновлений (без offset, чтобы увидеть все)
    try:
        updates = await bot.get_updates(offset=-1, limit=100, timeout=10, allowed_updates=["message"])
    except Exception as e:
        logger.error(f"Ошибка получения обновлений: {e}")
        return
    
    logger.info(f"Получено {len(updates)} обновлений.")
    processed_count = 0
    for update in updates:
        msg = update.message
        if not msg:
            continue
        
        # Проверяем, что сообщение из нужного чата (группа или канал)
        chat_id = msg.chat_id
        if COMMENTS_CHAT_ID and str(chat_id) != str(COMMENTS_CHAT_ID):
            continue
        
        # Проверяем, что это ответ на наш пост
        reply_to = msg.reply_to_message
        if not reply_to or reply_to.message_id != last_post['message_id']:
            continue
        
        comment_id = msg.message_id
        if is_comment_processed(comment_id):
            continue
        
        text = msg.text
        if not text:
            continue
        
        logger.info(f"📝 Найден комментарий (ID={comment_id}): {text[:50]}...")
        reply = generate_reply(text, last_post['content'])
        if reply:
            await send_telegram_message(chat_id, reply, bot)
            mark_comment_processed(comment_id, last_post['id'])
            processed_count += 1
            logger.info(f"✅ Ответ отправлен на комментарий {comment_id}")
        else:
            logger.warning(f"Не удалось сгенерировать ответ на комментарий {comment_id}")
            mark_comment_processed(comment_id, last_post['id'])  # чтобы не пытаться снова
    
    logger.info(f"Обработано комментариев: {processed_count}")

async def check_creator_messages(bot: Bot):
    logger.info("👤 Проверяю сообщения создателя...")
    offset = get_creator_offset()
    try:
        updates = await bot.get_updates(offset=offset, timeout=30)
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
                    await send_telegram_message(CREATOR_ID, stats, bot)
                else:
                    await send_telegram_message(CREATOR_ID, f"✅ Получено: {text}", bot)
        if max_id >= offset:
            save_creator_offset(max_id + 1)
    except Exception as e:
        logger.error(f"Ошибка проверки команд создателя: {e}")

# ===================== ПУБЛИКАЦИЯ ПОСТА =====================
async def publish_new_post(bot: Bot):
    logger.info("📝 Публикация нового поста...")
    try:
        news_list = fetch_fresh_news(limit=5)
        if not news_list:
            await send_telegram_message(CREATOR_ID, "❌ Нет новых новостей для публикации.", bot)
            return
        top_news = news_list[0]
        logger.info(f"🏆 Главная новость: {top_news['title'][:80]}...")
        post_content = generate_post(news_list)
        if not post_content or post_content.startswith("❌"):
            await send_telegram_message(CREATOR_ID, f"❌ Ошибка генерации: {post_content}", bot)
            return
        image = search_pexels_image(top_news['title'])
        if not image and HF_API_TOKEN:
            image = generate_image(f"{top_news['title']} {top_news['summary']}")
        if image:
            success, msg_id = await send_telegram_photo(TG_CHANNEL_ID, image, post_content, bot)
            if success and msg_id:
                logger.info(f"Фото отправлено, message_id={msg_id}")
                await delete_duplicate_from_group(msg_id, post_content, bot)
                save_post(msg_id, post_content)
                for news in news_list:
                    save_published_news(news['link'], news['title'])
                await send_telegram_message(CREATOR_ID, "✅ Пост опубликован!", bot)
            else:
                await send_telegram_message(CREATOR_ID, "❌ Ошибка публикации фото.", bot)
        else:
            logger.info("⚠️ Не удалось получить изображение, отправляю только текст.")
            success, msg_id = await send_telegram_message(TG_CHANNEL_ID, post_content, bot)
            if success and msg_id:
                save_post(msg_id, post_content)
                for news in news_list:
                    save_published_news(news['link'], news['title'])
                await send_telegram_message(CREATOR_ID, "✅ Пост опубликован (без фото).", bot)
            else:
                await send_telegram_message(CREATOR_ID, "❌ Ошибка публикации текста.", bot)
    except Exception as e:
        error_msg = f"❌ Критическая ошибка: {str(e)}"
        logger.error(error_msg)
        await send_telegram_message(CREATOR_ID, error_msg, bot)

# ===================== MAIN =====================
async def run_all(bot: Bot):
    await publish_new_post(bot)
    await check_and_reply_to_comments(bot)
    await check_creator_messages(bot)

async def run_publish(bot: Bot):
    await publish_new_post(bot)

async def run_check(bot: Bot):
    await check_and_reply_to_comments(bot)
    await check_creator_messages(bot)

def main():
    logger.info("🚀 Запуск бота...")
    init_db()
    clean_old_news()
    
    bot = Bot(token=TG_BOT_TOKEN)
    
    if TEST_MODE:
        logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ")
        asyncio.run(run_all(bot))
        return
    
    now = datetime.now(IZHEVSK_TZ)
    hour = now.hour
    logger.info(f"🕐 Текущее время по Ижевску: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if hour in [9, 10, 12, 14, 16, 18]:
        logger.info(f"⏰ {hour}:00 — публикую пост!")
        asyncio.run(run_publish(bot))
    else:
        logger.info(f"🕐 {hour}:00 — проверяю комментарии и сообщения создателя")
        asyncio.run(run_check(bot))
    
    logger.info("✅ Работа завершена")

if __name__ == "__main__":
    main()
