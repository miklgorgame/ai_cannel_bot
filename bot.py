import os
import re
import requests
from huggingface_hub import InferenceClient

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

PROMPT = """
Ты — опытный Python-разработчик и энтузиаст IT. 
Напиши короткий, увлекательный и полезный пост для Telegram-канала о программировании.
Тема: интересные факты о Python, советы по коду или новости технологий.
Стиль: дружелюбный, профессиональный, с эмодзи.
Язык: Русский.
Объем: до 1000 символов.
"""

def escape_markdown(text: str) -> str:
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def generate_post():
    if not HF_API_TOKEN:
        return None, "❌ Нет токена Hugging Face"
    try:
        client = InferenceClient(api_key=HF_API_TOKEN, provider="auto")
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=400,
            temperature=0.7,
        )
        text = completion.choices[0].message.content.strip()
        if text:
            return text, None
        else:
            return None, "⚠️ Модель вернула пустой ответ"
    except Exception as e:
        return None, f"❌ Ошибка генерации: {str(e)}"

def send_to_telegram(text, is_error=False):
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
            print("✅ Сообщение отправлено")
            return True
        if not is_error and "can't parse entities" in resp.text:
            # fallback без форматирования
            payload2 = {"chat_id": TG_CHAT_ID, "text": text}
            resp2 = requests.post(url, json=payload2, timeout=30)
            return resp2.status_code == 200
        print(f"❌ Ошибка Telegram: {resp.text}")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск генерации...")
    post, error = generate_post()
    if post:
        if send_to_telegram(post):
            print("[PUBLISH_SUCCESS] Пост отправлен")
        else:
            send_to_telegram("⚠️ Пост создан, но не отправлен", is_error=True)
    else:
        print(f"[PUBLISH_FAIL] {error}")
        send_to_telegram(error, is_error=True)
