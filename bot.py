import os
import requests
import json
import re
from huggingface_hub import InferenceClient

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# Упрощённый промпт для лучшей генерации
PROMPT = """
Ты — опытный Python-разработчик и энтузиаст IT. 
Напиши короткий, увлекательный и полезный пост для Telegram-канала о программировании.
Тема: интересные факты о Python, советы по коду или новости технологий.
Стиль: дружелюбный, профессиональный, с эмодзи.
Язык: Русский.
Объем: до 1000 символов.
"""

def escape_markdown(text: str) -> str:
    """Экранирует спецсимволы для Telegram MarkdownV2."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def generate_content():
    """Генерирует пост через InferenceClient."""
    if not HF_API_TOKEN:
        return False, "❌ Ошибка: Не указан токен Hugging Face."

    try:
        # Инициализируем клиент. Параметр provider="auto" — ключевой!
        # Он позволит API самому выбрать лучшего провайдера для модели[reference:3].
        client = InferenceClient(
            api_key=HF_API_TOKEN,
            provider="auto"
        )
        
        # Отправляем запрос на генерацию. 
        # Клиент сам сформирует правильный URL и тело запроса.
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",  # Надёжная, проверенная модель
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=400,
            temperature=0.7,
        )
        
        # Извлекаем сгенерированный текст
        generated_text = completion.choices[0].message.content.strip()
        if generated_text:
            return True, generated_text
        else:
            return False, "⚠️ Модель вернула пустой ответ."

    except Exception as e:
        return False, f"❌ Ошибка при генерации: {str(e)}"

def send_to_telegram(text: str, is_error: bool = False) -> bool:
    """Отправляет сообщение в Telegram."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("❌ Ошибка: Не указаны токен бота или ID чата.")
        return False

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    if is_error:
        # Для ошибок отправляем как обычный текст
        payload = {"chat_id": TG_CHAT_ID, "text": text}
    else:
        # Для поста экранируем Markdown
        safe_text = escape_markdown(text)
        payload = {"chat_id": TG_CHAT_ID, "text": safe_text, "parse_mode": "MarkdownV2"}

    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            print("✅ Сообщение отправлено в Telegram.")
            return True
        # Если не получилось с Markdown, пробуем без него
        elif resp.status_code == 400 and "can't parse entities" in resp.text and not is_error:
            payload_no_md = {"chat_id": TG_CHAT_ID, "text": text}
            resp2 = requests.post(url, json=payload_no_md, timeout=30)
            if resp2.status_code == 200:
                print("✅ Пост отправлен без форматирования.")
                return True
        else:
            print(f"❌ Ошибка Telegram: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка при отправке: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")
    
    success, content = generate_content()
    
    if success:
        if send_to_telegram(content):
            print("[PUBLISH_SUCCESS] Пост успешно отправлен.")
        else:
            send_to_telegram("⚠️ Пост создан, но не отправлен в Telegram.", is_error=True)
    else:
        print(f"[PUBLISH_FAIL] {content}")
        send_to_telegram(content, is_error=True)
