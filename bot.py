import os
import requests
import json
import re

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# Новый URL для роутера Hugging Face (совместим с OpenAI API)
HF_API_URL = "https://router.huggingface.co/hf-inference/v1"

PROMPT = """
Ты — опытный Python-разработчик и энтузиаст IT. 
Напиши короткий, увлекательный и полезный пост для Telegram-канала о программировании.
Тема: интересные факты о Python, советы по коду или новости технологий.
Стиль: дружелюбный, профессиональный, с эмодзи.
Язык: Русский.
Объем: до 1000 символов.
"""

def escape_markdown(text: str) -> str:
    """Экранирует специальные символы Markdown для Telegram."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def generate_content():
    if not HF_API_TOKEN:
        return None, "❌ Ошибка: Не указан токен Hugging Face."
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "IlyaGusev/saiga_llama3_8b",   # можно заменить на другую модель
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Новый формат ответа: { choices: [{ message: { content: ... } }] }
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"].strip()
                if generated_text:
                    return True, generated_text
                else:
                    return False, "⚠️ Модель вернула пустой ответ."
            else:
                return False, f"⚠️ Неожиданный формат ответа: {result}"
        else:
            # Обработка ошибок нового API
            error_msg = response.text[:200]
            if response.status_code == 410:
                return False, "❌ API устарел. Убедитесь, что используется router.huggingface.co"
            elif response.status_code == 403:
                return False, "❌ Доступ запрещён. Проверьте токен Hugging Face."
            elif response.status_code == 404:
                return False, "❌ Модель не найдена или недоступна на новом API."
            else:
                return False, f"Ошибка API HF ({response.status_code}): {error_msg}"
            
    except requests.exceptions.Timeout:
        return False, "❌ Таймаут соединения с Hugging Face."
    except requests.exceptions.ConnectionError:
        return False, "❌ Ошибка соединения с Hugging Face."
    except json.JSONDecodeError:
        return False, "❌ Неверный JSON-ответ от сервера."
    except Exception as e:
        return False, f"❌ Неизвестная ошибка: {str(e)}"

def send_to_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("❌ Ошибка: Не указаны токен бота или ID чата.")
        return False
    
    # Экранируем текст для безопасной отправки в Markdown
    safe_text = escape_markdown(text)
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": safe_text,
        "parse_mode": "MarkdownV2"   # используем более новую версию Markdown
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            print("✅ Пост успешно опубликован в канале!")
            return True
        else:
            # Если экранирование не помогло, пробуем отправить без форматирования
            if resp.status_code == 400 and "can't parse entities" in resp.text:
                print("⚠️ Ошибка парсинга Markdown, отправляем без форматирования...")
                payload["parse_mode"] = None
                payload["text"] = text  # исходный текст
                resp2 = requests.post(url, json=payload, timeout=30)
                if resp2.status_code == 200:
                    print("✅ Пост отправлен без форматирования.")
                    return True
                else:
                    print(f"❌ Ошибка повторной отправки: {resp2.text}")
                    return False
            else:
                print(f"❌ Ошибка отправки в Telegram: {resp.text}")
                return False
    except Exception as e:
        print(f"❌ Ошибка при отправке: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")
    
    success, content = generate_content()
    
    if success:
        print(f"📝 Сгенерированный текст:\n{content}\n")
        if send_to_telegram(content):
            print("[PUBLISH_SUCCESS] Post sent to Telegram.")
        else:
            print("[PUBLISH_FAIL] Failed to send post.")
    else:
        print(f"[PUBLISH_FAIL] Content generation failed: {content}")
