import os
import requests
import json
import re

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# URL роутера Hugging Face (OpenAI-совместимый)
HF_API_URL = "https://router.huggingface.co/hf-inference/v1"

# ✅ Используем модель, которая точно доступна через бесплатный Inference API
#    (у неё в списке провайдеров есть "hf-inference")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct:hyperbolic"

PROMPT = """
Ты — опытный Python-разработчик и энтузиаст IT. 
Напиши короткий, увлекательный и полезный пост для Telegram-канала о программировании.
Тема: интересные факты о Python, советы по коду или новости технологий.
Стиль: дружелюбный, профессиональный, с эмодзи.
Язык: Русский.
Объем: до 1000 символов.
"""

def escape_markdown(text: str) -> str:
    """Экранирует спецсимволы MarkdownV2 для Telegram."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def generate_content():
    """Генерирует пост через HF Inference API. Возвращает (success, content_or_error)."""
    if not HF_API_TOKEN:
        return False, "❌ Ошибка: не указан токен Hugging Face."

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": 400,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"].strip()
                if generated_text:
                    return True, generated_text
                else:
                    return False, "⚠️ Модель вернула пустой ответ."
            else:
                return False, f"⚠️ Неожиданный формат ответа: {result}"
        else:
            error_text = response.text[:200]
            if response.status_code == 404:
                return False, f"❌ Модель '{MODEL_NAME}' не найдена или недоступна. Проверьте имя модели."
            elif response.status_code == 403:
                return False, "❌ Доступ запрещён. Проверьте токен Hugging Face."
            elif response.status_code == 429:
                return False, "⚠️ Слишком много запросов. Попробуйте позже."
            else:
                return False, f"❌ Ошибка API HF ({response.status_code}): {error_text}"

    except requests.exceptions.Timeout:
        return False, "❌ Таймаут соединения с Hugging Face."
    except requests.exceptions.ConnectionError:
        return False, "❌ Ошибка соединения с Hugging Face."
    except json.JSONDecodeError:
        return False, "❌ Неверный JSON-ответ от сервера."
    except Exception as e:
        return False, f"❌ Неизвестная ошибка: {str(e)}"

def send_to_telegram(text, is_error=False):
    """
    Отправляет сообщение в Telegram.
    Если is_error=True – отправляет как обычный текст (без Markdown).
    """
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("❌ Ошибка: Не указаны токен бота или ID чата.")
        return False

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    
    if is_error:
        # Для ошибок – просто текст, без parse_mode
        payload = {
            "chat_id": TG_CHAT_ID,
            "text": text
        }
        # Убираем parse_mode, если он не нужен (не передаём поле)
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                print("✅ Сообщение об ошибке отправлено в Telegram.")
                return True
            else:
                print(f"❌ Не удалось отправить ошибку в Telegram: {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Ошибка при отправке ошибки: {str(e)}")
            return False
    else:
        # Для обычного поста – экранируем Markdown
        safe_text = escape_markdown(text)
        payload = {
            "chat_id": TG_CHAT_ID,
            "text": safe_text,
            "parse_mode": "MarkdownV2"
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
                    payload.pop("parse_mode", None)  # убираем parse_mode
                    payload["text"] = text          # исходный текст
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
        if send_to_telegram(content, is_error=False):
            print("[PUBLISH_SUCCESS] Пост отправлен в Telegram.")
        else:
            print("[PUBLISH_FAIL] Не удалось отправить пост.")
            # Дополнительно пробуем отправить в Telegram сообщение о сбое отправки
            send_to_telegram("⚠️ Не удалось опубликовать сгенерированный пост (ошибка отправки).", is_error=True)
    else:
        # Генерация не удалась – отправляем ошибку в Telegram
        error_message = f"❌ *Ошибка генерации поста*\n\n{content}"
        print(f"[PUBLISH_FAIL] Генерация не удалась: {content}")
        if send_to_telegram(error_message, is_error=True):
            print("[PUBLISH_FAIL] Сообщение об ошибке доставлено в Telegram.")
        else:
            print("[PUBLISH_FAIL] Не удалось даже отправить сообщение об ошибке.")
