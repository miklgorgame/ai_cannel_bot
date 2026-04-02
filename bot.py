import os
import requests
import json
import re
from typing import List, Tuple, Optional

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")   # <-- исправлено: закрывающая кавычка

# Правильные URL для роутера Hugging Face
ROUTER_MODELS_URL = "https://router.huggingface.co/v1/models"
ROUTER_CHAT_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"

WORKING_MODELS_FILE = "working_models.json"

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

def fetch_router_models(api_token: str) -> List[str]:
    """Получает список моделей, доступных через роутер Hugging Face."""
    print("🔍 Запрашиваю список моделей из роутера Hugging Face...")
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        response = requests.get(ROUTER_MODELS_URL, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        models = [item["id"] for item in data.get("data", []) if "id" in item]
        print(f"✅ Найдено {len(models)} моделей в роутере.")
        return models
    except Exception as e:
        print(f"❌ Ошибка при получении списка моделей из роутера: {e}")
        return []

def find_and_cache_working_models(api_token: str) -> List[str]:
    """Находит рабочие модели (те, на которые роутер отвечает 200), кэширует и возвращает."""
    if os.path.exists(WORKING_MODELS_FILE):
        try:
            with open(WORKING_MODELS_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached:
                print(f"📦 Загружено {len(cached)} моделей из кэша.")
                return cached
        except Exception as e:
            print(f"⚠️ Не удалось загрузить кэш: {e}")

    print("💡 Кэш не найден или пуст. Получаю список моделей из роутера...")
    all_models = fetch_router_models(api_token)
    if not all_models:
        return []

    print(f"🔎 Проверяю доступность {len(all_models)} моделей (может занять минуту)...")
    working = []
    for model_id in all_models:
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": False
        }
        try:
            resp = requests.post(ROUTER_CHAT_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                working.append(model_id)
                print(f"  ✅ {model_id} — доступна.")
        except:
            pass
    if working:
        with open(WORKING_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(working, f, indent=2, ensure_ascii=False)
        print(f"💾 Найдено и сохранено {len(working)} рабочих моделей.")
    else:
        print("⚠️ Не найдено ни одной рабочей модели через роутер.")
    return working

def generate_content_with_fallback(api_token: str, model_list: List[str]) -> Tuple[bool, str, Optional[str]]:
    if not model_list:
        return False, "❌ Список доступных моделей пуст.", None

    for model_id in model_list:
        print(f"🔄 Пробую модель: {model_id}")
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 400,
            "temperature": 0.7,
            "stream": False
        }
        try:
            response = requests.post(ROUTER_CHAT_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    text = result["choices"][0]["message"]["content"].strip()
                    if text:
                        print(f"✅ Успех! Пост создан с помощью {model_id}.")
                        return True, text, model_id
                    else:
                        print(f"⚠️ Модель {model_id} вернула пустой ответ.")
                else:
                    print(f"⚠️ Неожиданный формат ответа от {model_id}.")
            else:
                print(f"⚠️ Модель {model_id} вернула ошибку {response.status_code}. Пробую следующую...")
        except Exception as e:
            print(f"⚠️ Ошибка с моделью {model_id}: {str(e)[:100]}. Пробую следующую...")
    return False, "❌ Все доступные модели из списка не смогли создать пост.", None

def send_to_telegram(text: str, is_error: bool = False) -> bool:
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("❌ Ошибка: Не указаны токен бота или ID чата.")
        return False

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    if is_error:
        payload = {"chat_id": TG_CHAT_ID, "text": text}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                print("✅ Сообщение об ошибке отправлено в Telegram.")
                return True
            else:
                print(f"❌ Не удалось отправить ошибку: {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Ошибка при отправке: {e}")
            return False
    else:
        safe_text = escape_markdown(text)
        payload = {"chat_id": TG_CHAT_ID, "text": safe_text, "parse_mode": "MarkdownV2"}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                print("✅ Пост успешно опубликован!")
                return True
            elif resp.status_code == 400 and "can't parse entities" in resp.text:
                print("⚠️ Ошибка Markdown, отправляем без форматирования...")
                payload2 = {"chat_id": TG_CHAT_ID, "text": text}
                resp2 = requests.post(url, json=payload2, timeout=30)
                return resp2.status_code == 200
            else:
                print(f"❌ Ошибка отправки: {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")

    if not HF_API_TOKEN:
        send_to_telegram("❌ Не указан токен Hugging Face.", is_error=True)
        exit(1)

    working_models = find_and_cache_working_models(HF_API_TOKEN)
    if not working_models:
        send_to_telegram("❌ Не найдено ни одной доступной модели через роутер.", is_error=True)
        exit(1)

    success, content, used_model = generate_content_with_fallback(HF_API_TOKEN, working_models)

    if success:
        print(f"📝 Пост (модель {used_model}):\n{content}\n")
        if not send_to_telegram(content, is_error=False):
            send_to_telegram("⚠️ Пост создан, но не отправлен в Telegram.", is_error=True)
    else:
        print(f"[PUBLISH_FAIL] {content}")
        send_to_telegram(content, is_error=True)
