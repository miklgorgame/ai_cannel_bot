import os
import requests
import json
import re
from typing import List, Tuple, Optional

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

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

def fetch_router_models_with_providers(api_token: str) -> List[Tuple[str, str]]:
    """
    Возвращает список кортежей (model_id, provider) для всех моделей,
    у которых есть хотя бы один live-провайдер.
    """
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        resp = requests.get(ROUTER_MODELS_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = []
        for model in data.get("data", []):
            model_id = model.get("id")
            providers = model.get("providers", [])
            for p in providers:
                if p.get("status") == "live":
                    result.append((model_id, p.get("provider")))
                    break  # берём первого живого провайдера
        print(f"✅ Найдено {len(result)} связок модель+провайдер.")
        return result
    except Exception as e:
        print(f"❌ Ошибка получения списка: {e}")
        return []

def find_and_cache_working_models(api_token: str) -> List[str]:
    """Проверяет каждую связку модель:провайдер коротким запросом."""
    if os.path.exists(WORKING_MODELS_FILE):
        try:
            with open(WORKING_MODELS_FILE, "r") as f:
                cached = json.load(f)
            if cached:
                print(f"📦 Загружено {len(cached)} рабочих моделей из кэша.")
                return cached
        except Exception as e:
            print(f"⚠️ Ошибка чтения кэша: {e}")

    print("💡 Получаю список моделей с провайдерами...")
    model_provider_pairs = fetch_router_models_with_providers(api_token)
    if not model_provider_pairs:
        return []

    print(f"🔎 Проверяю доступность {len(model_provider_pairs)} связок (первые 30)...")
    working = []
    for model_id, provider in model_provider_pairs[:30]:
        full_model = f"{model_id}:{provider}"
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        payload = {
            "model": full_model,
            "messages": [{"role": "user", "content": "Привет"}],
            "max_tokens": 5,
            "stream": False
        }
        try:
            resp = requests.post(ROUTER_CHAT_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                working.append(full_model)
                print(f"  ✅ {full_model} — работает.")
            else:
                print(f"  ❌ {full_model} — ошибка {resp.status_code}: {resp.text[:60]}")
        except Exception as e:
            print(f"  ❌ {full_model} — исключение: {str(e)[:60]}")

    if working:
        with open(WORKING_MODELS_FILE, "w") as f:
            json.dump(working, f, indent=2)
        print(f"💾 Сохранено {len(working)} рабочих моделей.")
    else:
        print("⚠️ Не найдено ни одной рабочей модели.")
    return working

def generate_content_with_fallback(api_token: str, model_list: List[str]) -> Tuple[bool, str, Optional[str]]:
    if not model_list:
        return False, "❌ Нет доступных моделей.", None

    for full_model in model_list:
        print(f"🔄 Пробую {full_model}")
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        payload = {
            "model": full_model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 400,
            "temperature": 0.7,
            "stream": False
        }
        try:
            resp = requests.post(ROUTER_CHAT_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                if "choices" in result and result["choices"]:
                    text = result["choices"][0]["message"]["content"].strip()
                    if text:
                        print(f"✅ Пост создан через {full_model}")
                        return True, text, full_model
            print(f"⚠️ {full_model} ответил {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"⚠️ Ошибка с {full_model}: {str(e)[:100]}")
    return False, "❌ Все модели не смогли создать пост.", None

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
        print(f"❌ Исключение: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")
    if not HF_API_TOKEN:
        send_to_telegram("❌ Отсутствует HF_API_TOKEN", is_error=True)
        exit(1)
    working = find_and_cache_working_models(HF_API_TOKEN)
    if not working:
        send_to_telegram("❌ Не найдено рабочих моделей. Проверьте токен или подключение.", is_error=True)
        exit(1)
    success, content, used = generate_content_with_fallback(HF_API_TOKEN, working)
    if success:
        send_to_telegram(content, is_error=False)
    else:
        send_to_telegram(content, is_error=True)
