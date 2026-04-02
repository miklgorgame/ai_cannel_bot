import os
import requests
import json
import re
from typing import List, Dict, Tuple, Optional

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/v1"
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
    """Экранирует спецсимволы MarkdownV2 для Telegram."""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)

def fetch_available_models(api_token: str, limit: int = 40) -> List[str]:
    """
    Запрашивает список доступных моделей для текстовой генерации из Hugging Face Hub API.
    Документация: https://huggingface.co/docs/inference-providers/en/hub-api#list-models
    """
    print(f"🔍 Запрашиваю список доступных моделей с Hugging Face...")
    params = {
        "inference_provider": "all",      # Показывает модели от всех провайдеров[reference:3]
        "pipeline_tag": "text-generation", # Только модели для генерации текста
        "limit": limit
    }
    headers = {"Authorization": f"Bearer {api_token}"}
    url = "https://huggingface.co/api/models"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        models_data = response.json()
        # Извлекаем ID каждой модели
        model_ids = [model.get("id") for model in models_data if model.get("id")]
        print(f"✅ Найдено {len(model_ids)} кандидатов.")
        return model_ids
    except Exception as e:
        print(f"❌ Ошибка при получении списка моделей: {e}")
        return []

def check_model_availability(api_token: str, model_id: str) -> bool:
    """
    Проверяет, доступна ли конкретная модель для бесплатного инференса.
    Документация: https://huggingface.co/docs/inference-providers/en/hub-api#get-model-status
    """
    url = f"https://huggingface.co/api/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}
    # Параметр 'expand=inference' запрашивает статус доступности модели[reference:4]
    params = {"expand": "inference"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        model_info = response.json()
        # Если поле 'inference' имеет значение 'warm', модель готова к использованию
        return model_info.get("inference") == "warm"
    except Exception:
        return False

def find_and_cache_working_models(api_token: str) -> List[str]:
    """
    Находит рабочие модели, кэширует их в файл и возвращает список.
    """
    if os.path.exists(WORKING_MODELS_FILE):
        try:
            with open(WORKING_MODELS_FILE, "r", encoding="utf-8") as f:
                cached_models = json.load(f)
            if cached_models:
                print(f"📦 Загружено {len(cached_models)} моделей из кэша.")
                return cached_models
        except Exception as e:
            print(f"⚠️ Не удалось загрузить кэш: {e}")

    print("💡 Кэш не найден или пуст. Начинаю поиск доступных моделей...")
    candidates = fetch_available_models(api_token)
    if not candidates:
        print("❌ Не удалось получить список моделей-кандидатов.")
        return []

    working_models = []
    print(f"🔎 Проверяю доступность {len(candidates)} моделей (это может занять минуту)...")
    for model_id in candidates:
        if check_model_availability(api_token, model_id):
            working_models.append(model_id)
            print(f"  ✅ {model_id} — доступна.")
        # Можно добавить небольшую задержку, чтобы не нагружать API
        # time.sleep(0.2)

    if working_models:
        with open(WORKING_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(working_models, f, indent=2, ensure_ascii=False)
        print(f"💾 Найдено и сохранено {len(working_models)} рабочих моделей в {WORKING_MODELS_FILE}.")
    else:
        print("⚠️ Не найдено ни одной подходящей модели.")
    return working_models

def generate_content_with_fallback(api_token: str, model_list: List[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Пытается сгенерировать контент, перебирая модели из списка.
    Возвращает: (успех, контент_или_ошибка, использованная_модель_или_None)
    """
    if not model_list:
        return False, "❌ Список доступных моделей пуст. Попробуйте перезапустить бота.", None

    for model_id in model_list:
        print(f"🔄 Пробую модель: {model_id}")
        headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,  # Роутер сам выберет подходящего провайдера
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 400,
            "temperature": 0.7,
            "stream": False
        }
        try:
            response = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    generated_text = result["choices"][0]["message"]["content"].strip()
                    if generated_text:
                        print(f"✅ Успех! Пост создан с помощью {model_id}.")
                        return True, generated_text, model_id
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
    """Отправляет сообщение в Telegram. Для ошибок отключает Markdown."""
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
                print(f"❌ Не удалось отправить ошибку в Telegram: {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Ошибка при отправке ошибки: {str(e)}")
            return False
    else:
        safe_text = escape_markdown(text)
        payload = {"chat_id": TG_CHAT_ID, "text": safe_text, "parse_mode": "MarkdownV2"}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                print("✅ Пост успешно опубликован в канале!")
                return True
            elif resp.status_code == 400 and "can't parse entities" in resp.text:
                print("⚠️ Ошибка парсинга Markdown, отправляем без форматирования...")
                payload_no_md = {"chat_id": TG_CHAT_ID, "text": text}
                resp2 = requests.post(url, json=payload_no_md, timeout=30)
                return resp2.status_code == 200
            else:
                print(f"❌ Ошибка отправки в Telegram: {resp.text}")
                return False
        except Exception as e:
            print(f"❌ Ошибка при отправке: {str(e)}")
            return False

if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")

    if not HF_API_TOKEN:
        error_msg = "❌ Ошибка: Не указан токен Hugging Face."
        print(error_msg)
        send_to_telegram(error_msg, is_error=True)
        exit(1)

    # 1. Найти и сохранить рабочие модели
    working_models = find_and_cache_working_models(HF_API_TOKEN)
    if not working_models:
        error_msg = "❌ Не удалось найти ни одной доступной модели для генерации текста."
        print(error_msg)
        send_to_telegram(error_msg, is_error=True)
        exit(1)

    # 2. Попытаться сгенерировать пост, перебирая модели
    success, content, used_model = generate_content_with_fallback(HF_API_TOKEN, working_models)

    if success:
        print(f"📝 Сгенерированный текст (модель: {used_model}):\n{content}\n")
        if send_to_telegram(content, is_error=False):
            print("[PUBLISH_SUCCESS] Пост отправлен в Telegram.")
        else:
            error_msg = "⚠️ Не удалось опубликовать сгенерированный пост в Telegram (ошибка отправки)."
            print(error_msg)
            send_to_telegram(error_msg, is_error=True)
    else:
        print(f"[PUBLISH_FAIL] Генерация не удалась: {content}")
        send_to_telegram(content, is_error=True)
