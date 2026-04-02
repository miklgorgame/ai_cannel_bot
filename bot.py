import os
import requests
import json

# --- КОНФИГУРАЦИЯ ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# Новый URL для Hugging Face
HF_API_URL = "https://router.huggingface.co/hf-inference/models/IlyaGusev/saiga_llama3_8b"

PROMPT = """
Ты — опытный Python-разработчик и энтузиаст IT. 
Напиши короткий, увлекательный и полезный пост для Telegram-канала о программировании.
Тема: интересные факты о Python, советы по коду или новости технологий.
Стиль: дружелюбный, профессиональный, с эмодзи.
Язык: Русский.
Объем: до 1000 символов.
"""

def generate_content():
    if not HF_API_TOKEN:
        return "❌ Ошибка: Не указан токен Hugging Face."
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": PROMPT,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            else:
                return f"⚠️ Странный ответ от AI: {result}"
        else:
            return f"Ошибка API HF ({response.status_code}): {response.text[:100]}"
            
    except Exception as e:
        return f"❌ Ошибка соединения с HF: {str(e)}"

def send_to_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("❌ Ошибка: Не указаны токен бота или ID чата.")
        return False
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            print("✅ Пост успешно опубликован в канале!")
            return True
        else:
            print(f"❌ Ошибка отправки в Telegram: {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка при отправке: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск генерации поста...")
    
    content = generate_content()
    print(f"📝 Сгенерированный текст:\n{content}\n")
    
    if content and not content.startswith("❌") and not content.startswith("⚠️"):
        success = send_to_telegram(content)
        if success:
            print("[PUBLISH_SUCCESS] Post sent to Telegram.")
        else:
            print("[PUBLISH_FAIL] Failed to send post.")
    else:
        print("[PUBLISH_FAIL] Content generation failed.")
