import google.generativeai as genai

API_KEY = "AIzaSyAzJUib7_bavnMNqeWyLBJnCwTxcCxMusE"
genai.configure(api_key=API_KEY)

# Попробуем получить список доступных моделей
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"✅ {m.name}")