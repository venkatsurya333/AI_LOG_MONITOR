import google.generativeai as genai

# Configure Gemini API key from environment variable
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("GEMINI_API_KEY not set in .env")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
