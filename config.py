import os
from dotenv import load_dotenv

# โหลด .env อัตโนมัติ
load_dotenv()

class Config:
    # Groq API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # ตัวอย่างอื่น ๆ
    APP_PORT = int(os.getenv("APP_PORT", 8000))
    PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
