# ใช้ Python slim image เพื่อขนาดเล็ก
FROM python:3.10-slim

# กำหนด working directory
WORKDIR /app

# ติดตั้ง dependencies ของระบบที่จำเป็น
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements ก่อนเพื่อลดเวลา rebuild cache
COPY requirements.txt .

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โค้ดโปรเจกต์ทั้งหมด
COPY main.py graph.py ingest_pdfs.py config.py ./


# สร้างโฟลเดอร์สำหรับ PDF และ Chroma DB
RUN mkdir -p /app/pdfs /app/chroma_db

# เปิดพอร์ตสำหรับแอป
EXPOSE 8000

# Health check เพื่อ Docker สามารถตรวจสอบได้
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# รันแอปพลิเคชัน
CMD ["python", "main.py"]
