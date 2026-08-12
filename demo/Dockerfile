# HF Spaces Docker SDK build for the Fasal demo.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
