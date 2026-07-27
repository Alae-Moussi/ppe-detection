FROM python:3.10-slim

# Dépendances système nécessaires pour OpenCV (utilisé par Ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python (mise en cache Docker séparée du code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle
COPY . .

# Render fournit le port via la variable d'environnement PORT
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
