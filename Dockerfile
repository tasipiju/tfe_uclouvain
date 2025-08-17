# Image Python légère compatible ML
FROM python:3.11-slim

# Installer les dépendances système nécessaires à certaines libs ML et images
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements.txt et installer les paquets Python
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code (backend, frontent, models, etc.) dans le conteneur
COPY . .

# Exposer le port que Streamlit utilisera (8080 requis par Cloud Run)
EXPOSE 8080

# Commande de démarrage Streamlit avec le chemin vers ton point d'entrée
CMD ["streamlit", "run", "--server.port=8080", "--server.address=0.0.0.0", "frontent/app.py"]
