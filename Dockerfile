# Usa una versión ligera de Python
FROM python:3.10-slim

# Establece el directorio de trabajo en la nube
WORKDIR /app

# Instala dependencias del sistema necesarias para procesar audio (vital para librosa y soundfile)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copia primero el requirements para aprovechar el caché de Docker
COPY requirements.txt .

# Instala las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tus archivos (app.py, model.pt, .json)
COPY . .

# Expone el puerto que usa Gradio
EXPOSE 7860

# Comando para arrancar la aplicación
CMD ["python", "app.py"]
