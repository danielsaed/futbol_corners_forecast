FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Exponer el puerto que usa Hugging Face Spaces
EXPOSE 7860

# ✅ COMANDO PARA FASTAPI EN HUGGING FACE SPACES
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "7860"]