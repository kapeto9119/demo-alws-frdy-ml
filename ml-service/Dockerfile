# Usa una imagen base de Python
FROM python:3.9-slim

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto en el que FastAPI se ejecutará
EXPOSE 8000

# Comando para iniciar FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]