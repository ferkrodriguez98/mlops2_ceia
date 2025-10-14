# Usa una imagen base de Python ligera (Recomendado para producción)
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
COPY dockerfiles/grpc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación
COPY . .

# Documenta el puerto en el que escucha el servicio gRPC (50051 es el estándar)
EXPOSE 50051

# Define el comando para ejecutar el servidor gRPC de Python
CMD ["python", "server.py"]

# TESTING:
# docker build -t grpc:latest -f dockerfiles/grpc/Dockerfile .
# docker run -d -p 50051:50051 grpc