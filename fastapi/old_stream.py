import redis
import json
import time
import os
import mlflow
import pandas as pd

# --------------------------------------------------------
# Conexión a Redis
# --------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
STREAM_KEY = "mlops_stream"
GROUP_NAME = "workers"
CONSUMER_NAME = "consumer-1"

# Crear conexión Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Verificar conexión
try:
    r.ping()
    print(f"✅ Conectado a Redis en {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f"❌ Error conectando a Redis: {e}")
    exit(1)

# Crear consumer group si no existe
try:
    r.xgroup_create(STREAM_KEY, GROUP_NAME, id="$", mkstream=True)
    print(f"🆕 Grupo '{GROUP_NAME}' creado en stream '{STREAM_KEY}'")
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" in str(e):
        print(f"ℹ️ Grupo '{GROUP_NAME}' ya existía, continuando...")
    else:
        raise e

print("🚀 Worker escuchando Redis Stream...")

# --------------------------------------------------------
# Bucle principal
# --------------------------------------------------------
while True:
    try:
        resp = r.xreadgroup(
            groupname=GROUP_NAME,
            consumername=CONSUMER_NAME,
            streams={STREAM_KEY: ">"},
            count=1,
            block=5000
        )

        if resp:
            for stream, messages in resp:
                for msg_id, msg_data in messages:
                    try:
                        # --- Parsear datos ---
                        data = json.loads(msg_data["data"])
                        payload = data["payload"]
                        model_name = data.get("model", "svm")

                        print(f"📥 Mensaje recibido {msg_id} | Modelo: {model_name}")

                        # --- Cargar modelo desde MLflow ---
                        model_uri = f"models:/{model_name}/Production"
                        model = mlflow.pyfunc.load_model(model_uri)

                        # --- Convertir datos a DataFrame ---
                        df = pd.DataFrame(payload["data"])
                        if "columns" in payload and payload["columns"]:
                            df = df[payload["columns"]]

                        # --- Ejecutar predicción ---
                        predictions = model.predict(df)

                        # --- Log resultado ---
                        result = {
                            "id": msg_id,
                            "model": model_name,
                            "predictions": predictions.tolist()
                        }

                        print(f"✅ Predicción completada: {result}")

                        # --- Guardar en log local ---
                        with open("results.log", "a") as f:
                            f.write(json.dumps(result) + "\n")

                        # --- Confirmar mensaje procesado ---
                        r.xack(STREAM_KEY, GROUP_NAME, msg_id)

                    except Exception as e:
                        print(f"❌ Error procesando mensaje {msg_id}: {e}")
                        # No ACK para permitir reproceso

        else:
            print("⏳ Esperando nuevos mensajes...")

    except Exception as e:
        print(f"⚠️ Error general: {e}")
        time.sleep(5)

    time.sleep(1)

