import os
import json
import time
import redis
import mlflow
import pandas as pd
import csv
from datetime import datetime
import warnings
import logging

# ========================
# 🔇 Limpieza de logs
# ========================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("pandas").setLevel(logging.ERROR)

# ========================
# ⚙️ Configuración de Redis y MLflow
# ========================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
STREAM_KEY = "mlops_stream"
GROUP_NAME = "workers"
CONSUMER_NAME = "consumer-1"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ========================
# 🔁 Función para cargar el modelo
# ========================
def _load_model_from_registry(model_name: str):
    client = mlflow.tracking.MlflowClient()
    reg_name = model_name_map(model_name)
    versions = client.get_latest_versions(reg_name)
    if not versions:
        raise Exception(f"No se encontró ninguna versión registrada para el modelo {model_name}")
    model_uri = f"models:/{reg_name}/{versions[0].version}"
    print(f"[INFO] Loading model from URI: {model_uri}", flush=True)
    return mlflow.pyfunc.load_model(model_uri)


# ========================
# 🔁 Mapeo entre nombres de API y registro MLflow
# ========================
def model_name_map(model_name):
    mapping = {
        "svm": "SVC",
        "knn": "Knn_Classifier",
        "lightgbm": "LightGBM_Classifier"
    }
    return mapping.get(model_name.lower(), model_name)


# ========================
# 🚀 Main worker loop
# ========================
def main():
    print("🔄 Iniciando worker de streaming...", flush=True)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    print(f"✅ Conectado a Redis en {REDIS_HOST}:{REDIS_PORT}", flush=True)

    try:
        r.xgroup_create(name=STREAM_KEY, groupname=GROUP_NAME, id="$", mkstream=True)
        print(f"🆕 Grupo {GROUP_NAME} creado en {STREAM_KEY}", flush=True)
    except redis.exceptions.ResponseError:
        print(f"ℹ️ Grupo {GROUP_NAME} ya existía, continuando...", flush=True)

    print("🚀 Worker escuchando Redis Stream...", flush=True)

    while True:
        try:
            messages = r.xreadgroup(groupname=GROUP_NAME, consumername=CONSUMER_NAME,
                                    streams={STREAM_KEY: ">"}, count=1, block=5000)

            if not messages:
                print("⏳ Esperando nuevos mensajes...", flush=True)
                continue

            for stream_name, entries in messages:
                for msg_id, data in entries:
                    payload = json.loads(data["data"])
                    model_name = payload.get("model", "unknown")
                    print(f"📥 Mensaje recibido {msg_id} | Modelo: {model_name}", flush=True)

                    try:
                        model = _load_model_from_registry(model_name)
                        df = pd.DataFrame(payload["payload"]["data"])
                        preds = model.predict(df)
                        print(f"✅ Predicción completada ({len(preds)} muestras): {preds.tolist()}", flush=True)

                        # === Guardar resultado en CSV ===
                        output_file = "/app/predicciones.csv"
                        file_exists = os.path.isfile(output_file)

                        with open(output_file, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["timestamp", "modelo", "prediccion", "data"])

                            writer.writerow([
                                datetime.now().isoformat(timespec="seconds"),
                                model_name,
                                preds.tolist(),
                                df.to_json(orient="records")
                            ])

                        print(f"📝 Registro guardado en predicciones.csv para modelo {model_name}", flush=True)

                    except Exception as e:
                        print(f"❌ Error procesando mensaje {msg_id}: {e}", flush=True)

                    # Confirmar procesamiento del mensaje
                    r.xack(STREAM_KEY, GROUP_NAME, msg_id)

        except Exception as e:
            print(f"⚠️ Error general en loop principal: {e}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
