import os
import json
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import uvicorn
import threading
import time
import shutil
from dotenv import load_dotenv  # Charger les variables d'environnement locales

# Charger les variables d'environnement depuis un fichier .env s'il existe
load_dotenv()

# Configuration d'Application Insights
instrumentation_key = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

# Vérifier que la clé est bien définie
if not instrumentation_key:
    raise ValueError("Azure Application Insights connection string is missing. Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.")

# Configurer un logger pour envoyer des logs à Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    logger.addHandler(AzureLogHandler(connection_string=instrumentation_key))
except Exception as e:
    raise ValueError(f"Failed to set up AzureLogHandler: {str(e)}")

logger.info("Application Insights logging is configured successfully.")

# Définir le chemin temporaire où seront stockés les modèles
tmp_model_dir = "/tmp/models"
os.makedirs(tmp_model_dir, exist_ok=True)  # Crée le répertoire si non existant

# Définir le chemin des fichiers modèles d'origine (dans le repo)
source_model_dir = "./models"

# Copier les fichiers du dossier ./models vers /tmp/models
if os.path.exists(source_model_dir):
    for file_name in os.listdir(source_model_dir):
        src_file = os.path.join(source_model_dir, file_name)
        dst_file = os.path.join(tmp_model_dir, file_name)
        shutil.copy(src_file, dst_file)
    logger.info(f"Model files copied to {tmp_model_dir}")

# Chemins vers les fichiers du modèle et de TextVectorization
model_path = os.path.join(tmp_model_dir, "best_model_fasttext.keras")
config_path = os.path.join(tmp_model_dir, "tv_layer_config.json")
vocab_path = os.path.join(tmp_model_dir, "tv_layer_vocabulary.txt")

# Indicateur pour savoir si le modèle est bien chargé
model = None
tv_layer = None
model_loaded = False

def load_model():
    global model, tv_layer, model_loaded
    try:
        logger.info("Starting model loading in a background thread...")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")

        # Chargement de la configuration de TextVectorization
        with open(config_path, "r") as file:
            tv_layer_config = json.load(file)

        # Créer la couche TextVectorization
        tv_layer = tf.keras.layers.TextVectorization.from_config(tv_layer_config)

        # Charger le vocabulaire
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            vocabulary = [line.strip() for line in vocab_file]

        # Définir le vocabulaire dans la couche TextVectorization
        tv_layer.set_vocabulary(vocabulary)

        logger.info("TextVectorization layer configured successfully.")
        model_loaded = True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False

# Lancer le chargement du modèle dans un thread séparé
threading.Thread(target=load_model, daemon=True).start()

# Initialiser l'application FastAPI
app = FastAPI()

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

def wait_for_model():
    max_wait_time = 60
    waited = 0
    while not model_loaded and waited < max_wait_time:
        time.sleep(1)
        waited += 1
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again later.")

@app.post("/predict")
async def predict(input: TextInput):
    wait_for_model()
    try:
        sequences = tv_layer([input.text])
        prediction = model.predict(sequences)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

        logger.info(f"Prediction: {sentiment} for text: {input.text}")
        return {"prediction": sentiment}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        if not input.validation:
            logger.warning(f"Mal Predicted Tweet: {input.text}, Prediction: {input.prediction}")
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        logger.error(f"Error during feedback processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting FastAPI server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
