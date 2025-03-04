import os
import json
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
import uvicorn
from dotenv import load_dotenv  # Ajout de dotenv pour charger les variables d'environnement locales

# Charger les variables d'environnement depuis un fichier .env s'il existe (utile en local)
load_dotenv()

# Configuration d'Application Insights
instrumentation_key = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

# Vérifier que la clé est bien définie
if not instrumentation_key:
    raise ValueError("Azure Application Insights connection string is missing. Set APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.")

# Configurer un logger pour envoyer des logs à Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Définir le niveau de log

try:
    logger.addHandler(AzureLogHandler(connection_string=instrumentation_key))
except Exception as e:
    raise ValueError(f"Failed to set up AzureLogHandler: {str(e)}")

logger.info("Application Insights logging is configured successfully.")

# Définir le chemin de base (racine du script)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Chemins vers les fichiers du modèle et de la TextVectorization
model_path = os.path.join(base_dir, "models", "best_model_fasttext.keras")
config_path = os.path.join(base_dir, "models", "tv_layer_config.json")
vocab_path = os.path.join(base_dir, "models", "tv_layer_vocabulary.txt")

# Vérification de l'existence du modèle
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Chargement du modèle
try:
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully.")
except ValueError as e:
    logger.error("Error loading model: ensure it is in a compatible format (.keras or .h5).")
    raise ValueError("Error loading model: ensure it is in a compatible format (.keras or .h5).") from e

# Chargement de la configuration de TextVectorization
try:
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
except Exception as e:
    logger.error(f"Error setting up TextVectorization layer: {str(e)}")
    raise RuntimeError(f"Error setting up TextVectorization layer: {str(e)}")

# Initialiser l'application FastAPI
app = FastAPI()

# Rediriger la racine vers /docs
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url='/docs')

# Définir les classes d'entrée pour les prédictions et les retours
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: str
    validation: bool

# Point de terminaison pour les prédictions
@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Vectoriser le texte d'entrée
        sequences = tv_layer([input.text])
        # Prédire le sentiment
        prediction = model.predict(sequences)
        sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

        logger.info(f"Prediction: {sentiment} for text: {input.text}")  # Log de la prédiction

        return {"prediction": sentiment}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")  # Log de l'erreur
        raise HTTPException(status_code=500, detail=str(e))

# Point de terminaison pour le feedback utilisateur
@app.post("/feedback")
async def feedback(input: FeedbackInput):
    try:
        # Traiter le retour utilisateur
        if not input.validation:
            # Enregistrer les tweets mal prédits
            logger.warning(f"Mal Predicted Tweet: {input.text}, Prediction: {input.prediction}")
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        logger.error(f"Error during feedback processing: {str(e)}")  # Log de l'erreur
        raise HTTPException(status_code=500, detail=str(e))

# Exécuter l'API
if __name__ == "__main__":
    # Déterminer le port à utiliser
    port = int(os.getenv("PORT", 8000))  # Azure fournit le port via la variable d'environnement PORT

    logger.info(f"Starting FastAPI server on port {port}...")

    # Lancer l'application
    uvicorn.run("main:app", host="0.0.0.0", port=port)
