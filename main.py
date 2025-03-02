import os
import logging
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from opencensus.ext.azure.log_exporter import AzureLogHandler
from azure.storage.blob import BlobServiceClient

# Charger les variables d'environnement depuis .env (si disponible)
load_dotenv()

# === Configuration Azure Blob Storage ===
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = "models"

if not AZURE_STORAGE_ACCOUNT_NAME or not AZURE_STORAGE_ACCOUNT_KEY:
    raise ValueError("Les variables d'environnement AZURE_STORAGE_ACCOUNT_NAME et AZURE_STORAGE_ACCOUNT_KEY doivent être définies.")

# === Configuration des logs Azure Application Insights ===
connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

if not connection_string:
    raise ValueError("APPLICATIONINSIGHTS_CONNECTION_STRING n'est pas défini dans les variables d'environnement.")

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
azure_handler = AzureLogHandler(connection_string=connection_string)
logger.addHandler(azure_handler)

logger.info("API démarrée - Logs connectés à Azure Application Insights")

app = FastAPI()

# === Fonction pour télécharger un fichier depuis Azure Blob Storage ===
def download_model_from_azure(blob_name):
    """Télécharge un fichier depuis Azure Blob Storage si non présent en local."""
    local_file_path = os.path.join("models", blob_name)

    if not os.path.exists(local_file_path):
        logger.info(f"Téléchargement de {blob_name} depuis Azure...")
        try:
            blob_service_client = BlobServiceClient(
                account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=AZURE_STORAGE_ACCOUNT_KEY
            )
            blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

            with open(local_file_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

            logger.info(f"{blob_name} téléchargé avec succès !")
            time.sleep(3)

        except Exception as e:
            logger.exception(f"Erreur lors du téléchargement de {blob_name} : {e}")
            return None

    return local_file_path

@app.get("/")
async def root():
    logger.info("Endpoint '/' appelé - Téléchargement du modèle")
    return {"message": "Téléchargement du modèle en cours !"}
