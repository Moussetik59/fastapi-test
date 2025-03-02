import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from opencensus.ext.azure.log_exporter import AzureLogHandler

# Charger les variables d'environnement depuis .env (si disponible)
load_dotenv()

# === Configuration des logs Azure Application Insights ===
connection_string = os.getenv("APPINSIGHTS_CONNECTION_STRING")

if not connection_string:
    raise ValueError("APPINSIGHTS_CONNECTION_STRING n'est pas défini dans les variables d'environnement.")

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
azure_handler = AzureLogHandler(connection_string=connection_string)
logger.addHandler(azure_handler)

logger.info("API démarrée - Logs connectés à Azure Application Insights")

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Endpoint '/' appelé")
    return {"message": "Hello World avec logs !"}
