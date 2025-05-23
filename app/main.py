from fastapi import FastAPI
from app.api.endpoints.v1 import chat
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


app = FastAPI(title="RETINA_UJI")

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])