# from langchain_community.chat_models.ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
from .image_preprocessing_services import convert_to_base64
# from langchain_core.messages import HumanMessage
from .translate_services import translate_text
from .config import URL_LLM
import logging
# from .google_serper import google_serper_agent
# from langchain.agents import initialize_agent
import requests

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def invoke_vlm(prompt:str, img:str) -> str:
    """
    post prompt dan data ke server vlm.
    params:
        - prompt (str) : prompt hasil trascribe STT.
        - image (str) : image hasil encode base64 UTF-8.
    """
    
    payload = {
        "prompt" : prompt, 
        "image" : img
    }
    response = requests.post(
        URL_LLM,
        json=payload  
    )
    print(f"Send to VLM ==> {response.status_code}")
    data = response.json()['response']
    return data



