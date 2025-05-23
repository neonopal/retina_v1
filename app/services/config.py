import os
from dotenv import load_dotenv

load_dotenv()


DEEPL_KEY = os.getenv('DEEPL_KEY')
DEEPGRAM_KEY = os.getenv('DEEPGRAM_KEY')
URL_LLM = os.getenv('URL_LLM')