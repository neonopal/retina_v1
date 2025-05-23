import deepl
from .config import DEEPL_KEY

def translate_text(text, target_lang="ID"):
    
    """
    parameters:
        - text (str) = text to translate
        - target_lang (str) = target language
    """
    
    auth_key = DEEPL_KEY  
    translator = deepl.Translator(auth_key)
    result = translator.translate_text(text, target_lang=target_lang)
    return result.text