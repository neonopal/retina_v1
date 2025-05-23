from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from .image_preprocessing_services import convert_to_base64
from langchain_core.messages import HumanMessage
from .translate_services import translate_text
from .config import URL_LLM
import logging
from .google_serper import google_serper_agent
from langchain.agents import initialize_agent
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


"""
load model.
"""

llm = ChatOllama(model="gemma3:4b-it-q4_K_M",
                 temperature=0,
                 base_url=URL_LLM)


def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def invoke(txt:str, image_b64):
    """
    Function to invoke llm to response the user's prompt.
    
    parameters:
        - txt (str) = prompt user, dari hasil Speech to text.
        - image_b64 (any) = gambar dari ESP32.
    returns:
        - response (str) = hasil dari model.    
    """
    
    prompt = translate_text(txt, target_lang="EN-US")
    print(prompt)
    logging.debug(prompt)
    
    # tools = google_serper_agent()
     
    # agent = initialize_agent(
    #     tools = tools, 
    #     llm = llm, 
    #     agent = "zero-shot-react-description",
    #     handle_parsing_errors=True
    # )
    
    # input_prompt = {"text": prompt, "image": image_b64}
    # message = prompt_func(input_prompt)
    # logging.debug("prompt_func : ", message)
    # response = agent.run(prompt)
    # chain = prompt_func | agent
    # chain = prompt_func | agent.run | StrOutputParser()
    chain = prompt_func | llm | StrOutputParser()
    
    response = chain.invoke(
        {"text": prompt, "image": image_b64}
    )
    
    logging.debug("log invoke: ", response)
    
    response_trans = translate_text(response, target_lang="ID") 
    # logging.debug(response_trans)
    # return str(response_trans)
    return response_trans





