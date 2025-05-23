from pydantic import BaseModel
from fastapi import Form, UploadFile, File

class ChatModel:
    def __init__(self,
                 prompt:str = Form(...), 
                 image:UploadFile = File(...)):
        self.prompt = prompt
        self.image = image
