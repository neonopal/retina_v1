from fastapi import APIRouter
from app.services.llm_services import invoke_vlm
from app.schemas.chat_model import ChatModel
from app.services.image_preprocessing_services import convert_to_base64
# from app.services.image_preprocessing_services import processor_image
from app.services.stt_services import Whisper
# from app.services.stt_services import Wav2Vec2, Whisper
from fastapi import UploadFile, File, HTTPException, Form
import logging
import io
import time
import shutil
import os
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


stt = Whisper()
# stt = Wav2Vec2()
router = APIRouter()

@router.post("/chat")
async def chat_to_llm(
                # prompt:str = Form(...),
                file_audio:UploadFile = File(...),
                file:UploadFile = File(...)
                ):
    try:
        print("===MULAI===")
        time1 = time.time()
        response = ""
        contents = file.file.read()
        # img_to_feed = processor_image(contents)
        img_base = convert_to_base64(contents)
        print("===GAMBAR SELESAI===")
        
        
        audio_contents = await file_audio.read()
        audio_buffer = io.BytesIO(audio_contents)
        prompt = stt.transcribe(audio_buffer)
        print("===AUDIO SELESAI===")
        # prompt = "jelaskan apa yang kamu lihat"
        
        if prompt != None or prompt != "":
            response = invoke_vlm(prompt, img_base)
        time2 = time.time()
            
    except HTTPException:
        raise HTTPException(status_code=500, detail="Failed to process!")
    finally:
        await file.close()
        await file_audio.close()
    return {
        "prompt": prompt,
        "response": response,
        "time" : f"{time2 - time1} detik"
    }        

    
    
        # filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        # save_path = os.path.join("saved_audio", filename)

        # # # Pastikan folder ada
        # os.makedirs("saved_audio", exist_ok=True)

        # # # Simpan file audio ke disk
        # with open(save_path, "wb") as buffer:
        #     shutil.copyfileobj(file_audio.file, buffer)
        
        # prompt_from_audio = await speech_to_text(audio_contents, file_audio.content_type)
        # logging.debug("prompt_from_audio ", prompt_from_audio)
        # response = invoke("jelaskan apa ini", img_base)
