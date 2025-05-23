import librosa
import torch
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import time
import logging

class Whisper:
    processor = WhisperProcessor.from_pretrained("app\safetensors\whisper-tiny-manifestasi-indo-v2")
    model = WhisperForConditionalGeneration.from_pretrained("app\safetensors\whisper-tiny-manifestasi-indo-v2")
    model.config.forced_decoder_ids = None    
# class Wav2Vec2:
#     processor  = Wav2Vec2Processor.from_pretrained("app\safetensors\wav2vec2-large-xlsr-bahasa-indonesia-manifestasi-indo-v3")
#     model = Wav2Vec2ForCTC.from_pretrained("app\safetensors\wav2vec2-large-xlsr-bahasa-indonesia-manifestasi-indo-v3")
        
    def __init__(self):
        print("===Whisper has init.===")
        # logging.debug("===WAV2VEC2 has init.===")
    
    def transcribe(self, audio_file, sampling_rate = 16_000)->str:
        transcription:str
        
        try:
            logging.debug("===AUDIO LOADING===")
            audio, rate = librosa.load(audio_file, sr = sampling_rate)
            input_features = self.processor(audio, sampling_rate = sampling_rate, return_tensors="pt").input_features
            logging.debug("===AUDIO DIPROSES===")
            # logits = self.model(input_values).logits
            logits = self.model.generate(input_features)
            transcription = self.processor.batch_decode(logits, skip_special_tokens=True)
            transcription = str(transcription[0])                 
            # prediction = torch.argmax(logits, dim = -1)
            # transcription = self.processor.batch_decode(prediction)[0]
            print(f"transcription : {transcription}")
            logging.debug("===AUDIO SELESAI DIPROSES===")
        except Exception as e:
            transcription = None
            logging.exception("Error : ", e)
        return transcription