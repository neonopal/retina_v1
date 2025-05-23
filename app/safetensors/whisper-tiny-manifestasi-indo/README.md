---
library_name: transformers
language:
- id
license: apache-2.0
base_model: manifestasi/whisper-tiny-manifestasi-indo
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: Whisper Tiny Id v2 - Manifestasi
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Example

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66da99283bc7bd82e34fb066/A1iyahJY3aWk40G0fno00.png)

# Example Code

```
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import librosa
import time

processor = WhisperProcessor.from_pretrained("manifestasi/whisper-tiny-manifestasi-indo-v2")
model = WhisperForConditionalGeneration.from_pretrained("manifestasi/whisper-tiny-manifestasi-indo-v2")
model.config.forced_decoder_ids = None

a = time.time()
audio, rate = librosa.load("/kaggle/input/berhasil-konek/berhasil_retina.mp3", sr = 16000)
input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features 

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print(transcription)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
b = time.time() - a
print(f"{b} detik")

```

# Whisper Tiny Id v2 - Manifestasi

This model is a fine-tuned version of [manifestasi/whisper-tiny-manifestasi-indo](https://huggingface.co/manifestasi/whisper-tiny-manifestasi-indo) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3239
- Wer: 15.4907

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- training_steps: 6000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer     |
|:-------------:|:-----:|:----:|:---------------:|:-------:|
| 0.4061        | 0.48  | 300  | 0.5425          | 29.6021 |
| 0.4917        | 0.96  | 600  | 0.5056          | 28.6207 |
| 0.3126        | 1.44  | 900  | 0.4572          | 27.1618 |
| 0.2996        | 1.92  | 1200 | 0.4088          | 23.2626 |
| 0.1762        | 2.4   | 1500 | 0.3780          | 20.7427 |
| 0.1702        | 2.88  | 1800 | 0.3536          | 20.1857 |
| 0.0976        | 3.36  | 2100 | 0.3423          | 18.3289 |
| 0.1074        | 3.84  | 2400 | 0.3294          | 17.9576 |
| 0.0549        | 4.32  | 2700 | 0.3214          | 17.2414 |
| 0.0584        | 4.8   | 3000 | 0.3213          | 16.7905 |
| 0.0334        | 5.28  | 3300 | 0.3158          | 18.3820 |
| 0.0335        | 5.76  | 3600 | 0.3165          | 16.4456 |
| 0.0183        | 6.24  | 3900 | 0.3175          | 16.0743 |
| 0.0175        | 6.72  | 4200 | 0.3200          | 16.1008 |
| 0.01          | 7.2   | 4500 | 0.3197          | 15.6499 |
| 0.0098        | 7.68  | 4800 | 0.3212          | 15.7825 |
| 0.0085        | 8.16  | 5100 | 0.3229          | 17.7719 |
| 0.0076        | 8.64  | 5400 | 0.3237          | 16.0743 |
| 0.0082        | 9.12  | 5700 | 0.3237          | 15.7825 |
| 0.0068        | 9.6   | 6000 | 0.3239          | 15.4907 |


### Framework versions

- Transformers 4.51.1
- Pytorch 2.5.1+cu124
- Datasets 3.5.0
- Tokenizers 0.21.0
