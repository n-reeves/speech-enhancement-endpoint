from typing import List
from fastapi import FastAPI, status
from pydantic import BaseModel
import torch
import numpy as np
import os
from utils import batch_stft, batch_istft, file_to_batch, batch_to_file

MODEL_PATH = "ml/model"
MODEL_NAME = "model.pt"

class Payload(BaseModel):
    audio: List[List[float]]

def load_model() -> torch.jit._script.RecursiveScriptModule:
    """Load the model from the model_dir.
    Args:
        model_dir (str): path to the directory containing model.pth
        
    Returns:
        torch.jit.ScriptModule
    """
    model = torch.jit.load(os.path.join(MODEL_PATH, MODEL_NAME))
    return model

def load_payload():
    pass

def preprocess(input_audio: List[List[float]]) -> torch.Tensor:
    """Preprocess the input audio.

    Args:
        input_audio (_type_): Open

    Returns:
        torch.Tensor: audio tensor of shape (Batch, 1, frequency bins, time frames, 2)
    """
    audio_tensor = torch.tensor(input_audio) #convert input audio to audio tensor
    audio_batch = file_to_batch(audio_tensor)
    stft_batch = batch_stft(audio_batch)
    return stft_batch

def predict(model: torch.jit._script.RecursiveScriptModule, input: torch.Tensor) -> torch.Tensor:
    """Predict the output using the model and input.

    Args:
        model (torch.jit.ScriptModule): traced model
        input (torch.Tensor): audio tensor of shape (Batch(variable), 1, 257, 256, 2)

    Returns:
        torch.Tensor: tensor of shape (Batch, 1, frequency bins, time frames)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input = input.to(device)
    output = model(input)
    return output

def postprocess(output: torch.Tensor, input_length: int) -> np.ndarray:
    """Postprocess the output audio.

    Args:
        output (torch.Tensor): audio tensor of shape (Batch, 1, frequency bins, time frames)

    Returns:
        _type_: List[List[floats]]
    """
    batch_wav = batch_istft(output)
    wav = batch_to_file(batch_wav)
    wav = wav.detach()
    if wav.device != torch.device('cpu'):
        wav = wav.cpu()
    wav = wav[:, :input_length]
    return wav.tolist()

### Decorators

#Reference note: to run server with uvicorn, type in terminal: uvicorn main:app --reload
app = FastAPI()

#required formatting to create endpoint healthcheck  
@app.get('/ping')
def ping():
    return {"message": "pong"}

#to invoke the model, need to send data to /invocations
@app.post('/invocations')
def invoke():
    pass


### Test endpoint routes

##test load model function
@app.get('/test/load_model')
def test_load_model():
    model = load_model()
    return {"model": str(model)}


##test recieve payload
#from python
@app.post('/test/recieve_payload')
def test_recieve_payload(input_audio: Payload):
    return {'input': input_audio.audio}

#from javascript (load payload)
#pending

##preprocess debug
@app.post('/test/preprocessdebug')
def test_preprocess(input_audio: Payload):
    out = input_audio.audio
    return {"output": out, "output_tensor": torch.tensor(out).tolist()}


##test preprocess function
@app.post('/test/preprocess')
def test_preprocess(input_audio: Payload):
    stft = preprocess(input_audio.audio)
    return {"stft": stft.tolist()}


##test predict function
@app.post('/test/predict')
def test_predict(input_audio: Payload):
    model = load_model()
    model_in = preprocess(input_audio.audio)
    output = predict(model, model_in) #complex numbers not json serializable
    return {"output":list(output.shape)}


##test postprocess function
@app.post('/test/postprocess')
def test_postprocess(input_audio: Payload):
    model = load_model()
    model_in = preprocess(input_audio.audio)
    output = predict(model, model_in)
    wav = postprocess(output, len(input_audio.audio[0]))
    return {"wav": wav}
    

