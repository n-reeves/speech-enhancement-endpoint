from typing import List
from fastapi import FastAPI, status
from pydantic import BaseModel
import torch
import os
from utils import batch_stft, batch_istft, file_to_batch, batch_to_file

MODEL_PATH = "ml/model"
MODEL_NAME = "model.pth"

def load_model():
    """Load the model from the model_dir.
    Args:
        model_dir (str): path to the directory containing model.pth
        
    Returns:
        torch.jit.ScriptModule
    """
    model = torch.jit.load(os.path.join(MODEL_PATH, MODEL_NAME))
    return model

def preprocess(input_audio: _type_):
    """Preprocess the input audio.

    Args:
        input_audio (_type_): Open

    Returns:
        torch.tensor: audio tensor of shape (Batch, 1, frequency bins, time frames, 2)
    """
    audio_tensor = torch.tensor(input_audio) #convert input audio to audio tensor
    audio_batch = file_to_batch(audio_tensor)
    stft_batch = batch_stft(audio_batch)
    return stft_batch

def predict(model, input: torch.tensor):
    """Predict the output using the model and input.

    Args:
        model (torch.jit.ScriptModule): traced model
        input (torch.tensor): audio tensor of shape (Batch(variable), 1, 257, 256, 2)

    Returns:
        torch.tensor: torch.tensor of shape (Batch, 1, frequency bins, time frames)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input = input.to(device)
    output = model(input)
    return output

def postprocess(output: torch.tensor):
    """Postprocess the output audio.

    Args:
        output (torch.tensor): audio tensor of shape (Batch, 1, frequency bins, time frames)

    Returns:
        _type_: Open
    """
    batch_wav = batch_istft(output)
    wav = batch_to_file(batch_wav)
    wav = wav.detach().cpu().numpy()
    return wav

def load_payload():
    pass

app = FastAPI()

class Payload(BaseModel):
    input: List[float] #what is this?

#required formatting to create endpoint healthcheck  
@app.get('/ping')
def ping():
    return {"message": "pong"}

#to invoke the model, need to send data to /invocations
@app.post('/invocations')
def invoke():
    pass