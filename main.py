from typing import List
from fastapi import FastAPI, status
from pydantic import BaseModel
import torch
import numpy as np
import os
from utils import batch_stft, batch_istft, file_to_batch, batch_to_file, normalize_loudness

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
        torch.Tensor: audio tensor of shape (Batch, 1, frequency bins, time frames)
    """
    audio_batch = file_to_batch(input_audio)
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
    input = torch.view_as_real(input).to(device) #(Batch, 1, 257, 256) -> (Batch, 1, 257, 256, 2)
    output = model(input)
    return output

def postprocess(output: torch.Tensor, wav_input: torch.Tensor) -> dict:
    """Postprocess the output audio.

    Args:
        output (torch.Tensor): batched STFT coefs masked by model. shape (Batch, 1, frequency bins, time frames)
        input (torch.Tensor): audio segment that has been processed (1, clip_length)

    Returns:
        dict: contains the output audio and input audio as numpy arrays
    """
    #output of model -> (1, B*segment_length)
    batch_output = batch_istft(output)
    wav_output = batch_to_file(batch_output)
    
    #put both input and output audio files on cpu if on gpu
    wav_output = wav_output.detach()
    if wav_output.device != torch.device('cpu'):
        wav_output = wav_output.cpu()
    
    if wav_input.device != torch.device('cpu'):
        wav_input = wav_input.cpu()
    
    #crop model out to match input
    input_length = wav_input.shape[-1]
    wav_output = wav_output[:, :input_length]
    
    #normalize output loudness
    wav_output = normalize_loudness(wav_output, wav_input)
    
    #input and output audio files to numpy
    wav_output = wav_output.numpy()
    wav_input = wav_input.numpy()
    return {'output_audio': wav_output, 'input_audio': wav_input}

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


### Test decorators

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


##test preprocess function
@app.post('/test/preprocess')
def test_preprocess(input_audio: Payload):
    input_tensor = torch.tensor(input_audio.audio)
    model_in = preprocess(input_tensor)
    return {"output": list(model_in.shape)}


##test predict function
@app.post('/test/predict')
def test_predict(input_audio: Payload):
    model = load_model()
    input_tensor = torch.tensor(input_audio.audio)
    model_in = preprocess(input_tensor)
    output_tensor = predict(model, model_in)
    return {"output":list(output_tensor.shape)}


##test postprocess function
@app.post('/test/postprocess')
def test_postprocess(input_audio: Payload):
    model = load_model()
    input_tensor = torch.tensor(input_audio.audio)
    model_in = preprocess(input_tensor)
    output_tensor = predict(model, model_in)
    wav_dict = postprocess(output_tensor, input_tensor)
    return {"wav": wav_dict['output_audio'].tolist()}