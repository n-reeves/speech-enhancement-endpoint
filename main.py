from typing import List
from fastapi import FastAPI, status, Request
from pydantic import BaseModel
import torch
import numpy as np
import os
from utils import batch_stft, batch_istft, file_to_batch, batch_to_file, normalize_loudness, stereo_to_mono

MODEL_PATH = "ml/model"
MODEL_NAME = "model.pt"

FILE_SAMPLE_RATE = 8000
FILE_LENGTH_CAP_SECONDS = 31
FILE_LENGTH_CAP_SAMPLES = int(FILE_SAMPLE_RATE * FILE_LENGTH_CAP_SECONDS + FILE_SAMPLE_RATE//2)

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


def check_payload(input_audio: List[List[float]]) -> str:
    """Check basic properties of the input audio.

    Args:
        input_audio (List[List[float]]): input audio
    """
    #check number of channels
    channels = len(input_audio)
    if channels not in [1,2]:
        return "Error: Audio file must have 1 or 2 channels. Input has {}".format(channels)
    
    #check length of first channel
    c1_len = len(input_audio[0])
    if c1_len > FILE_LENGTH_CAP_SAMPLES:
        return "Error: Audio file is too long. Input samples are {0} but capped at {1}.".format(c1_len,FILE_LENGTH_CAP_SAMPLES)
    
    #check if channel lengths are equal
    if channels == 2:
        c2_len = len(input_audio[1])
        if c1_len != c2_len:
            return "Error: Channels must have same length. Channel 1 has length {0} but channel 2 has length {1}".format(c1_len, c2_len)
    return ""


def preprocess(input_audio: torch.tensor) -> torch.Tensor:
    """Preprocess the input audio.

    Args:
        input_audio (_type_): Open

    Returns:
        torch.Tensor: audio tensor of shape (Batch, 1, frequency bins, time frames)
    """
    #convert to mono by taking the average of the two channels
    if input_audio.shape[0] == 2:
        input_audio = stereo_to_mono(input_audio)
        
    #add padding 
    if input_audio.shape[1] < 512:
        padding = torch.zeros(1, 512 - input_audio.shape[1])
        input_audio = torch.cat([input_audio, padding], dim=1)
    
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
    input = torch.view_as_real(input) #(Batch, 1, 257, 256) -> (Batch, 1, 257, 256, 2)
    output = model(input)
    return output


def postprocess(output: torch.Tensor, wav_input: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Postprocess the output audio.

    Args:
        output (torch.Tensor): batched STFT coefs masked by model. shape (Batch, 1, frequency bins, time frames)
        input (torch.Tensor): audio segment that has been processed (1, clip_length)
        normalize_audio (bool): whether to normalize the output audio to match the input audio loudness

    Returns:
        (torch.Tensor): audio tensor of shape (1, clip_length)
    """
    #output of model -> (1, B*segment_length)
    batch_output = batch_istft(output)
    wav_output = batch_to_file(batch_output)
    
    #put both input and output audio files on cpu if on gpu
    wav_output = wav_output.detach()
    
    #crop model out to match input
    input_length = wav_input.shape[-1]
    wav_output = wav_output[:, :input_length]
    
    #normalize output loudness
    if normalize:
        wav_output = normalize_loudness(wav_output, wav_input)
    
    return wav_output


### Decorators
#Reference note: to run server with uvicorn, type in terminal: uvicorn main:app --reload
app = FastAPI()

#required formatting to create endpoint healthcheck  
@app.get('/ping')
async def ping():
    return {"message": "pong"}


#to invoke the model, need to send data to /invocations
@app.post('/invocations')
def invoke(payload: Payload):
    input_audio_raw = payload.audio
    
    error = check_payload(input_audio_raw)
    if len(error) > 0:
        return {"error": error}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)
    input_tensor = torch.tensor(input_audio_raw).to(device)
    model_input = preprocess(input_tensor)
    output_tensor = predict(model, model_input)
    output_audio = postprocess(output_tensor, input_tensor)  
    
    if output_audio.device != torch.device('cpu'):
        output_audio = output_audio.cpu()
    return {"output_audio": output_audio.tolist()}


### Test decorators

##test load model function
@app.get('/test/load_model')
def test_load_model():
    model = load_model()
    return {"model": str(model)}


##test recieve payload
@app.post('/test/recieve_payload')
def test_recieve_payload(input_audio: Payload):
    return {'input': input_audio.audio}


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.tensor(input_audio.audio).to(device)
    model = load_model().to(device)
    input_tensor = torch.tensor(input_audio.audio).to(device)
    model_in = preprocess(input_tensor)
    output_tensor = predict(model, model_in)
    out_audio = postprocess(output_tensor, input_tensor)  
    
    if out_audio.device != torch.device('cpu'):
        out_audio = out_audio.cpu()
    return {"wav": out_audio.tolist()}