from typing import List
from fastapi import FastAPI, status
from pydantic import BaseModel
import torch
import os


def load_model(model_dir: str):
    """Load the model from the model_dir.
    Args:
        model_dir (str): path to the directory containing model.pth
        
    Returns:
        torch.jit.ScriptModule
    """
    model_path = os.path.join(model_dir, 'model.pth')
    model = torch.jit.load(model_path)
    # if torch.cuda.is_available():
    #     model.to('cuda') 
    return model

def file_to_batch(wav, clip_length=32640):
    """Convert the input audio file to a batch of audio tensors.

    Args:
        wav (torch.tensor): audio tensor of shape (1, file length in samples)
        clip_length (int, optional): Length of each batch item in samples. Defaults to 32640.

    Returns:
        torch.tensor: tensor of shape (Batch, 1, clip_length)
    """
    n = wav.shape[1]
    
    pad_len = clip_length - (n % clip_length)
    pad = torch.zeros((1,pad_len))
    wav = torch.cat((wav,pad),dim=1)

    wav_batch = wav.reshape(-1,1,clip_length)
    return wav_batch

def batch_stft(data, sr=8000, hop_len_s=.016, win_s=.064):
    #out dim (B, 1, f_bins, h_bins,2)
    """Transforms batched audio to STFT coefficients

    Args:
        data (torch.tensor): audio tensor of shape (Batch, 1, clip_length)
        sr (int, optional): sample rate. Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): window size in seconds. Defaults to .064.

    Returns:
        torch.tensor: audio tensor of shape (Batch, 1, frequency bins, time frames, 2)
    """
    hl_sam = int(round(hop_len_s*sr))
    wl_sam = int(round(win_s*sr))

    data = data.squeeze(1)
    
    #return_complex provides extra dim that contains real vals corresponding to real and im parts
    stft_coefs = torch.stft(data
                          ,n_fft=wl_sam
                          ,window=torch.hann_window(wl_sam)
                          ,hop_length=hl_sam
                          ,normalized=True
                          ,return_complex=True)
    stft_coefs = torch.view_as_real(stft_coefs)
    
    return stft_coefs.unsqueeze(1)

def preprocess(input_audio):
    """Preprocess the input audio.

    Args:
        input_audio (_type_): Open

    Returns:
        torch.tensor: audio tensor of shape (Batch, 1, frequency bins, time frames, 2)
    """
    audio_tensor = torch.tensor(input_audio) #convert input audio to audio tensor
    batch = file_to_batch(audio_tensor)
    batch_stft = batch_stft(batch)
    return batch_stft

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

def batch_istft(data, sr=8000, hop_len_s=.016, win_s=.064):
    """Converts batched audio from STFT coefficients to audio

    Args:
        data (torch.tensor): complex tensor of shape (Batch, 1, frequency bins, time frames)
        sr (int, optional): sample rate. Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): window seize in seconds. Defaults to .064.

    Returns:
        torch.tensor: torch tensor with shape (Batch, 1, clip_length)
    """
    hl_sam = int(round(hop_len_s*sr))
    wl_sam = int(round(win_s*sr))

    data = data.squeeze(1)
    
    wavs = torch.istft(data
                      ,n_fft=wl_sam
                      ,window=torch.hann_window(wl_sam,device=data.device)
                      ,hop_length=hl_sam)
    
    return wavs.unsqueeze(1)

def batch_to_file(batch):
    """Converts batched audio to a single audio file

    Args:
        batch (torch.tensor): audio tensor of shape (Batch, 1, clip_length)

    Returns:
        torch.tensor: audio tensor of shape (1, Batch*clip_length)
    """
    wav = batch.reshape(1,-1)
    return wav

def postprocess():
    pass

def load_payload():
    pass

app = FastAPI()

class Payload(BaseModel):
    input: List[float] #what is this?

#required healthcheck  
@app.get('/ping')
def ping():
    return "pong"

#to invoke the model, need to send data to /invocations
@app.post('/invocations')
def invoke():
    pass