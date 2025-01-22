import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt


def load_wav(filepath: str, sr: int = 8000) -> torch.Tensor:
    """Load the audio file from the filepath

    Args:
        filepath (str): _description_
        sr (int, optional): sample rate. Defaults to 8000.

    Returns:
        torch.Tensor: tensor representation of audio file
    """
    audio, nat_sr = torchaudio.load(filepath) #dim: ch x samples 
    if sr is not None and sr != nat_sr :
        audio = torchaudio.functional.resample(audio, nat_sr, sr)
    return audio

def write_wav(filepath: str, audio: torch.Tensor, sr: int = 8000) -> None:
    torchaudio.save(uri=filepath, src=audio, sample_rate=sr)


def file_to_batch(wav: torch.Tensor, clip_length: int=32640) -> torch.Tensor:
    """Convert the input audio file to a batch of audio tensors.

    Args:
        wav (torch.Tensor): audio tensor of shape (1, file length in samples)
        clip_length (int, optional): Length of each batch item in samples. Defaults to 32640.

    Returns:
        torch.Tensor: tensor of shape (Batch, 1, clip_length)
    """
    n = wav.shape[1]
    
    pad_len = clip_length - (n % clip_length)
    pad = torch.zeros((1,pad_len))
    wav = torch.cat((wav,pad),dim=1)

    wav_batch = wav.reshape(-1,1,clip_length)
    return wav_batch

def batch_stft(data: torch.Tensor, 
               sr: int=8000, 
               hop_len_s: float=.016, 
               win_s: float=.064) -> torch.Tensor:
    """Transforms batched audio to STFT coefficients

    Args:
        data (torch.Tensor): audio tensor of shape (Batch, 1, clip_length)
        sr (int, optional): sample rate. Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): window size in seconds. Defaults to .064.

    Returns:
        torch.Tensor: audio tensor of shape (Batch, 1, frequency bins, time frames, 2)
    """
    hl_sam = int(round(hop_len_s*sr))
    wl_sam = int(round(win_s*sr))

    data = data.squeeze(1)
    
    stft_coefs = torch.stft(data
                          ,n_fft=wl_sam
                          ,window=torch.hann_window(wl_sam)
                          ,hop_length=hl_sam
                          ,normalized=True
                          ,return_complex=True)
    stft_coefs = torch.view_as_real(stft_coefs)
    return stft_coefs.unsqueeze(1)


def batch_istft(data: torch.Tensor, 
                sr: int=8000, 
                hop_len_s: float=.016, 
                win_s: float=.064) -> torch.Tensor:
    """Converts batched audio from STFT coefficients to audio

    Args:
        data (torch.Tensor): complex tensor of shape (Batch, 1, frequency bins, time frames)
        sr (int, optional): sample rate. Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): window seize in seconds. Defaults to .064.

    Returns:
        torch.Tensor: torch tensor with shape (Batch, 1, clip_length)
    """
    hl_sam = int(round(hop_len_s*sr))
    wl_sam = int(round(win_s*sr))

    data = data.squeeze(1)
    
    wavs = torch.istft(data
                      ,n_fft=wl_sam
                      ,window=torch.hann_window(wl_sam,device=data.device)
                      ,hop_length=hl_sam)
    
    return wavs.unsqueeze(1)

def batch_to_file(batch: torch.Tensor) -> torch.Tensor:
    """Converts batched audio to a single audio file

    Args:
        batch (torch.Tensor): audio tensor of shape (Batch, 1, clip_length)

    Returns:
        torch.Tensor: audio tensor of shape (1, Batch*clip_length)
    """
    wav = batch.reshape(1,-1)
    return wav


# Pending
def spec_plot(audio, sr=8000, n_fft=512, hop_length=128, save_png=False, png_name='test.png'):
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X)) 
    
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    
    if save_png:
        plt.savefig(png_name)
        
    plt.show()



    