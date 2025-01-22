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
        torch.Tensor: audio tensor of shape (Batch, 1, frequency bins, time frames)
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


def normalize_loudness(audio: torch.Tensor, 
                       reference: torch.Tensor,
                       sample_rate: int = 8000,
                       n_fft: int = 512,
                       hop_length: int = 128,
                       max_magnitude: bool = True) -> torch.Tensor:
    """Normalize the audio tensor

    Args:
        audio (torch.Tensor): audio tensor to be normalized of shape (1, file length in samples)
        referenc (torch.Tensor): reference audio tensor of shape (1, file length in samples)
        sample_rate (int, optional): sample rate. Defaults to 8000.
        n_fft (int, optional): number of fft bins. Defaults to 512.
        hop_length (int, optional): hop length. Defaults to 128.
        max_magnitude (bool, optional): Caps strenght of normalized audio coefs at the max magnitude in reference

    Returns:
        torch.tensor: normalized audio tensor
    """
    audio_stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    reference_stft = torch.stft(reference, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    
    audio_mag = torch.abs(audio_stft)
    reference_mag = torch.abs(reference_stft)
    
    audio_power = torch.abs(audio_stft)**2
    reference_power = torch.abs(reference_stft)**2
    
    audio_db = 10 * torch.log10(torch.max(audio_power,1e-10))
    reference_db = 10 * torch.log10(torch.max(reference_power,1e-10))
    
    #need to calc offset
    
    #librosa
    #offset = convert.frequency_weighting(frequencies, kind=kind).reshape((-1, 1))
    #result: np.ndarray = offset + power_to_db(S, **kwargs)
    
    # Chat GPT way
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(freqs)
    
    a_weighting = torch.tensor(a_weighting, device=audio.device, dtype=audio_mag.dtype)
    
    audio_weighted = audio_mag * a_weighting[:, None]
    reference_weighted = reference_mag * a_weighting[:, None]
    
    # Compute perceptual loudness for each frame
    audio_loudness = audio_weighted.sum(dim=0)  # Sum across frequency bins
    reference_loudness = reference_weighted.sum(dim=0)

    # Optimize scaling factor c
    c = (reference_loudness / audio_loudness).mean().item()  # Simplified scalar scaling

    # Ensure no clipping by limiting the maximum magnitude
    if max_magnitude:
        max_audio_mag = (c * audio_mag).max()
        max_reference_mag = reference_mag.max()
        if max_audio_mag > max_reference_mag:
            c = c * (max_reference_mag / max_audio_mag)

    # Scale the input STFT
    normalized_audio_stft = c * audio_stft
    normalized_audio = torch.istft(normalized_audio_stft, n_fft=n_fft, hop_length=hop_length)
    return normalized_audio

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


if __name__ == "__main__":
    pass