import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-10
MIN_DB = -300

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
    """Write the audio tensor to a file

    Args:
        filepath (str): path to save the audio file
        audio (torch.Tensor): audio tensor
        sr (int, optional): sample rate. Defaults to 8000.
    """
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
               hop_len: int = 128, 
             win: int = 512) -> torch.Tensor:
    """Transforms batched audio to STFT coefficients

    Args:
        data (torch.Tensor): audio tensor of shape (Batch, 1, clip_length)
        hop_len (int, optional): hop length. Defaults to 128.
        win (int, optional): window size. Defaults to 512.

    Returns:
        torch.Tensor: audio tensor of shape (Batch, 1, frequency bins, time frames)
    """
    data = data.squeeze(1)
    
    stft_coefs = torch.stft(data
                          ,n_fft=win
                          ,window=torch.hann_window(win)
                          ,hop_length=hop_len
                          ,normalized=True
                          ,return_complex=True)
    return stft_coefs.unsqueeze(1)


def batch_istft(data: torch.Tensor, 
                hop_len: int = 128, 
                win: int = 512) -> torch.Tensor:
    """Converts batched audio from STFT coefficients to audio

    Args:
        data (torch.Tensor): complex tensor of shape (Batch, 1, frequency bins, time frames)
        sr (int, optional): sample rate. Defaults to 8000.
        hop_len_s (float, optional): hop length in seconds. Defaults to .016.
        win_s (float, optional): window seize in seconds. Defaults to .064.

    Returns:
        torch.Tensor: torch tensor with shape (Batch, 1, clip_length)
    """
    data = data.squeeze(1)
    
    wavs = torch.istft(data
                      ,n_fft=win
                      ,window=torch.hann_window(win,device=data.device)
                      ,hop_length=hop_len)
    
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
                       n_fft: int = 2048,
                       hop_length: int = 128,
                       threshold:float = 1) -> torch.Tensor:
    """Normalize the audio tensor

    Args:
        audio (torch.Tensor): audio tensor to be normalized of shape (1, file length in samples)
        referenc (torch.Tensor): reference audio tensor of shape (1, file length in samples)
        n_fft (int, optional): number of fft bins. Defaults to 512.
        hop_length (int, optional): hop length. Defaults to 128.
        threshold (float, optional): if coef magnitude has reduced by factor > threshold, scaling factor not applied
        
    Returns:
        torch.tensor: normalized audio tensor (1, file length in samples)
    """
    #if audio and reference have different shapes, device, or dtype, no normalization
    if audio.shape != reference.shape or audio.dtype != reference.dtype or audio.device != reference.device:
        return audio
    
    window = torch.hann_window(window_length = n_fft, device = audio.device)
    audio_stft = torch.stft(audio, 
                            n_fft=n_fft, 
                            hop_length=hop_length, 
                            window=window, 
                            return_complex=True)
    reference_stft = torch.stft(reference, 
                                n_fft=n_fft, 
                                hop_length=hop_length, 
                                window=window, 
                                return_complex=True)
    
    audio_mag = torch.abs(audio_stft)
    reference_mag = torch.abs(reference_stft)
    
    audio_mag = torch.clamp(audio_mag, min=EPSILON)
    reference_mag = torch.clamp(reference_mag, min=EPSILON)
    
    #Calculate frame wise scaling factors used to match percieved loudness of audio to reference
    #shape: (B, bins, frames)
    bins = audio_mag.shape[1]
    
    #geometric mean of magnitudes at each frame, scales audio to match intensity while keeping
    #ratio of coefs produced by model out
    log_scaling_ref = torch.log10(reference_mag)
    log_scaling_audio = torch.log10(audio_mag)
    log_scaling_factors = (torch.sum(log_scaling_ref, dim = 1, keepdim=True) - torch.sum(log_scaling_audio, dim = 1, keepdim=True))/bins
    frame_scaling_factors = 10**log_scaling_factors
    
    #calc pointwise scaling factors to scale each coef based on how much the net reduced its mag
    point_scaling_factors = reference_mag/audio_mag
    
    #cant increase mag of a coeffient to a level above input
    cap_scaling_factor = torch.min(point_scaling_factors, frame_scaling_factors.expand(-1, bins, -1))

    #simple, min max normalization for both audio and reference. 
    #If relative loudness of a coef has reduced by more than 5, replace factor by one
    audio_mag_max = torch.max(audio_mag)
    audio_mag_min = torch.min(audio_mag)
    audio_mag_norm = (audio_mag - audio_mag_min)/(audio_mag_max- audio_mag_min)
    
    reference_mag_max = torch.max(reference_mag)
    reference_mag_min = torch.min(reference_mag)
    reference_mag_norm = (reference_mag - reference_mag_min)/(reference_mag_max - reference_mag_min)
    
    mag_reduction_ratios = reference_mag_norm/audio_mag_norm
    cap_scaling_factor = torch.where(mag_reduction_ratios > threshold, torch.ones_like(cap_scaling_factor), cap_scaling_factor)

    cap_scaled_stft = audio_stft * cap_scaling_factor #(B, bins, frames)
    
    out_audio = torch.istft(cap_scaled_stft, 
                            n_fft=n_fft, 
                            hop_length=hop_length, 
                            window=window)
    return out_audio


# Pending
def spec_plot_old(audio, sr=8000, n_fft=512, hop_length=128, save_png=False, png_name='test.png'):
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X)) 
    
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    
    if save_png:
        plt.savefig(png_name)
        
    plt.show()


def analysis_plots(specs: dict):
    """Plot visuals for a set of spectrograms

    Args:
        specs (dict): dictionary of torch tensors with dim (1, frequency bins, time frames)  or (frequency bins, time frames)
    """
    #clean up
    for name in specs:
        spec = specs[name]
        if spec.device != torch.device('cpu'):
            spec = spec.cpu()
        if len(spec.shape) == 3:
            spec = spec.squeeze(0)
        bound_min = torch.full(spec.shape, MIN_DB)
        spec = torch.max(spec, bound_min)
        spec = spec.numpy()
        specs[name] = spec
        print(name)
        print(np.max(spec))
        print(np.min(spec))
        print(np.mean(spec))
    
     # Calculate global min and max for consistent scales
    all_data = np.concatenate([spec.flatten() for spec in specs.values()])
    vmin, vmax = np.min(all_data), np.max(all_data)

    # First Plot: Spectrograms side by side with consistent color scale
    num_specs = len(specs)
    fig, axes = plt.subplots(1, num_specs, figsize=(5 * num_specs, 5), sharey=True)
    if num_specs == 1:
        axes = [axes]  # Ensure axes is iterable for a single spectrogram

    for ax, (name, spec) in zip(axes, specs.items()):
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Frequency Bins')
        fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.show(block=True)  # Keep the plot open for inspection

    # Second Plot: Frame-wise mean decibel levels
    plt.figure(figsize=(10, 6))
    for name, spec in specs.items():
        frame_means = np.mean(spec, axis=0)  # Average over frequency bins
        plt.plot(frame_means, label=name)
    
    plt.title('Frame-wise Mean Decibel Levels')
    plt.xlabel('Time Frames')
    plt.ylabel('Mean Decibel Level (dB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    test_in = load_wav('./test-files/two-speech-low-drone_in.wav')
    test_out = load_wav('./test-files/two-speech-low-drone_out_old.wav')
    
    outs = normalize_loudness(test_out, test_in)
    # out_file = batch_to_file(batch_istft(outs['stft']))
    
    # write_wav('./test-files/zzz.wav', out_file)

    
    