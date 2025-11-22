import torch

def wav_to_complex_spec(wav, n_fft=512, hop=160, win=400):
    w = torch.hann_window(win).to(wav.device)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
    stft = stft.permute(0,2,1)
    real = stft.real.unsqueeze(1)
    imag = stft.imag.unsqueeze(1)
    spec = torch.cat([real, imag], dim=1)
    return spec

def complex_spec_to_wav(spec_complex, n_fft=512, hop=160, win=400, length=None):
    real = spec_complex[:,0,...]
    imag = spec_complex[:,1,...]
    comp = torch.complex(real, imag).permute(0,2,1)
    w = torch.hann_window(win).to(comp.device)
    wav = torch.istft(comp, n_fft=n_fft, hop_length=hop, win_length=win, window=w, length=length)
    return wav
