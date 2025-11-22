import torch
import torch.nn.functional as F
import torch

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, fft_sizes=[512,1024,2048], hop_sizes=[128,256,512], win_lengths=[512,1024,2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
    def forward(self, x, y):
        loss = 0.0
        for nfft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            w = torch.hann_window(win).to(x.device)
            X = torch.stft(x, n_fft=nfft, hop_length=hop, win_length=win, window=w, return_complex=True)
            Y = torch.stft(y, n_fft=nfft, hop_length=hop, win_length=win, window=w, return_complex=True)
            magX = torch.abs(X)
            magY = torch.abs(Y)
            sc = torch.norm(magY - magX, p='fro') / (torch.norm(magY, p='fro') + 1e-8)
            log_mag_loss = F.l1_loss(torch.log(magX+1e-7), torch.log(magY+1e-7))
            loss = loss + sc + log_mag_loss
        return loss
