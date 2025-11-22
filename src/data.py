import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import torch.nn.functional as F

class NoisyCleanDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sr=16000, seg_seconds=2.5):
        self.noisy_files = sorted([os.path.join(noisy_dir,f) for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_files = sorted([os.path.join(clean_dir,f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.sr = sr
        self.seg_len = int(seg_seconds * sr)
    def __len__(self):
        return min(len(self.noisy_files), len(self.clean_files))
    def __getitem__(self, idx):
        noisy, _ = torchaudio.load(self.noisy_files[idx])
        clean, _ = torchaudio.load(self.clean_files[idx])
        noisy = noisy.mean(dim=0)[:self.seg_len]
        clean = clean.mean(dim=0)[:self.seg_len]
        if noisy.numel() < self.seg_len:
            pad = self.seg_len - noisy.numel()
            noisy = F.pad(noisy, (0,pad))
            clean = F.pad(clean, (0,pad))
        return noisy, clean

def make_dataloader(noisy_dir, clean_dir, batch_size=4, sr=16000, seg_seconds=2.5, num_workers=4):
    ds = NoisyCleanDataset(noisy_dir, clean_dir, sr, seg_seconds)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
