import argparse
import torch
import torchaudio
import os
from src.models import DPDCRN_Backbone
from src.utils import wav_to_complex_spec, complex_spec_to_wav

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True)
parser.add_argument('--input_wav', required=True)
parser.add_argument('--out_wav', default='out_enhanced.wav')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# instantiate student architecture matching config (small student)
student = DPDCRN_Backbone(in_ch=2, channels=[64,64], ft_hidden=64, n_ft=1).to(device)

d = torch.load(args.ckpt, map_location=device)
student.load_state_dict(d['student_state'])
student.eval()

wav, sr = torchaudio.load(args.input_wav)
if wav.ndim>1:
    wav = wav.mean(dim=0, keepdim=True)
wav = wav.to(device)
wav = wav.unsqueeze(0)

# use default STFT params consistent with config
nfft = 512
win = int(32 * 16000 / 1000)
hop = int(16 * 16000 / 1000)

spec = wav_to_complex_spec(wav, n_fft=nfft, hop=hop, win=win)
with torch.no_grad():
    out_spec, _ = student(spec)
est_spec = out_spec * spec
est_wav = complex_spec_to_wav(est_spec, n_fft=nfft, hop=hop, win=win, length=wav.shape[-1])
out_path = args.out_wav
# save
import soundfile as sf
sf.write(out_path, est_wav.cpu().numpy().squeeze(), sr)
print('Saved enhanced audio to', out_path)
