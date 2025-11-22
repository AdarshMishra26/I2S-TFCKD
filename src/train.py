# src/train.py
# Robust training entrypoint for I2S-TFCKD with config type coercion, checkpoints, AMP and TensorBoard.
import os, sys
# ensure project root is on sys.path so `import src.*` works when running `python src/train.py ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from src.models import DPDCRN_Backbone
from src.distill import TFCalibrationKD, ResidualFusion
from src.losses import MultiResolutionSTFTLoss
from src.data import make_dataloader
from src.utils import wav_to_complex_spec, complex_spec_to_wav

# -------------------------
# Helpers for robust config parsing
# -------------------------
def parse_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default

def parse_int(v, default):
    if v is None:
        return default
    if isinstance(v, int):
        return v
    try:
        return int(float(str(v)))
    except Exception:
        return default

def parse_float(v, default):
    if v is None:
        return default
    if isinstance(v, float):
        return v
    try:
        return float(str(v))
    except Exception:
        return default

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# -------------------------
# Read and coerce config values
# -------------------------
training_cfg = cfg.get('training', {})
data_cfg = cfg.get('data', {})
model_cfg = cfg.get('model', {})

lr = parse_float(training_cfg.get('lr'), 6e-4)
batch_size = parse_int(training_cfg.get('batch_size'), 4)
epochs = parse_int(training_cfg.get('epochs'), 20)
device_str = training_cfg.get('device', 'cuda')
save_every = parse_int(training_cfg.get('save_every'), 1)
resume_flag = parse_bool(training_cfg.get('resume'), True)
use_amp = parse_bool(training_cfg.get('use_amp'), False)
num_workers = parse_int(training_cfg.get('num_workers'), 4)

device = torch.device('cuda' if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
print(f"Using device: {device}  (torch.cuda.is_available()={torch.cuda.is_available()})")
print(f"lr={lr} batch_size={batch_size} epochs={epochs} use_amp={use_amp} resume={resume_flag}")

# -------------------------
# Build models
# -------------------------
teacher_cfg = model_cfg.get('backbone', {}).get('teacher', {})
student_cfg = model_cfg.get('backbone', {}).get('student', {})

teacher = DPDCRN_Backbone(
    in_ch=teacher_cfg.get('in_ch', 2),
    channels=teacher_cfg.get('channels', [128, 128]),
    ft_hidden=teacher_cfg.get('ft_hidden', 128),
    n_ft=teacher_cfg.get('n_ft', 4)
).to(device)

student = DPDCRN_Backbone(
    in_ch=student_cfg.get('in_ch', 2),
    channels=student_cfg.get('channels', [64, 64]),
    ft_hidden=student_cfg.get('ft_hidden', 64),
    n_ft=student_cfg.get('n_ft', 1)
).to(device)

# freeze teacher parameters
for p in teacher.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(student.parameters(), lr=lr)
mrstft = MultiResolutionSTFTLoss()
manager_tf = TFCalibrationKD(emb_hidden_factor=cfg.get('distillation', {}).get('tfckd', {}).get('emb_factor', 4))
fusion_module = ResidualFusion(
    in_channels_cur=teacher_cfg.get('channels', [128, 128])[1],
    in_channels_prev=teacher_cfg.get('channels', [128, 128])[1],
    out_channels=teacher_cfg.get('channels', [128, 128])[1]
)

# dataloader
train_loader = make_dataloader(
    cfg.get('data', {}).get('noisy_path'),
    cfg.get('data', {}).get('clean_path'),
    batch_size=batch_size,
    sr=data_cfg.get('sr', 16000),
    seg_seconds=data_cfg.get('seg_seconds', 2.5),
    num_workers=num_workers
)

# logging & checkpoints
writer = SummaryWriter()
os.makedirs('checkpoints', exist_ok=True)

# resume logic (robust)
start_epoch = 0
last_ckpt = os.path.join('checkpoints', 'ckpt_last.pt')
if resume_flag and os.path.exists(last_ckpt):
    try:
        d = torch.load(last_ckpt, map_location=device)
        student.load_state_dict(d['student_state'])
        optimizer.load_state_dict(d['optimizer_state'])
        start_epoch = int(d.get('epoch', 0))
        print(f"Resumed from {last_ckpt} (starting epoch {start_epoch})")
    except Exception as e:
        print("Warning: failed to resume from checkpoint:", e)

# AMP scaler
scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

# STFT params from config
nfft = data_cfg.get('n_fft', 512)
win = int(parse_float(data_cfg.get('window_ms', 32) * data_cfg.get('sr', 16000) / 1000, 512)) if isinstance(data_cfg.get('window_ms'), (int, float)) else int(parse_float(data_cfg.get('window_ms'), 32) * data_cfg.get('sr', 16000) / 1000)
hop = int(parse_float(data_cfg.get('hop_ms', 16) * data_cfg.get('sr', 16000) / 1000, 160)) if isinstance(data_cfg.get('hop_ms'), (int, float)) else int(parse_float(data_cfg.get('hop_ms'), 16) * data_cfg.get('sr', 16000) / 1000)

# Training loop
for epoch in range(start_epoch, epochs):
    student.train()
    total_loss = 0.0
    for i, (noisy, clean) in enumerate(train_loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        spec = wav_to_complex_spec(noisy, n_fft=nfft, hop=hop, win=win)

        with torch.no_grad():
            t_out, t_feats = teacher(spec)

        s_out, s_feats = student(spec)
        est_spec = s_out * spec
        est_wav = complex_spec_to_wav(est_spec, n_fft=nfft, hop=hop, win=win, length=clean.shape[-1])

        loss_backbone = mrstft(est_wav, clean) + torch.nn.functional.l1_loss(est_wav, clean)

        # simplified intra/inter demo; wrap in try in case layers count differ
        try:
            L_intra = manager_tf(s_feats['ft'][0], t_feats['ft'][0])
        except Exception:
            L_intra = torch.tensor(0.0, device=device)
        try:
            L_inter = manager_tf(s_feats['encoder'][-1], t_feats['encoder'][-1])
        except Exception:
            L_inter = torch.tensor(0.0, device=device)

        loss = loss_backbone + L_intra + L_inter

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())

        if i % 10 == 0:
            print(f"Epoch {epoch+1} [{i}/{len(train_loader)}] loss={loss.item():.4f}")

    avg_loss = total_loss / max(1, len(train_loader))
    writer.add_scalar('loss/train', avg_loss, epoch+1)
    print(f"Epoch {epoch+1} completed. avg_loss={avg_loss:.4f}")

    # save checkpoints
    ckpt = {'epoch': int(epoch+1), 'student_state': student.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, os.path.join('checkpoints', f'ckpt_epoch_{epoch+1}.pt'))
    if (epoch + 1) % save_every == 0:
        torch.save(ckpt, os.path.join('checkpoints', 'ckpt_last.pt'))

writer.close()
