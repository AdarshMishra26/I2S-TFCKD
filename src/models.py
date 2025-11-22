import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(c_in, c_out, kernel=(1,3), stride=(1,2), padding=(0,1)):
    return nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding)

def deconv2d(c_in, c_out, kernel=(1,3), stride=(1,2), padding=(0,1), output_padding=(0,1)):
    return nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding)

class DilatedConvBlock(nn.Module):
    def __init__(self, channels, kernel=(2,3), padd=(1,1), dilations=[1,2,4,8]):
        super().__init__()
        layers = []
        for d in dilations:
            layers.append(nn.Conv2d(channels, channels, kernel_size=kernel, padding=(padd[0]*d,padd[1]), dilation=(1,d)))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.PReLU())
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class FTModule(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.gru_t = nn.GRU(input_size=channels, hidden_size=hidden, batch_first=True)
        self.gru_f = nn.GRU(input_size=channels, hidden_size=hidden, batch_first=True)
        self.proj = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        B,C,T,F = x.shape
        t_in = x.permute(0,3,2,1).contiguous().view(B*F, T, C)
        t_out, _ = self.gru_t(t_in)
        t_feat = t_out[:, -1, :].view(B, F, -1).permute(0,2,1).unsqueeze(-1)

        f_in = x.permute(0,2,3,1).contiguous().view(B*T, F, C)
        f_out, _ = self.gru_f(f_in)
        f_feat = f_out[:, -1, :].view(B, T, -1).permute(0,2,1).unsqueeze(-1)

        t_exp = t_feat.expand(-1, -1, T, -1)
        f_exp = f_feat.expand(-1, -1, -1, F)
        fused = t_exp + f_exp
        out = self.proj(fused)
        return out

class DPDCRN_Backbone(nn.Module):
    def __init__(self, in_ch=2, channels=[128,128], ft_hidden=128, n_ft=4):
        super().__init__()
        self.conv1 = conv2d(in_ch, channels[0], kernel=(1,3), stride=(1,2), padding=(0,1))
        self.conv2 = conv2d(channels[0], channels[1], kernel=(1,3), stride=(1,2), padding=(0,1))
        self.dilated_blocks = nn.ModuleList([DilatedConvBlock(channels[1]) for _ in range(4)])
        self.ft_modules = nn.ModuleList([FTModule(channels[1], ft_hidden) for _ in range(n_ft)])
        self.deconv1 = deconv2d(channels[1], channels[0], kernel=(1,3), stride=(1,2), padding=(0,1), output_padding=(0,1))
        self.deconv2 = deconv2d(channels[0], 2, kernel=(1,3), stride=(1,2), padding=(0,1), output_padding=(0,1))

    def forward(self, spec_complex):
        x = self.conv1(spec_complex)
        x = self.conv2(x)
        enc_feats = []
        for block in self.dilated_blocks:
            x = block(x)
            enc_feats.append(x)
        ft_feats = []
        for ft in self.ft_modules:
            x = x + ft(x)
            ft_feats.append(x)
        for block in reversed(self.dilated_blocks):
            x = block(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x, {"encoder": enc_feats, "ft": ft_feats}
