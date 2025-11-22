import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TFCalibrationKD(nn.Module):
    def __init__(self, emb_hidden_factor=4, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.embQ_T = nn.Sequential(nn.Linear(1, emb_hidden_factor), nn.ReLU(), nn.LayerNorm(emb_hidden_factor))
        self.embK_T = nn.Sequential(nn.Linear(1, emb_hidden_factor), nn.ReLU(), nn.LayerNorm(emb_hidden_factor))
        self.embQ_F = nn.Sequential(nn.Linear(1, emb_hidden_factor), nn.ReLU(), nn.LayerNorm(emb_hidden_factor))
        self.embK_F = nn.Sequential(nn.Linear(1, emb_hidden_factor), nn.ReLU(), nn.LayerNorm(emb_hidden_factor))

    def compute_time_selfsim(self, feat):
        B,C,T,F = feat.shape
        mf = feat.permute(0,2,1,3).contiguous().view(B, T, C*F)
        mf = F.normalize(mf, dim=-1, eps=self.eps)
        PT = torch.matmul(mf, mf.transpose(-1,-2))
        PT = (PT + 1.0)/2.0
        return PT

    def compute_freq_selfsim(self, feat):
        B,C,T,F = feat.shape
        mf = feat.permute(0,3,1,2).contiguous().view(B, F, C*T)
        mf = F.normalize(mf, dim=-1, eps=self.eps)
        PF = torch.matmul(mf, mf.transpose(-1,-2))
        PF = (PF + 1.0)/2.0
        return PF

    def emb_and_alpha(self, P_s, P_t, flow='T'):
        B, Ls, _ = P_s.shape
        Lt = P_t.shape[1]
        q_s = P_s.mean(dim=-1, keepdim=True)
        k_t = P_t.mean(dim=-1, keepdim=True)
        if flow == 'T':
            q_emb = self.embQ_T(q_s).squeeze(-1)
            k_emb = self.embK_T(k_t).squeeze(-1)
        else:
            q_emb = self.embQ_F(q_s).squeeze(-1)
            k_emb = self.embK_F(k_t).squeeze(-1)
        attn = torch.matmul(q_emb, k_emb.transpose(-1,-2)) / math.sqrt(q_emb.size(-1)+1e-8)
        alpha = F.softmax(attn, dim=-1)
        return alpha

    def tfckd_loss(self, feat_s, feat_t):
        PTs = self.compute_time_selfsim(feat_s)
        PTt = self.compute_time_selfsim(feat_t)
        PFs = self.compute_freq_selfsim(feat_s)
        PFt = self.compute_freq_selfsim(feat_t)
        alphaT = self.emb_and_alpha(PTs, PTt, flow='T')
        alphaF = self.emb_and_alpha(PFs, PFt, flow='F')
        PTs_row = F.softmax(PTs, dim=-1) + 1e-8
        PTt_row = F.softmax(PTt, dim=-1) + 1e-8
        PFs_row = F.softmax(PFs, dim=-1) + 1e-8
        PFt_row = F.softmax(PFt, dim=-1) + 1e-8

        def pairwise_kl(Ps, Pt):
            Ls = Ps.size(1); Lt = Pt.size(1)
            if Ls != Lt:
                Pt_res = F.interpolate(Pt.unsqueeze(1), size=(Ls, Ls), mode='bilinear', align_corners=False).squeeze(1)
                Ps_res = Ps
            else:
                Pt_res = Pt
                Ps_res = Ps
            Pt_rows_expand = Pt_res.unsqueeze(2).expand(-1, -1, Ls, -1)
            Ps_rows_expand = Ps_res.unsqueeze(1).expand(-1, Lt, -1, -1)
            PT = Pt_rows_expand
            PS = Ps_rows_expand
            kl = (PT * (torch.log(PT + 1e-8) - torch.log(PS + 1e-8))).sum(dim=-1)
            kl = kl.permute(0,2,1)
            return kl

        KL_time = pairwise_kl(PTs_row, PTt_row)
        KL_freq = pairwise_kl(PFs_row, PFt_row)
        loss_time = (alphaT * KL_time).sum() / (alphaT.numel() + 1e-8)
        loss_freq = (alphaF * KL_freq).sum() / (alphaF.numel() + 1e-8)
        return loss_time + loss_freq

    def forward(self, feat_s, feat_t):
        return self.tfckd_loss(feat_s, feat_t)

class ResidualFusion(nn.Module):
    def __init__(self, in_channels_cur, in_channels_prev, out_channels, upsample_scale=(1,2)):
        super().__init__()
        self.conv_align_prev = nn.Conv2d(in_channels_prev, out_channels, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=upsample_scale, mode='nearest')
        self.conv_align_cur = nn.Conv2d(in_channels_cur, out_channels, kernel_size=1)
        self.conv1d_att = nn.Conv1d(out_channels*2, 2, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, F_j, R_jm1):
        R_prev = self.conv_align_prev(R_jm1)
        F_cur = self.up(F_j)
        F_cur = self.conv_align_cur(F_cur)
        cat = torch.cat([F_cur, R_prev], dim=1)
        B, Ch, T, Fr = cat.shape
        cat_flat = cat.view(B, Ch, T*Fr)
        att = self.conv1d_att(cat_flat)
        att = self.sig(att).view(B,2,T,Fr)
        Aj_R = att[:,0:1,:,:]
        Aj_F = att[:,1:2,:,:]
        Rj = Aj_R * R_prev + Aj_F * F_cur
        Tj = self.conv_out(Rj)
        return Tj, Rj
