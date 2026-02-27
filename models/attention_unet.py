import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        )
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.block2(h)
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = x.view(B, C, H * W)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (C ** 0.5), dim=-1)
        x = torch.bmm(v, attn.transpose(1, 2))
        x = self.proj_out(x)
        x = x.view(B, C, H, W)
        return x + x_in


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.enc1 = ResidualBlock(in_channels, base_channels, time_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.attn2 = AttentionBlock(base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.attn3 = AttentionBlock(base_channels * 4)

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.attn_bottleneck = AttentionBlock(base_channels * 4)

        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_dim)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, time_dim)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embedding(t))

        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.downsample(x1), t_emb)
        x2 = self.attn2(x2)
        x3 = self.enc3(self.downsample(x2), t_emb)
        x3 = self.attn3(x3)

        b = self.bottleneck(self.downsample(x3), t_emb)
        b = self.attn_bottleneck(b)

        d3 = self.dec3(torch.cat([self.upsample(b), x3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), x2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), x1], dim=1), t_emb)

        return self.out_conv(d1)
