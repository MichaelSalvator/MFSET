import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA(nn.Module):
    def __init__(self, dim, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: [B, S, D]
        y = x.mean(dim=1, keepdim=True)      # [B, 1, D]
        y = self.conv(y)                     # [B, 1, D]
        y = self.sigmoid(y)                  # [B, 1, D]
        return x * y                         # [B, S, D]

class GEGLU(nn.Module):
    def __init__(self, d_model, d_ff, p=0.3):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff * 2)
        self.proj = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.gelu(a) * b)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead, k, dropout=0.3):
        super().__init__()
        self.nhead = nhead
        self.k = k
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(k / nhead)))
        self.proj_q = nn.Linear(hidden_dim, k)
        self.proj_k = nn.Linear(hidden_dim, k)
        self.proj_v = nn.Linear(hidden_dim, k)
        self.proj_out = nn.Linear(k, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.eca = ECA(hidden_dim)
        self.ffn = nn.Sequential(
            GEGLU(hidden_dim, hidden_dim * 2, dropout),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        residual = x
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        b, s, _ = q.shape
        q = q.view(b, s, self.nhead, self.k // self.nhead).transpose(1, 2)
        k = k.view(b, s, self.nhead, self.k // self.nhead).transpose(1, 2)
        v = v.view(b, s, self.nhead, self.k // self.nhead).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(b, s, self.k)
        x = self.proj_out(x)
        x = self.ln1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.eca(x)
        x = self.ln2(x + residual)
        return x

class ImprovedFixedAdaptiveTransformerV3(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=32, nhead=4, k=32, max_seq_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.max_seq_len = max_seq_len
        self.block1 = TransformerBlock(hidden_dim, nhead, k)
        self.block2 = TransformerBlock(hidden_dim, nhead, k)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def _get_pos_encoding(self, seq_len, d_model, device):
        pos = torch.arange(seq_len,                                                                                   device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        return pe
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, s, _ = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        pe = self._get_pos_encoding(s + 1, x.size(-1), x.device)
        x = x + pe
        x = self.block1(x)
        x = self.block2(x)
        x = x[:, 0, :]
        return self.fc(x)
