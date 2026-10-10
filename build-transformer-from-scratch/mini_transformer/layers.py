"""Transformer 的基础积木：嵌入、注意力、FFN、归一化和 Block。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig


class TokenEmbedding(nn.Module):
    """将离散 token ID 映射为连续向量。"""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # sqrt(hidden_size) 缩放来自原始 Transformer，避免嵌入幅值过小。
        return self.embedding(input_ids) * math.sqrt(self.hidden_size)


class PositionalEncoding(nn.Module):
    """固定的正弦/余弦位置编码，不作为可训练参数更新。"""

    def __init__(self, hidden_size: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # hidden_size 为奇数时，奇数位置比偶数位置少一列。
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # buffer 会随模型迁移到 GPU、进入 state_dict，但不会参与梯度更新。
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"序列长度 {seq_len} 超过 max_seq_len={self.pe.size(1)}")
        return self.dropout(x + self.pe[:, :seq_len].to(dtype=x.dtype))


class MultiHeadAttention(nn.Module):
    """仅包含自注意力的多头注意力层。"""

    def __init__(self, hidden_size: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        # (B, L, H) -> (B, heads, L, head_dim)，每个头独立计算注意力。
        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.q_proj(x))
        key = split_heads(self.k_proj(x))
        value = split_heads(self.v_proj(x))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask 可为 (L,L)、(B,L,L) 或 (B,1,L,L)，PyTorch 会自动广播。
            scores = scores.masked_fill(~mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)

        # 合并所有头：(B, heads, L, head_dim) -> (B, L, H)。
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(context)


class FeedForward(nn.Module):
    """逐 token 的两层前馈网络。"""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class RMSNorm(nn.Module):
    """只按均方根缩放、不减均值的归一化层。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 计算可减少半精度训练时的数值误差，再转回原 dtype。
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms.to(dtype=x.dtype)) * self.weight


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block：注意力和 FFN 前各做一次归一化。"""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(config.hidden_size, config.num_heads, config.dropout)
        self.ffn = FeedForward(config.hidden_size, config.intermediate_size, config.dropout)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # 两个 “x + ...” 就是残差连接，帮助深层网络传播梯度。
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
