"""完整的 Decoder-only Transformer 语言模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LMConfig
from .layers import PositionalEncoding, RMSNorm, TokenEmbedding, TransformerBlock


class Transformer(nn.Module):
    """用于“下一个 token 预测”的自回归语言模型。"""

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = PositionalEncoding(config.hidden_size, config.max_seq_len, config.dropout)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # PyTorch 的 Embedding 默认初始化幅值偏大。语言模型通常采用较小的正态
        # 初始化，使训练开始时的 logits 和交叉熵保持在合理范围。
        self.apply(self._init_weights)

        # 输入嵌入与输出分类器共享权重，既减少参数量，也常能改善语言建模效果。
        self.lm_head.weight = self.embedding.embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """执行前向传播，返回 ``(loss, logits)``。

        ``input_ids`` 形状为 ``(B, L)``，输出 logits 形状为
        ``(B, L, vocab_size)``。传入 labels 时，第 t 个位置预测第 t+1 个 token。
        padding 或无需计入损失的位置应在 labels 中填为 -100。
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是形状为 (batch_size, seq_len) 的二维张量")
        seq_len = input_ids.size(1)
        if seq_len == 0:
            raise ValueError("输入序列不能为空")
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过 max_seq_len={self.config.max_seq_len}")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels 与 input_ids 的形状必须一致")

        # 下三角为 True：位置 t 只能看到 0..t，不能偷看未来 token。
        causal_mask = torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool).tril()
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

        x = self.pos_encoding(self.embedding(input_ids))
        for layer in self.layers:
            x = layer(x, causal_mask)
        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        eos_token_id: int | None = 2,
    ) -> torch.Tensor:
        """通过温度采样逐 token 生成文本。

        模型超过上下文窗口后只读取最近 ``max_seq_len`` 个 token，但返回值仍保留
        完整输入。模型未经训练时生成内容没有语义，这是正常现象。
        """
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids 必须是非空的二维张量")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能小于 0")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        was_training = self.training
        self.eval()
        generated = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len :]
            _, logits = self(context)
            probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if eos_token_id is not None:
                # 已结束的样本后续始终补 eos，让不同 batch 样本可保持相同长度。
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token.squeeze(1).eq(eos_token_id)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and finished.all():
                break

        self.train(was_training)
        return generated
