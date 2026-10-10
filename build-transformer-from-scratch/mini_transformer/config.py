"""模型配置。

把超参数集中在一个 dataclass 中，便于保存、打印和构造不同规模的模型。
"""

from dataclasses import dataclass


@dataclass
class LMConfig:
    """Decoder-only 语言模型的结构超参数。"""

    vocab_size: int = 6400
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int | None = None
    max_seq_len: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        # FFN 的中间维度通常是隐藏维度的 4 倍；仍允许调用者显式覆盖。
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须大于 0")
        if self.hidden_size <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("hidden_size、num_layers 和 num_heads 必须大于 0")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size 必须大于 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len 必须大于 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间")
