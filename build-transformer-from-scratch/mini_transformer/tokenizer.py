"""BPE 分词器的训练与推理封装。"""

from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_bpe_tokenizer(
    texts: Iterable[str] | Sequence[str | PathLike[str]],
    vocab_size: int = 6400,
    save_path: str | PathLike[str] = "my_tokenizer.json",
) -> Tokenizer:
    """训练并保存一个 ByteLevel BPE 分词器。

    ``texts`` 有两种用法：

    1. 传入文本字符串的可迭代对象；
    2. 传入一个或多个文本文件路径。

    ByteLevel 会将 UTF-8 字节映射到可打印字符，因此 tokenizer.json 中看到的
    中文 token 可能像乱码；通过这里配置的 ByteLevel decoder 可以正确还原。
    """
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size 不能小于特殊 token 数量 {len(SPECIAL_TOKENS)}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    # 只有具体的序列才能可靠判断“所有元素是否都是文件路径”。普通生成器直接
    # 作为训练文本迭代器使用，避免为了探测首元素而意外丢掉一条数据。
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts 应为文本列表/迭代器或文件路径列表，不能是单个字符串")
    if isinstance(texts, Sequence):
        paths = [Path(item) for item in texts]
        try:
            all_are_files = bool(paths) and all(path.is_file() for path in paths)
        except OSError:
            # 很长的普通文本可能不是合法文件名，应将其视为训练语料而非路径。
            all_are_files = False
        if all_are_files:
            tokenizer.train([str(path) for path in paths], trainer)
        else:
            tokenizer.train_from_iterator((str(item) for item in texts), trainer)
    else:
        tokenizer.train_from_iterator(texts, trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"分词器已保存到 {output_path}，词表大小: {tokenizer.get_vocab_size()}")
    return tokenizer


class BPETokenizer:
    """为训练好的 Hugging Face Tokenizer 提供简单、明确的教学接口。"""

    def __init__(self, tokenizer_path: str | PathLike[str] = "my_tokenizer.json") -> None:
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_id = self._required_token_id("<pad>")
        self.bos_id = self._required_token_id("<bos>")
        self.eos_id = self._required_token_id("<eos>")
        self.unk_id = self._required_token_id("<unk>")

    def _required_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"分词器中缺少必需的特殊 token: {token}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, max_length: int | None = None, padding: bool = False) -> list[int]:
        """把文本编码为 token ID，可选截断和定长补齐。"""
        ids = self.tokenizer.encode(text).ids
        if max_length is not None:
            if max_length <= 0:
                raise ValueError("max_length 必须大于 0")
            ids = ids[:max_length]
            if padding:
                ids += [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """把 token ID 还原成文本。"""
        ids = list(ids)
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.tokenizer.decode(ids)

    def tokenize(self, text: str) -> list[str]:
        """返回便于观察 BPE 切分结果的 token 字符串列表。"""
        return self.tokenizer.encode(text).tokens
