# Build Transformer from Scratch

本项目已将 `transformer.ipynb` 中可执行的代码按职责拆成 Python 模块。原 notebook
保留用于配合图片和公式阅读；`.py` 文件适合逐层调试、复用与后续训练。

## 目录结构

```text
mini_transformer/
├── config.py       # LMConfig：模型超参数及合法性检查
├── tokenizer.py    # ByteLevel BPE 的训练、编码与解码
├── layers.py       # Embedding、位置编码、注意力、FFN、RMSNorm、Block
└── model.py        # 完整 Decoder-only Transformer 与自回归生成
examples/
├── train_tokenizer.py     # 训练 notebook 中的示例分词器
├── demo_scratch_model.py  # 从分词到前向/损失/生成的最小闭环
└── inspect_pretrained.py  # 下载并查看 GPT-2 或 T5-small
main.py                    # 最简运行入口
```

notebook Step 4 中的数据集、预训练、SFT、RL 和部署单元只有说明或伪代码，没有可分离
的训练实现，因此本次没有虚构训练流程。

## 环境安装

```bash
conda create -n scratch python=3.10 -y
conda activate scratch
python -m pip install -r requirements.txt
```

若终端还未初始化 Conda，可执行：

```bash
source /root/autodl-tmp/tangfan/miniconda3/etc/profile.d/conda.sh
conda activate scratch
```

## 建议运行顺序

项目已经包含 `my_tokenizer.json`，所以可以直接运行完整模型演示：

```bash
conda run -n scratch python main.py
```

需要修改输入文本或生成长度时，可使用完整示例入口：

```bash
conda run -n scratch python -m examples.demo_scratch_model \
  --text "注意力机制是怎样工作的？" --max-new-tokens 20
```

如需从头训练示例分词器：

```bash
conda run -n scratch python -m examples.train_tokenizer
```

查看或生成预训练模型（首次执行会联网下载模型）：

```bash
conda run -n scratch python -m examples.inspect_pretrained t5-small
conda run -n scratch python -m examples.inspect_pretrained gpt2 --generate
```

## 阅读路线

建议依次阅读 `config.py` → `tokenizer.py` → `layers.py` → `model.py` →
`demo_scratch_model.py`。数据形状的主线为：

```text
input_ids (B, L)
  → token embedding + positional encoding (B, L, H)
  → N × TransformerBlock (B, L, H)
  → RMSNorm + LM Head (B, L, vocab_size)
  → next-token cross entropy loss
```
