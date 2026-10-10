"""训练 notebook 中使用的示例 BPE 分词器。"""

import argparse

from mini_transformer import train_bpe_tokenizer


TRAIN_TEXTS = [
    "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
    "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的表示。",
    "Transformer是一种基于注意力机制的神经网络架构，广泛用于自然语言处理任务。",
    "大语言模型通过在大规模文本数据上进行预训练，学习了丰富的语言知识。",
    "预训练和微调是当前NLP的主流范式，先在通用数据上预训练，再在特定任务上微调。",
    "自然语言处理让计算机能够理解、生成和处理人类语言。",
    "神经网络由多个层组成，每层包含多个神经元，通过权重连接。",
    "注意力机制允许模型关注输入的不同部分，捕获长距离依赖关系。",
] * 100


def main() -> None:
    parser = argparse.ArgumentParser(description="训练示例 ByteLevel BPE 分词器")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--output", default="my_tokenizer.json")
    args = parser.parse_args()
    train_bpe_tokenizer(TRAIN_TEXTS, vocab_size=args.vocab_size, save_path=args.output)


if __name__ == "__main__":
    main()
