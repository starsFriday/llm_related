"""串起分词、前向传播、损失计算和生成的最小示例。"""

import argparse

import torch

from mini_transformer import BPETokenizer, LMConfig, Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="运行从零实现的 Transformer")
    parser.add_argument("--tokenizer", default="my_tokenizer.json")
    parser.add_argument("--text", default="Transformer是深度学习的核心架构")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = BPETokenizer(args.tokenizer)
    token_ids = tokenizer.encode(args.text)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    # 使用小配置便于在 CPU 上快速理解和调试；真正训练时再扩大模型。
    config = LMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=512,
        max_seq_len=128,
    )
    model = Transformer(config)
    loss, logits = model(input_ids, labels=input_ids)

    print(f"原文: {args.text}")
    print(f"Tokens: {tokenizer.tokenize(args.text)}")
    print(f"Token IDs: {token_ids}")
    print(f"模型参数量: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"logits 形状: {tuple(logits.shape)}")
    print(f"随机初始化模型的 loss: {loss.item():.4f}")

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        eos_token_id=tokenizer.eos_id,
    )
    print(f"生成结果（模型尚未训练，仅演示流程）: {tokenizer.decode(generated_ids[0].tolist())}")


if __name__ == "__main__":
    main()
