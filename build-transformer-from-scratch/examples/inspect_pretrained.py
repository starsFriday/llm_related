"""下载并查看 notebook 中的 T5-small 或 GPT-2 预训练模型。"""

import argparse
from pathlib import Path

from modelscope import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, snapshot_download


MODEL_SPECS = {
    "gpt2": ("AI-ModelScope/gpt2", AutoModelForCausalLM),
    "t5-small": ("AI-ModelScope/t5-small", AutoModelForSeq2SeqLM),
}


def download_model(name: str, model_dir: Path) -> Path:
    """下载 PyTorch/Transformers 所需文件，排除其他框架和 ONNX 权重。"""
    model_id, _ = MODEL_SPECS[name]
    downloaded = snapshot_download(
        model_id,
        local_dir=str(model_dir),
        ignore_file_pattern=["*.bin", "*.tflite", "*.h5", "*.ot", "*.msgpack", "onnx/*", "onnx/**"],
    )
    return Path(downloaded)


def print_config(name: str, model) -> None:
    config = model.config
    print(f"总参数量: {sum(parameter.numel() for parameter in model.parameters()):,}")
    if name == "gpt2":
        print(
            f"隐藏维度: {config.n_embd}\n层数: {config.n_layer}\n"
            f"注意力头数: {config.n_head}\n词表大小: {config.vocab_size}"
        )
    else:
        print(
            f"隐藏维度: {config.d_model}\nFFN 中间维度: {config.d_ff}\n"
            f"注意力头数: {config.num_heads}\nEncoder 层数: {config.num_layers}\n"
            f"Decoder 层数: {config.num_decoder_layers}\n词表大小: {config.vocab_size}\n"
            f"注意力头维度: {config.d_kv}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并查看一个预训练 Transformer")
    parser.add_argument("model", choices=MODEL_SPECS)
    parser.add_argument("--generate", action="store_true", help="GPT-2 额外执行一次文本生成")
    parser.add_argument("--prompt", default="What is artificial intelligence?")
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    if not (model_dir / "config.json").exists():
        print(f"正在下载 {args.model}...")
        model_dir = download_model(args.model, model_dir)

    _, model_class = MODEL_SPECS[args.model]
    model = model_class.from_pretrained(model_dir)
    print(model)
    print_config(args.model, model)

    if args.generate:
        if args.model != "gpt2":
            parser.error("--generate 示例目前只用于 GPT-2")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        inputs = tokenizer(args.prompt, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(f"\nGPT-2 生成结果:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
