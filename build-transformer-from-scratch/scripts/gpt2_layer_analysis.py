"""
GPT2 模型层级分析脚本
功能：输入文本，打印每一层的输入输出形状
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 配置 ====================
model_dir = '../models/gpt2'
input_text = "Do you Know what is love"

# ==================== 加载模型和分词器 ====================
print("=" * 60)
print("加载 GPT2 模型...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()  # 设置为评估模式

print(f"\n模型配置:")
print(f"  - 词汇表大小: {model.config.vocab_size}")
print(f"  - 隐藏层维度: {model.config.n_embd}")
print(f"  - Transformer层数: {model.config.n_layer}")
print(f"  - 注意力头数: {model.config.n_head}")
print(f"  - 最大序列长度: {model.config.n_positions}")

# ==================== 模型生成测试 ====================
print("\n" + "=" * 60)
print("🚀 GPT2 模型生成测试")
print("=" * 60)

print(f"\n输入文本: {input_text}")

# 生成文本
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(
        inputs['input_ids'],
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"\n生成结果:")
print(f"  {generated_text}")
print(f"\n生成的 Token 数量: {generated_ids.shape[1]}")

# ==================== 分词 ====================
print("\n" + "=" * 60)
print("分词处理（用于层级分析）...")
print("=" * 60)

inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids']

print(f"\n输入文本: {input_text}")
print(f"Token IDs: {input_ids.tolist()}")
print(f"Token 数量: {input_ids.shape[1]}")
print(f"input_ids 形状: {input_ids.shape}  # (batch_size, sequence_length)")

# ==================== Hook 函数定义 ====================
layer_outputs = {}

def create_hook(name):
    """创建 hook 函数来捕获每一层的输出"""
    def hook(module, input, output):
        # 处理不同类型的输出
        if isinstance(output, tuple):
            # 如果输出是元组（如 Attention 层），取第一个元素
            actual_output = output[0]
        else:
            actual_output = output
        
        # 处理输入
        if isinstance(input, tuple) and len(input) > 0:
            actual_input = input[0]
        else:
            actual_input = input
            
        layer_outputs[name] = {
            'input_shape': actual_input.shape if torch.is_tensor(actual_input) else 'N/A',
            'output_shape': actual_output.shape if torch.is_tensor(actual_output) else 'N/A'
        }
    return hook

# ==================== 注册 Hooks ====================
hooks = []

# 1. Token Embedding
hooks.append(model.transformer.wte.register_forward_hook(create_hook('Token_Embedding')))

# 2. Position Embedding
hooks.append(model.transformer.wpe.register_forward_hook(create_hook('Position_Embedding')))

# 3. 每个 Transformer Block
for i in range(model.config.n_layer):
    # 整个 block
    hooks.append(model.transformer.h[i].register_forward_hook(create_hook(f'Block_{i}')))
    
    # Attention 层
    hooks.append(model.transformer.h[i].attn.register_forward_hook(create_hook(f'Block_{i}_Attention')))
    
    # Layer Norm 1 (Attention 之前)
    hooks.append(model.transformer.h[i].ln_1.register_forward_hook(create_hook(f'Block_{i}_LayerNorm1')))
    
    # MLP (Feed-Forward Network)
    hooks.append(model.transformer.h[i].mlp.register_forward_hook(create_hook(f'Block_{i}_MLP')))
    
    # Layer Norm 2 (MLP 之前)
    hooks.append(model.transformer.h[i].ln_2.register_forward_hook(create_hook(f'Block_{i}_LayerNorm2')))

# 4. Final Layer Norm
hooks.append(model.transformer.ln_f.register_forward_hook(create_hook('Final_LayerNorm')))

# 5. LM Head
hooks.append(model.lm_head.register_forward_hook(create_hook('LM_Head')))

# ==================== 前向传播 ====================
print("\n" + "=" * 60)
print("开始前向传播...")
print("=" * 60)

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print(f"\n最终输出 logits 形状: {logits.shape}  # (batch_size, sequence_length, vocab_size)")

# ==================== 打印层级信息 ====================
print("\n" + "=" * 60)
print("GPT2 模型层级分析 - 数据流")
print("=" * 60)

print(f"\n{'步骤':<5} {'层名称':<30} {'输入形状':<25} {'输出形状':<25}")
print("-" * 90)

# 按照流程图顺序打印
step = 1

# 1. Token Embedding
if 'Token_Embedding' in layer_outputs:
    info = layer_outputs['Token_Embedding']
    print(f"{step:<5} {'Token Embedding':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

# 2. Position Embedding
if 'Position_Embedding' in layer_outputs:
    info = layer_outputs['Position_Embedding']
    print(f"{step:<5} {'Position Embedding':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

print(f"\n{'--- Transformer Blocks ---':<30}")

# 3. Transformer Blocks
for i in range(model.config.n_layer):
    print(f"\n  Block {i}:")
    
    # LayerNorm 1
    key = f'Block_{i}_LayerNorm1'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ LayerNorm 1':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Attention
    key = f'Block_{i}_Attention'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Multi-Head Attention':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # LayerNorm 2
    key = f'Block_{i}_LayerNorm2'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ LayerNorm 2':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # MLP
    key = f'Block_{i}_MLP'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Feed-Forward (MLP)':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # 整个 Block 的输出
    key = f'Block_{i}'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Block Output':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1

print(f"\n{'--- 输出层 ---':<30}")

# 4. Final LayerNorm
if 'Final_LayerNorm' in layer_outputs:
    info = layer_outputs['Final_LayerNorm']
    print(f"{step:<5} {'Final LayerNorm':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

# 5. LM Head
if 'LM_Head' in layer_outputs:
    info = layer_outputs['LM_Head']
    print(f"{step:<5} {'LM Head (输出层)':<30} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

# ==================== 流程图可视化 ====================
print("\n" + "=" * 60)
print("GPT2 数据流程图")
print("=" * 60)

B, L = input_ids.shape
H = model.config.n_embd
V = model.config.vocab_size
N = model.config.n_layer

print(f"""
输入: input_ids
形状: ({B}, {L})
     │
     ▼
┌─────────────────────────┐
│   Token Embedding       │  →  ({B}, {L}, {H})
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│  Position Embedding     │  →  ({B}, {L}, {H})
└─────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Transformer Blocks × {N} 层              │
│  ┌───────────────────────────────────┐  │
│  │  LayerNorm 1                      │  │
│  │  Multi-Head Attention             │  │
│  │  Residual Connection              │  │
│  │  LayerNorm 2                      │  │
│  │  Feed-Forward Network (MLP)       │  │
│  │  Residual Connection              │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│   Final LayerNorm       │  →  ({B}, {L}, {H})
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│      LM Head            │  →  ({B}, {L}, {V})
└─────────────────────────┘
     │
     ▼
输出: logits
形状: ({B}, {L}, {V})
""")

# ==================== 清理 Hooks ====================
for hook in hooks:
    hook.remove()

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
