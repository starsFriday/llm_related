"""
T5 模型逐层分析脚本
功能：输入文本，打印 Encoder 和 Decoder 每一层的输入输出形状
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==================== 配置 ====================
model_dir = '../models/t5-small'
input_text = "Do you know what is artificial intelligence?"


# ==================== 加载模型和分词器 ====================
print("=" * 60)
print("加载 T5 模型...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.eval()  # 设置为评估模式

print(f"\n模型配置:")
print(f"  - 词汇表大小: {model.config.vocab_size}")
print(f"  - 隐藏层维度: {model.config.d_model}")
print(f"  - Encoder 层数: {model.config.num_layers}")
print(f"  - Decoder 层数: {model.config.num_decoder_layers}")
print(f"  - 注意力头数: {model.config.num_heads}")
print(f"  - FFN 维度: {model.config.d_ff}")
print(f"  - KV 维度: {model.config.d_kv}")


# ==================== 分析模型的输入输出 ====================
print("\n" + "=" * 60)
print("开始分析输入输出")
print("=" * 60)

print(f"\n输入文本: {input_text}")

# 生成文本
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(
        inputs['input_ids'],
        max_length=50,
        num_beams=4,
        early_stopping=True
    )

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"\n生成结果:")
print(f"  {generated_text}")
print(f"\n生成的 Token 数量: {generated_ids.shape[1]}")


# ==================== 分词 ====================
print("\n" + "=" * 60)
print("分词处理（用于层级分析）...")
print("=" * 60)

# Encoder 输入
encoder_inputs = tokenizer(input_text, return_tensors="pt")
input_ids = encoder_inputs['input_ids']

# Decoder 输入 (T5 需要 decoder_input_ids)
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

print(f"\n输入文本: {input_text}")
print(f"Encoder Input IDs: {input_ids.tolist()}")
print(f"Encoder Token 数量: {input_ids.shape[1]}")
print(f"Encoder input_ids 形状: {input_ids.shape}  # (batch_size, encoder_seq_len)")
print(f"\nDecoder Input IDs: {decoder_input_ids.tolist()}")
print(f"Decoder input_ids 形状: {decoder_input_ids.shape}  # (batch_size, decoder_seq_len)")


# ==================== Hook 函数定义 ====================
layer_outputs = {}

def create_hook(name):
    """创建 hook 函数来捕获每一层的输出"""
    def hook(module, input, output):
        # 处理不同类型的输出
        if isinstance(output, tuple):
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

# ========== ENCODER ==========
print("\n注册 Encoder Hooks...")

# Encoder Embedding
hooks.append(model.encoder.embed_tokens.register_forward_hook(create_hook('Encoder_Embedding')))

# Encoder Blocks
for i in range(model.config.num_layers):
    # 整个 Encoder Block
    hooks.append(model.encoder.block[i].register_forward_hook(create_hook(f'Encoder_Block_{i}')))
    
    # Layer Norm (Self-Attention 之前)
    hooks.append(model.encoder.block[i].layer[0].layer_norm.register_forward_hook(
        create_hook(f'Encoder_Block_{i}_SelfAttn_LayerNorm')))
    
    # Self-Attention
    hooks.append(model.encoder.block[i].layer[0].SelfAttention.register_forward_hook(
        create_hook(f'Encoder_Block_{i}_SelfAttention')))
    
    # Layer Norm (FFN 之前)
    hooks.append(model.encoder.block[i].layer[1].layer_norm.register_forward_hook(
        create_hook(f'Encoder_Block_{i}_FFN_LayerNorm')))
    
    # Feed-Forward Network
    hooks.append(model.encoder.block[i].layer[1].DenseReluDense.register_forward_hook(
        create_hook(f'Encoder_Block_{i}_FFN')))

# Encoder Final Layer Norm
hooks.append(model.encoder.final_layer_norm.register_forward_hook(create_hook('Encoder_Final_LayerNorm')))

# ========== DECODER ==========
print("注册 Decoder Hooks...")

# Decoder Embedding
hooks.append(model.decoder.embed_tokens.register_forward_hook(create_hook('Decoder_Embedding')))

# Decoder Blocks
for i in range(model.config.num_decoder_layers):
    # 整个 Decoder Block
    hooks.append(model.decoder.block[i].register_forward_hook(create_hook(f'Decoder_Block_{i}')))
    
    # Layer Norm (Self-Attention 之前)
    hooks.append(model.decoder.block[i].layer[0].layer_norm.register_forward_hook(
        create_hook(f'Decoder_Block_{i}_SelfAttn_LayerNorm')))
    
    # Self-Attention
    hooks.append(model.decoder.block[i].layer[0].SelfAttention.register_forward_hook(
        create_hook(f'Decoder_Block_{i}_SelfAttention')))
    
    # Layer Norm (Cross-Attention 之前)
    hooks.append(model.decoder.block[i].layer[1].layer_norm.register_forward_hook(
        create_hook(f'Decoder_Block_{i}_CrossAttn_LayerNorm')))
    
    # Cross-Attention
    hooks.append(model.decoder.block[i].layer[1].EncDecAttention.register_forward_hook(
        create_hook(f'Decoder_Block_{i}_CrossAttention')))
    
    # Layer Norm (FFN 之前)
    hooks.append(model.decoder.block[i].layer[2].layer_norm.register_forward_hook(
        create_hook(f'Decoder_Block_{i}_FFN_LayerNorm')))
    
    # Feed-Forward Network
    hooks.append(model.decoder.block[i].layer[2].DenseReluDense.register_forward_hook(
        create_hook(f'Decoder_Block_{i}_FFN')))

# Decoder Final Layer Norm
hooks.append(model.decoder.final_layer_norm.register_forward_hook(create_hook('Decoder_Final_LayerNorm')))

# LM Head
hooks.append(model.lm_head.register_forward_hook(create_hook('LM_Head')))

# ==================== 前向传播 ====================
print("\n" + "=" * 60)
print("开始前向传播...")
print("=" * 60)

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids
    )
    logits = outputs.logits

print(f"\n最终输出 logits 形状: {logits.shape}  # (batch_size, decoder_seq_len, vocab_size)")

# ==================== 打印 ENCODER 层级信息 ====================
print("\n" + "=" * 60)
print("T5 ENCODER 层级分析 - 数据流")
print("=" * 60)

print(f"\n{'步骤':<5} {'层名称':<40} {'输入形状':<25} {'输出形状':<25}")
print("-" * 100)

step = 1

# Encoder Embedding
if 'Encoder_Embedding' in layer_outputs:
    info = layer_outputs['Encoder_Embedding']
    print(f"{step:<5} {'Encoder Embedding':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

print(f"\n{'--- Encoder Transformer Blocks ---':<40}")

# Encoder Blocks
for i in range(model.config.num_layers):
    print(f"\n  Encoder Block {i}:")
    
    # Self-Attention LayerNorm
    key = f'Encoder_Block_{i}_SelfAttn_LayerNorm'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Self-Attn LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Self-Attention
    key = f'Encoder_Block_{i}_SelfAttention'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Self-Attention':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # FFN LayerNorm
    key = f'Encoder_Block_{i}_FFN_LayerNorm'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ FFN LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # FFN
    key = f'Encoder_Block_{i}_FFN'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Feed-Forward Network':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Block Output
    key = f'Encoder_Block_{i}'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Block Output':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1

# Encoder Final LayerNorm
if 'Encoder_Final_LayerNorm' in layer_outputs:
    info = layer_outputs['Encoder_Final_LayerNorm']
    print(f"\n{step:<5} {'Encoder Final LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

# ==================== 打印 DECODER 层级信息 ====================
print("\n" + "=" * 60)
print("T5 DECODER 层级分析 - 数据流")
print("=" * 60)

print(f"\n{'步骤':<5} {'层名称':<40} {'输入形状':<25} {'输出形状':<25}")
print("-" * 100)

step = 1

# Decoder Embedding
if 'Decoder_Embedding' in layer_outputs:
    info = layer_outputs['Decoder_Embedding']
    print(f"{step:<5} {'Decoder Embedding':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

print(f"\n{'--- Decoder Transformer Blocks ---':<40}")

# Decoder Blocks
for i in range(model.config.num_decoder_layers):
    print(f"\n  Decoder Block {i}:")
    
    # Self-Attention LayerNorm
    key = f'Decoder_Block_{i}_SelfAttn_LayerNorm'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Self-Attn LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Self-Attention
    key = f'Decoder_Block_{i}_SelfAttention'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Self-Attention (Masked)':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Cross-Attention LayerNorm
    key = f'Decoder_Block_{i}_CrossAttn_LayerNorm'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Cross-Attn LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Cross-Attention
    key = f'Decoder_Block_{i}_CrossAttention'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Cross-Attention (Enc-Dec)':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # FFN LayerNorm
    key = f'Decoder_Block_{i}_FFN_LayerNorm'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ FFN LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # FFN
    key = f'Decoder_Block_{i}_FFN'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Feed-Forward Network':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1
    
    # Block Output
    key = f'Decoder_Block_{i}'
    if key in layer_outputs:
        info = layer_outputs[key]
        print(f"  {step:<5} {'  └─ Block Output':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
        step += 1

# Decoder Final LayerNorm
if 'Decoder_Final_LayerNorm' in layer_outputs:
    info = layer_outputs['Decoder_Final_LayerNorm']
    print(f"\n{step:<5} {'Decoder Final LayerNorm':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1

# LM Head
if 'LM_Head' in layer_outputs:
    info = layer_outputs['LM_Head']
    print(f"{step:<5} {'LM Head (输出层)':<40} {str(info['input_shape']):<25} {str(info['output_shape']):<25}")
    step += 1


# ==================== 流程图可视化 ====================
print("\n" + "=" * 60)
print("T5 数据流程图 (Encoder-Decoder 架构)")
print("=" * 60)

B = input_ids.shape[0]
L_enc = input_ids.shape[1]
L_dec = decoder_input_ids.shape[1]
H = model.config.d_model
V = model.config.vocab_size
N_enc = model.config.num_layers
N_dec = model.config.num_decoder_layers

print(f"""
{'='*30} ENCODER {'='*30}

输入: encoder_input_ids
形状: ({B}, {L_enc})
     │
     ▼
┌─────────────────────────┐
│   Token Embedding       │  →  ({B}, {L_enc}, {H})
└─────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Encoder Blocks × {N_enc} 层              │
│  ┌───────────────────────────────────┐  │
│  │  LayerNorm                        │  │
│  │  Self-Attention                   │  │
│  │  Residual Connection              │  │
│  │  LayerNorm                        │  │
│  │  Feed-Forward Network             │  │
│  │  Residual Connection              │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│   Final LayerNorm       │  →  ({B}, {L_enc}, {H})
└─────────────────────────┘
     │
     │ (Encoder Output = encoder_hidden_states)
     │
     
{'='*30} DECODER {'='*30}

输入: decoder_input_ids
形状: ({B}, {L_dec})
     │
     ▼
┌─────────────────────────┐
│   Token Embedding       │  →  ({B}, {L_dec}, {H})
└─────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Decoder Blocks × {N_dec} 层              │
│  ┌───────────────────────────────────┐  │
│  │  LayerNorm                        │  │
│  │  Masked Self-Attention            │  │
│  │  Residual Connection              │  │
│  │  LayerNorm                        │  │
│  │  Cross-Attention ←────────────────┼──┐ (从 Encoder)
│  │  Residual Connection              │  │
│  │  LayerNorm                        │  │
│  │  Feed-Forward Network             │  │
│  │  Residual Connection              │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│   Final LayerNorm       │  →  ({B}, {L_dec}, {H})
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│      LM Head            │  →  ({B}, {L_dec}, {V})
└─────────────────────────┘
     │
     ▼
输出: logits
形状: ({B}, {L_dec}, {V})

注意：
- Encoder 处理输入文本，生成上下文表示
- Decoder 通过 Cross-Attention 使用 Encoder 的输出
- Decoder 使用 Masked Self-Attention 保证自回归特性
""")

# ==================== 清理 Hooks ====================
for hook in hooks:
    hook.remove()

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
