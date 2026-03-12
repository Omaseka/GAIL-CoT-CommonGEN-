#!/usr/bin/env python3
"""
判别器全量预训练脚本 - 只训练头部版本（针对量化模型）

用法：
    python pretrain_discriminator.py --data_path /path/to/data.json --epochs 10 --lr 5e-6
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from discriminator import HierarchicalDiscriminator, ReasoningTrajectoryDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from datetime import datetime
from tqdm import tqdm


def generate_fake_using_model(model, tokenizer, expert_dataset, max_length, num_fake=None):
    """
    使用模型本身生成假轨迹 (Hard Negatives)
    这样的假样本也是通顺的句子，迫使判别器学习区分逻辑质量，而不是简单的语法错误
    """
    if num_fake is None:
        num_fake = len(expert_dataset)
        
    fake_trajs = []
    print(f"Generating {num_fake} fake trajectories using the model (Hard Negatives)...")
    
    device = model.device
    model.eval() # 切换到评估模式
    
    # 批量生成以提高速度
    batch_size = 8 
    indices = list(range(len(expert_dataset)))
    
    # 如果需要生成的数量小于数据集大小，随机采样
    if num_fake < len(expert_dataset):
        indices = np.random.choice(indices, num_fake, replace=False)
    else:
        # 如果需要更多，则重复采样
        indices = np.random.choice(indices, num_fake, replace=True)
        
    for i in tqdm(range(0, num_fake, batch_size)):
        batch_indices = indices[i : i + batch_size]
        batch_prompts = []
        batch_concepts = []
        batch_questions = []
        
        for idx in batch_indices:
            expert = expert_dataset.trajectories[idx]
            concepts = expert.get('concepts', [])
            if not concepts:
                # 兼容旧格式逻辑...
                concepts_line = expert['steps'][0] if len(expert['steps']) > 0 else ""
                if "Identify required concepts:" in concepts_line:
                    concepts_str = concepts_line.split("Identify required concepts:")[-1].strip()
                    concepts = [c.strip() for c in concepts_str.split(',')]
                else:
                    concepts = expert['question'].split(':')[-1].split(',') if ':' in expert['question'] else []
                    concepts = [c.strip() for c in concepts]
            
            batch_concepts.append(concepts)
            batch_questions.append(expert['question'])
            
            # 关键修正：Prompt 必须与专家数据的 Question 格式保持一致
            # 专家数据的 question 是: "Generate a sentence with: c1, c2..."
            # 我们要让模型基于这个 question 生成，这样判别器就无法通过前缀来作弊
            
            # 注意：expert['question'] 已经是 "Generate a sentence with: ..." 格式
            prompt = f"{expert['question']}\n\nSteps:"
            batch_prompts.append(prompt)
            
        # Tokenize
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Process output
        # batch_decode 包含 prompt + completion
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, gen_text in enumerate(generated_texts):
            # gen_text 是 "Generate a sentence with: ... \n\nSteps: Step 1..."
            # 我们需要构造与 expert 相同的 full_text 结构
            
            # 尝试提取生成的 steps 部分
            if "Steps:" in gen_text:
                steps_part = gen_text.split("Steps:", 1)[1].strip()
            else:
                # Fallback: 如果模型没生成 Steps:，取 question 之后的内容
                steps_part = gen_text.replace(batch_questions[j], "").strip()
            
            # 简单切分 steps 用于显示/存储 (判别器主要看 full_text)
            steps_text = [l.strip() for l in steps_part.split('\n') if l.strip()]
            if not steps_text:
                steps_text = ["Step 1. (Generation failed)"]
            
            # 构造 full_text: Question + \n + Steps
            # 必须与 discriminator.py 中的 _process_trajectories 逻辑一致
            # discriminator 逻辑: full_text = question + "\n" + step1 + "\n" + step2...
            
            full_text = batch_questions[j]
            for step in steps_text:
                full_text += f"\n{step}"
            
            # 重新 tokenize full_text 以获得 input_ids
            full_encoding = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            fake_traj = {
                'question': batch_questions[j],
                'full_text': full_text,
                'input_ids': full_encoding['input_ids'].squeeze(0),
                'attention_mask': full_encoding['attention_mask'].squeeze(0),
                'steps': steps_text,
                'step_spans': [],
                'is_expert': 0.0
            }
            fake_trajs.append(fake_traj)
            
    return fake_trajs


def train_discriminator_head_only(
    discriminator, 
    expert_dataset, 
    fake_trajs, 
    epochs=10,
    batch_size=16, 
    lr=5e-6
):
    """
    只训练判别器头部（编码器保持冻结）
    """
    device = discriminator.device
    results = []
    
    print("\n" + "="*60)
    print("🔥 训练判别器头部（编码器保持冻结）")
    print("="*60)
    
    # 只优化头部参数
    head_params = [p for n, p in discriminator.named_parameters() if 'encoder' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=0.01)
    
    print(f"可训练参数: {sum(p.numel() for p in head_params):,}")
    print(f"训练epochs: {epochs}")
    print(f"学习率: {lr}")
    print(f"批大小: {batch_size}")
    
    best_gap = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        discriminator.train()
        indices = np.random.permutation(len(expert_dataset))
        num_batches = min(len(expert_dataset) // batch_size, 100) # 限制每个epoch最多100个batch，避免训练太久
        
        epoch_d_exp_sum = 0.0
        epoch_d_gen_sum = 0.0
        epoch_loss_sum = 0.0
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx in pbar:
            batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            expert_batch = [expert_dataset[i] for i in batch_indices]
            fake_batch = [fake_trajs[np.random.randint(len(fake_trajs))] for _ in range(batch_size)]
            
            # Collate
            def collate(batch):
                max_len = max([item['input_ids'].shape[0] for item in batch])
                input_ids, attention_mask = [], []
                for item in batch:
                    ids, mask = item['input_ids'], item['attention_mask']
                    pad_len = max_len - ids.shape[0]
                    if pad_len > 0:
                        ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
                        mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                    input_ids.append(ids)
                    attention_mask.append(mask)
                return {
                    'input_ids': torch.stack(input_ids),
                    'attention_mask': torch.stack(attention_mask),
                }
            
            expert_b = collate(expert_batch)
            fake_b = collate(fake_batch)
            
            expert_b = {k: v.to(device) for k, v in expert_b.items()}
            fake_b = {k: v.to(device) for k, v in fake_b.items()}
            
            optimizer.zero_grad()
            
            expert_logits, _, _ = discriminator.forward(expert_b['input_ids'], expert_b['attention_mask'])
            fake_logits, _, _ = discriminator.forward(fake_b['input_ids'], fake_b['attention_mask'])
            
            loss_exp = torch.nn.functional.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            )
            loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)
            )
            loss = loss_exp + loss_fake
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()
            
            with torch.no_grad():
                d_exp = torch.sigmoid(expert_logits).mean().item()
                d_fake = torch.sigmoid(fake_logits).mean().item()
                epoch_d_exp_sum += d_exp
                epoch_d_gen_sum += d_fake
                epoch_loss_sum += loss.item()
                
            pbar.set_postfix({'Gap': f"{d_exp - d_fake:.3f}", 'Loss': f"{loss.item():.3f}"})
        
        avg_d_exp = epoch_d_exp_sum / num_batches
        avg_d_gen = epoch_d_gen_sum / num_batches
        avg_loss = epoch_loss_sum / num_batches
        gap = avg_d_exp - avg_d_gen
        
        # 记录最佳结果
        if gap > best_gap:
            best_gap = gap
            best_epoch = epoch + 1
        
        result = {
            'epoch': epoch + 1,
            'D(expert)': avg_d_exp,
            'D(gen)': avg_d_gen,
            'Gap': gap,
            'Loss': avg_loss,
            'is_best': (gap == best_gap)
        }
        results.append(result)
        
        status = "🌟" if gap == best_gap else ""
        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | D(exp)={avg_d_exp:.4f} | D(gen)={avg_d_gen:.4f} | Gap={gap:.4f} {status}")
    
    print(f"\n🏆 最佳Gap: {best_gap:.4f} (Epoch {best_epoch})")
    
    return results, best_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/data/data_commongenv/commongenv_train_trajectories.json", help="Path to expert trajectories JSON")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    max_length = 512
    
    print("\n" + "="*60)
    print("🚀 判别器全量预训练")
    print("="*60)
    print(f"设备: {device}")
    print(f"数据路径: {args.data_path}")
    
    # 加载模型
    print("\n加载模型...")
    model_name = "/home/zhoukaining/.cache/huggingface/models--Qwen2.5-7B-Instruct"
    
    # 强制使用单卡，避免 accelerate 自动切分模型导致 device mismatch
    # device_map={'': 0} 会将所有层加载到 cuda:0
    target_device_map = {'': 0} 
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        encoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=target_device_map, # 强制指定设备映射
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Error loading model from local cache: {e}")
        print("Trying to load from HuggingFace...")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # ... (添加bnb_config等)
        encoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=target_device_map,
            trust_remote_code=True
        )

    # 加载数据
    print("\n加载数据...")
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        return
        
    expert_dataset = ReasoningTrajectoryDataset(args.data_path, tokenizer, max_length=max_length)
    print(f"专家轨迹数: {len(expert_dataset)}")
    
    # 生成假轨迹 (使用模型本身生成 Hard Negatives)
    print("\n生成假轨迹 (使用模型本身)...")
    fake_trajs = generate_fake_using_model(encoder, tokenizer, expert_dataset, max_length, num_fake=len(expert_dataset))
    print(f"假轨迹数: {len(fake_trajs)}")
    
    # 初始化判别器
    print("\n初始化判别器...")
    discriminator = HierarchicalDiscriminator(encoder, tokenizer, device=device, max_length=max_length)
    
    # 训练
    print("\n开始训练...")
    results, best_gap = train_discriminator_head_only(
        discriminator, 
        expert_dataset, 
        fake_trajs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # 保存结果
    output_dir = "../logs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/pretrain_discriminator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'best_gap': best_gap,
            'config': vars(args)
        }, f, indent=2)
    
    # 保存最佳checkpoint
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/discriminator_pretrained_gap{best_gap:.4f}.pt"
    torch.save({
        'discriminator_state_dict': discriminator.state_dict(),
        'best_gap': best_gap,
        'config': vars(args)
    }, checkpoint_path)
    
    print("\n" + "="*60)
    print("📊 训练完成！")
    print(f"Checkpoint已保存到: {checkpoint_path}")

if __name__ == "__main__":
    main()
