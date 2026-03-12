"""
CommonGEN数据预处理 - 转换为CoT轨迹格式

原始格式:
{
  "source": "island_N#sea_N#town_N",
  "target": "old town and sea on the island"
}

目标格式（模拟CoT推理）:
{
  "question": "Generate a sentence using these concepts: island, sea, town",
  "is_expert": true,
  "steps": [
    {"s1": "Identify key concepts: island, sea, town", "a1": "island, sea, town"},
    {"s2": "Choose sentence structure: [noun phrase] [location phrase]", "a2": "structure"},
    {"s3": "Compose sentence: old town and sea on the island", "a3": "old town and sea on the island"}
  ]
}

策略：
1. 将概念识别作为第一步
2. 将句子规划作为第二步（可选）
3. 将最终生成作为最后一步
"""

import json
import random
from pathlib import Path
from tqdm import tqdm
import re

def parse_concepts(source_str):
    """解析概念词"""
    # 移除POS标签 (_N, _V等)
    concepts = [c.split('_')[0] for c in source_str.split('#')]
    return concepts

def create_cot_trajectory_v1(source, target, add_intermediate=True):
    """
    版本1: 简单两步
    - Step 1: 列出概念
    - Step 2: 生成句子
    """
    concepts = parse_concepts(source)
    concepts_str = ", ".join(concepts)
    
    question = f"Generate a coherent sentence using these concepts: {concepts_str}"
    
    steps = [
        {
            "s1": f"Step 1. List the concepts to include: {concepts_str}",
            "a1": concepts_str
        },
        {
            "s2": f"Step 2. Compose the sentence: {target}",
            "a2": target
        },
        {
            "s3": f"Final Answer: {target}",
            "a3": target
        }
    ]
    
    return {
        "question": question,
        "is_expert": True,
        "steps": steps
    }

def create_cot_trajectory_v2(source, target):
    """
    版本2: 三步（更像推理）
    - Step 1: 识别概念
    - Step 2: 规划句子结构
    - Step 3: 生成句子
    """
    concepts = parse_concepts(source)
    concepts_str = ", ".join(concepts)
    
    question = f"Create a natural sentence incorporating: {concepts_str}"
    
    # 启发式识别句子模式
    if ' and ' in target:
        pattern = "coordination structure"
    elif ' with ' in target:
        pattern = "prepositional phrase"
    elif ' in ' in target or ' on ' in target:
        pattern = "location description"
    else:
        pattern = "simple declarative"
    
    steps = [
        {
            "s1": f"Step 1. Identify required concepts: {concepts_str}",
            "a1": concepts_str
        },
        {
            "s2": f"Step 2. Plan sentence pattern: {pattern}",
            "a2": pattern
        },
        {
            "s3": f"Step 3. Generate sentence: {target}",
            "a3": target
        },
        {
            "s4": f"Final Answer: {target}",
            "a4": target
        }
    ]
    
    return {
        "question": question,
        "is_expert": True,
        "steps": steps
    }

def create_cot_trajectory_v3(source, target):
    """
    版本3: 更细粒度（模拟逐词生成）
    - Step 1: 识别概念
    - Step 2-N: 逐步构建句子
    - Step N+1: 完成
    
    适合用于步骤级判别器
    """
    concepts = parse_concepts(source)
    concepts_str = ", ".join(concepts)
    
    question = f"Build a sentence from: {concepts_str}"
    
    # 将句子分成几个片段
    words = target.split()
    mid_point = len(words) // 2
    
    part1 = " ".join(words[:mid_point])
    part2 = " ".join(words[mid_point:])
    
    steps = [
        {
            "s1": f"Step 1. Concepts to use: {concepts_str}",
            "a1": concepts_str
        },
        {
            "s2": f"Step 2. Start building: {part1}",
            "a2": part1
        },
        {
            "s3": f"Step 3. Continue: {part1} {part2}",
            "a3": target
        },
        {
            "s4": f"Final Answer: {target}",
            "a4": target
        }
    ]
    
    return {
        "question": question,
        "is_expert": True,
        "steps": steps
    }

def process_commongenv_dataset(
    input_file,
    output_file,
    version='v2',
    max_samples=None,
    min_concepts=3,
    max_concepts=6
):
    """
    处理CommonGEN数据集
    
    Args:
        input_file: 输入jsonl文件
        output_file: 输出json文件
        version: 'v1', 'v2', 'v3' 选择不同的CoT构造策略
        max_samples: 最多处理多少样本（None=全部）
        min_concepts: 最少概念数
        max_concepts: 最多概念数
    """
    
    version_map = {
        'v1': create_cot_trajectory_v1,
        'v2': create_cot_trajectory_v2,
        'v3': create_cot_trajectory_v3
    }
    
    create_trajectory_fn = version_map.get(version, create_cot_trajectory_v2)
    
    trajectories = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Processing {input_file.name}"):
            if max_samples and len(trajectories) >= max_samples:
                break
            
            data = json.loads(line)
            source = data['source']
            target = data['target']
            
            # 过滤：概念数量
            concepts = parse_concepts(source)
            if len(concepts) < min_concepts or len(concepts) > max_concepts:
                continue
            
            # 过滤：句子质量（简单启发式）
            if len(target.split()) < 4:  # 太短
                continue
            if target.count('.') > 2:  # 太多句子
                continue
            
            # 构造轨迹
            traj = create_trajectory_fn(source, target)
            trajectories.append(traj)
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(trajectories)} trajectories to {output_file}")
    
    # 打印样例
    print("\n📝 Sample trajectory:")
    print(json.dumps(trajectories[0], indent=2, ensure_ascii=False))
    
    return trajectories

def create_train_val_split(
    train_input,
    val_input,
    output_dir,
    train_size=5000,
    val_size=500,
    version='v2'
):
    """
    创建训练集和验证集
    """
    output_dir = Path(output_dir)
    
    print("=" * 80)
    print("🔄 Converting CommonGEN to CoT trajectory format")
    print("=" * 80)
    
    # 处理训练集
    print(f"\n[1/2] Processing training set...")
    train_trajs = process_commongenv_dataset(
        input_file=Path(train_input),
        output_file=output_dir / "commongenv_train_trajectories.json",
        version=version,
        max_samples=train_size
    )
    
    # 处理验证集
    print(f"\n[2/2] Processing validation set...")
    val_trajs = process_commongenv_dataset(
        input_file=Path(val_input),
        output_file=output_dir / "commongenv_val_trajectories.json",
        version=version,
        max_samples=val_size
    )
    
    print("\n" + "=" * 80)
    print("✅ Preprocessing complete!")
    print(f"Train: {len(train_trajs)} trajectories")
    print(f"Val:   {len(val_trajs)} trajectories")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # 统计信息
    print("\n📊 Statistics:")
    train_concepts = [len(parse_concepts(t['question'].split(': ')[1])) 
                     for t in train_trajs]
    print(f"  Avg concepts per sample: {sum(train_concepts)/len(train_concepts):.2f}")
    
    train_steps = [len(t['steps']) for t in train_trajs]
    print(f"  Avg steps per trajectory: {sum(train_steps)/len(train_steps):.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', type=str, 
                       default='/home/zhoukaining/TextGAIL/data/CommonGEN/train.jsonl')
    parser.add_argument('--val_input', type=str,
                       default='/home/zhoukaining/TextGAIL/data/CommonGEN/val.jsonl')
    parser.add_argument('--output_dir', type=str,
                       default='/home/zhoukaining/pro_cusor/GAIL_train/data_commongenv')
    parser.add_argument('--train_size', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=500,
                       help='Number of validation samples')
    parser.add_argument('--version', type=str, default='v2',
                       choices=['v1', 'v2', 'v3'],
                       help='CoT construction version')
    
    args = parser.parse_args()
    
    create_train_val_split(
        train_input=args.train_input,
        val_input=args.val_input,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        version=args.version
    )

