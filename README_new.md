# CommonGEN-GAIL 项目

基于 **GAIL (Generative Adversarial Imitation Learning)** 的常识生成任务训练框架。  
使用 **Qwen2.5-7B-Instruct** 作为基座模型，通过层级判别器 + PPO 强化学习实现生成器与判别器的对抗训练。

## 📁 项目结构

```
CommonGEN/
├── code/                                    # 核心代码
│   ├── train_commongen.py                       # GAIL 主训练脚本（PPO + 判别器对抗）
│   ├── pretrain_discriminator.py                # 判别器预训练（构建硬负样本）
│   ├── warmup_generator.py                      # 生成器 SFT 热身（行为克隆）
│   ├── evaluate_warmup_effect.py                # 热身效果评估
│   ├── preprocess_commongenv2.py                # 原始数据 → CoT 轨迹格式
│   ├── data_gen/generate_cot_data.py            # 调用 LLM 批量生成 CoT 数据
│   ├── swanlog/                                 # SwanLab 可视化日志
│   └── logs/                                    # 训练运行日志
│
├── data/data_commongenv/
│   ├── commongenv_cot_train_llm.json            # 主训练集（2000 条 LLM 生成的 CoT）
│   ├── commongenv_train_trajectories.json       # 原始训练轨迹（规则生成）
│   └── commongenv_val_trajectories.json         # 原始验证轨迹
│
├── checkpoints/
│   ├── discriminator_pretrained_gap0.9866.pt    # 判别器预训练权重（训练使用）
│   ├── discriminator_pretrained_gap0.9963.pt    # 判别器预训练权重（高 Gap）
│   ├── discriminator_pretrained_gap0.9974.pt    # 判别器预训练权重（最高 Gap）
│   ├── sft_warmup/final_adapter/                # 生成器 SFT 热身 LoRA adapter
│   ├── run_commongen_20260109_153558/           # 第 1 次完整 GAIL 训练 checkpoint
│   └── run_commongen_20260128_182137/           # 最新 GAIL 训练 checkpoint（含 latest/best/interrupt）
│
├── results/
│   ├── run_commongen_20260109_153558/           # 第 1 次 trajectory comparison 结果
│   ├── run_commongen_20260128_182137/           # 最新 trajectory comparison 结果
│   └── warmup_effect_eval_*.json                # 生成器热身前后对比评估
│
├── logs/                                    # 判别器调优阶段的实验日志
│   ├── pretrain_discriminator_*.json            # 3 轮预训练的结果指标
│   └── *.log                                    # 各阶段调试日志
│
└── docs/                                    # 实验文档
    ├── COMMONGENV_EXPERIMENT_PLAN.md            # 完整实验计划
    ├── DISCRIMINATOR_CAPACITY_FIX.md            # 判别器容量修复方案
    ├── REPRODUCIBILITY.md                       # 可复现性说明
    └── WHY_UNFREEZING_FAILED.md                 # 解冻实验失败分析
```

## 🎯 项目目标

将 GAIL 框架应用于 **CommonGEN 常识生成任务**：给定一组概念词（如 broccoli, cheese, chicken, pizza），生成一句自然、流畅、包含所有概念的句子。

### 核心方法

- **生成器 (Generator)**：Qwen2.5-7B-Instruct + LoRA，通过 PPO 强化学习优化
- **判别器 (Discriminator)**：层级判别器（序列级 + 步骤级 + 前缀级），区分专家轨迹与生成轨迹
- **混合奖励**：`R = α·R_seq + β·R_step + γ·ΔR_prefix`，从 single 模式渐进到 composite 模式 (hybrid warmup)
- **动态平衡**：Gate 机制自动调控判别器训练频率，防止判别器过强导致生成器崩溃

## 📊 实验进展

### 阶段 1：数据准备 ✅

- 使用 Qwen-Max API 为 CommonGEN 训练集生成 **2000 条 CoT 专家轨迹**
- 每条轨迹包含：概念识别 → 句式规划 → 句子生成 → 最终答案
- 数据格式兼容层级判别器的 step span 计算

### 阶段 2：判别器预训练 ✅

通过构造硬负样本（打乱步骤顺序 / 替换概念 / 截断轨迹），对判别器头部进行预训练。

| 轮次 | Gap | D(expert) | D(generated) | 状态 |
|------|-----|-----------|--------------|------|
| 初始 | 0.19 | 0.60 | 0.41 | 容量不足 |
| 预训练后 | **0.9866** | ~1.0 | ~0.01 | 用于正式训练 |
| 最佳 | 0.9974 | ~1.0 | ~0.003 | 存档 |

### 阶段 3：生成器 SFT 热身 ✅

在 GAIL 对抗训练之前，先对生成器进行 1 个 epoch 的行为克隆（模仿专家数据的 CoT 格式），避免生成器因格式错误直接被判别器判死。

- LoRA rank=16, alpha=32
- SFT loss 稳定下降
- 热身后生成器能产出格式正确的 CoT 轨迹

### 阶段 4：GAIL 对抗训练 🔄 进行中

当前使用 `train_commongen.py` 进行正式的 GAIL 训练。

**训练配置：**
- 基座模型：Qwen2.5-7B-Instruct (4-bit 量化)
- 生成器：LoRA (rank=16) + PPO
- 判别器：预训练 head (Gap 0.9866) + Gate 自动调控
- 前 1000 步冻结判别器，之后自动解冻进入对抗阶段
- 双卡：Generator on cuda:1, Discriminator on cuda:0

**已完成的训练 run：**
- `run_commongen_20260109_153558`：第 1 次完整 GAIL（5 epoch），已保存 latest/best
- `run_commongen_20260128_182137`：最新一轮 GAIL，含 latest/best/interrupt checkpoint

### 阶段 5：评估与调优 📅 待完成

- [ ] 使用 CommonGEN 标准指标评估（ROUGE, BLEU, SPICE, CIDEr）
- [ ] 概念覆盖率 (Concept Coverage) 统计
- [ ] 消融实验：single vs composite vs hybrid 奖励模式
- [ ] 与基线模型对比（原始 Qwen / SFT-only / GAIL）
- [ ] 人工评估生成句子的质量与多样性

## 🚀 快速开始

### 1. 判别器预训练（若需从头开始）

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code
CUDA_VISIBLE_DEVICES=0 python pretrain_discriminator.py
```

### 2. 生成器 SFT 热身

```bash
CUDA_VISIBLE_DEVICES=0 python warmup_generator.py
```

### 3. 正式 GAIL 训练

```bash
# 首次训练（跳过 150 小时的 compare_trajectories）
CUDA_VISIBLE_DEVICES=0,1 python train_commongen.py --skip-initial-eval

# 断点续训（从上次 checkpoint 恢复）
CUDA_VISIBLE_DEVICES=0,1 python train_commongen.py \
  --resume-from ../checkpoints/run_commongen_20260128_182137 \
  --resume-tag latest \
  --skip-initial-eval
```

### 4. 查看可视化日志

```bash
# SwanLab 离线模式下，训练结束后手动同步
swanlab sync ./swanlog
```

## 📖 数据格式

### 训练数据 (commongenv_cot_train_llm.json)

```json
{
  "question": "Generate a sentence with: broccoli, cheese, chicken, pizza",
  "concepts": ["broccoli", "cheese", "chicken", "pizza"],
  "is_expert": true,
  "steps": [
    "Step 1. Identify required concepts: broccoli, cheese, chicken, pizza",
    "Step 2. Plan sentence pattern: A person is making a pizza with specific toppings.",
    "Step 3. Generate sentence: The chef is preparing a pizza topped with broccoli, cheese, and chicken.",
    "Final Answer: The chef is preparing a pizza topped with broccoli, cheese, and chicken."
  ],
  "gold_references": ["A grilled pizza with chicken, broccoli and cheese."]
}
```

## 🔧 核心脚本说明

| 脚本 | 功能 |
|------|------|
| `train_commongen.py` | GAIL 主训练：PPO 生成器 + 层级判别器对抗，支持 Gate 自动调控、断点续训、Ctrl+C 中断保存 |
| `pretrain_discriminator.py` | 判别器预训练：用硬负样本拉大 expert/fake 的 Gap |
| `warmup_generator.py` | 生成器 SFT 热身：行为克隆学习 CoT 格式 |
| `evaluate_warmup_effect.py` | 评估热身效果：对比热身前后生成质量 |
| `preprocess_commongenv2.py` | 数据预处理：CommonGEN 原始数据 → CoT 轨迹格式 |
| `data_gen/generate_cot_data.py` | 调用 LLM API 批量生成 CoT 专家数据 |

## 🎓 理论基础

### 为什么 CommonGEN 适合 GAIL？

| 维度 | CommonGEN | GSM8K | GAIL 需求 |
|------|-----------|-------|-----------|
| 答案唯一性 | 开放多样 | 唯一正确 | 需要分布对齐而非精确匹配 |
| 评估方式 | 主观质量 | 客观指标 | 判别器天然适合主观质量评估 |
| 判别器任务 | 流畅度 / 常识 | 计算正确性 | 区分特征更明显 |

### GAIL 训练的动态平衡

GAIL 训练的关键在于生成器与判别器的动态平衡：

1. **判别器太强** → 生成器收到全负奖励 → 梯度消失 / 模式坍塌
2. **判别器太弱** → 生成器无法获得有效信号 → 训练停滞
3. **理想状态** → Gap 在 0.3~0.7 之间波动，双方交替进步

代码中通过 Gate 机制实现自动调控：
- 判别器准确率 > 0.7 连续 3 次 → 自动跳过 3 个 batch 的判别器训练
- 前 1000 步强制冻结判别器 → 让生成器先适应 PPO 环境

## 🔗 依赖的上级模块

| 文件 | 位置 | 功能 |
|------|------|------|
| `discriminator.py` | `GAIL_train/` | 层级判别器 (HierarchicalDiscriminator) 实现 |
| `generator.py` | `GAIL_train/` | 生成器 (Generator) 实现，含 PPO 接口 |
| `compare_trajectories.py` | `GAIL_train/` | 专家 vs 生成轨迹对比工具 |
| `cas_utils.py` | `GAIL_train/` | CAS 奖励计算工具 |

---

**最后更新**：2026 年 2 月 10 日  
**当前状态**：GAIL 对抗训练进行中（阶段 4）
