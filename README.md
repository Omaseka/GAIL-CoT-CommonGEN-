# CommonGEN-GAIL 项目

## 📁 项目结构

```
CommonGEN/
├── ANSWER.md                     # 参数冻结问题的回答文档
├── README.md                     # 项目说明
├── quick_start.sh                # 快速启动脚本
│
├── data/data_commongenv/
│   ├── commongenv_cot_train_llm.json        # 主训练数据 (2000条 CoT)
│   ├── commongenv_train_trajectories.json   # 原始训练轨迹
│   └── commongenv_val_trajectories.json     # 验证轨迹
│
├── code/                         # ✅ 核心代码（5个脚本）
│   ├── train_commongen.py            # GAIL 主训练脚本
│   ├── pretrain_discriminator.py     # 判别器预训练
│   ├── warmup_generator.py           # 生成器 SFT 热身
│   ├── evaluate_warmup_effect.py     # 热身效果评估
│   ├── preprocess_commongenv2.py     # 数据预处理
│   ├── data_gen/generate_cot_data.py # CoT 数据生成
│   ├── logs/gail_2026-01-28_1821.log # 完整训练日志 (453KB)
│   └── swanlog/
│       ├── run-20260109_153642 (38M) # 第1次完整训练可视化
│       └── run-20260210_161026 (33M) # 最新训练可视化
│
├── checkpoints/                  # ✅ 模型权重
│   ├── discriminator_pretrained_gap0.9866.pt  # 训练用判别器
│   ├── discriminator_pretrained_gap0.9963.pt  # 高 gap 判别器
│   ├── discriminator_pretrained_gap0.9974.pt  # 最高 gap 判别器
│   ├── sft_warmup/final_adapter/              # SFT 热身 adapter
│   ├── run_commongen_20260109_153558/         # 第1次完整 GAIL 训练
│   │   ├── generator_latest.pt / generator_best.pt
│   │   └── discriminator_latest.pt / discriminator_best.pt
│   └── run_commongen_20260128_182137/         # 最新 GAIL 训练
│       ├── generator_latest.pt / generator_best.pt / generator_interrupt.pt
│       └── discriminator_latest.pt / discriminator_best.pt
│
├── results/                      # ✅ 实验结果
│   ├── run_commongen_20260109_153558/trajectory_comparison_epoch_0.json
│   ├── run_commongen_20260128_182137/trajectory_comparison_epoch_0.json
│   ├── warmup_effect_eval_20260128_164611.json
│   └── warmup_effect_eval_20260128_173411.json
│
├── logs/                         # ✅ 判别器调优历史日志 (保留有完整结果的)
│   ├── pretrain_discriminator_*.json  x3  # 预训练结果
│   ├── disc_commongenv_test.log           # 初始判别器测试
│   ├── head_only_*.log/json               # Head-only 实验
│   ├── layerwise_*.log                    # 分层学习率实验
│   └── test_v2_fixed_*.log               # 容量修复测试
│
└── docs/                         # 实验文档
    ├── COMMONGENV_EXPERIMENT_PLAN.md
    ├── DISCRIMINATOR_CAPACITY_FIX.md
    ├── NEXT_STEPS.md
    ├── REPRODUCIBILITY.md
    └── WHY_UNFREEZING_FAILED.md
```

## 🎯 项目目标

将 **GAIL (Generative Adversarial Imitation Learning)** 与 **PRM (Process Reward Model)** 结合，应用于 **CommonGEN** 常识推理任务。

### 核心创新点

1. **任务适配性分析**：对比 CommonGEN（开放生成）vs GSM8K（目标驱动）
2. **混合奖励机制**：GAIL隐式奖励 + PRM过程监督
3. **多粒度判别**：序列级、步骤级、前缀级三重判别器

## 📊 当前进度

### ✅ 已完成

- [x] CommonGEN数据预处理（转换为CoT格式）
  - 训练集：100条 → 目标5000条
  - 验证集：20条 → 目标500条
- [x] 判别器独立测试脚本
- [x] 初步测试结果分析
- [x] 识别核心问题（判别器容量不足）

### 📉 实验结果（初步）

| 任务 | D(expert) | D(fake) | Gap | Accuracy |
|------|-----------|---------|-----|----------|
| GSM8K | 0.58 | 0.42 | 0.16 | 72% |
| CommonGEN | 0.60 | 0.41 | **0.19** | 92% |

**期望值**：Gap > 0.4, D(expert) > 0.75, D(fake) < 0.25

### 🔄 进行中

- [ ] 解冻编码器顶层（增加判别器容量）
- [ ] 重新测试判别器（期望Gap > 0.35）
- [ ] 预处理完整数据集（5000条）

### 📅 待完成

- [ ] 判别器预训练（10-20 epochs）
- [ ] 完整GAIL训练流程
- [ ] 消融实验（GAIL-only vs GAIL+PRM）
- [ ] 论文撰写

## 🚀 快速开始

### 1. 数据预处理

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code

# 小规模测试（100条）
python preprocess_commongenv2.py --train_size 100 --val_size 20 --version v2

# 完整数据（5000条）
python preprocess_commongenv2.py --train_size 5000 --val_size 500 --version v2
```

**输出位置**：`../data/data_commongenv/`

### 2. 判别器测试

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code

# 基础测试
CUDA_VISIBLE_DEVICES=0 python test_discriminator_commongenv.py 2>&1 | tee ../logs/test_$(date +%Y%m%d_%H%M%S).log
```

### 3. 修复判别器容量（推荐）

```bash
# 查看参数变化
python fix_discriminator_capacity.py
```

## 📖 文档说明

### docs/COMMONGENV_EXPERIMENT_PLAN.md

完整的实验计划，包括：
- 为什么CommonGEN更适合GAIL
- 与GSM8K的对比分析
- 4个阶段的详细实验步骤
- 评估指标和成功标准
- 论文思路

### docs/NEXT_STEPS.md

当前状态和下一步行动：
- 实验结果总结
- 根本原因分析
- 3个解决方案（解冻编码器/增大头部/LoRA）
- 立即执行计划

## 🔧 主要脚本说明

### preprocess_commongenv2.py

将CommonGEN原始数据转换为CoT轨迹格式。

**功能：**
- 读取CommonGEN的jsonl文件
- 构造3-4步的CoT推理过程
- 保存为判别器可用的格式

**三种版本：**
- `v1`: 简单两步（概念→生成）
- `v2`: 三步（概念→规划→生成）⭐ 推荐
- `v3`: 细粒度（概念→逐步构建）

### test_discriminator_commongenv.py

判别器独立训练和测试脚本。

**功能：**
- 加载CommonGEN专家数据
- 生成劣质假轨迹（4种策略）
- 训练判别器并评估性能
- 输出D(expert)、D(fake)、Gap指标

**假轨迹策略：**
1. 简单堆砌：`broccoli cheese chicken pizza.`
2. 重复词汇：`broccoli broccoli cheese pizza cheese.`
3. 缺少连接词：`broccoli the cheese the chicken.`
4. 语序混乱：`the pizza the broccoli the cheese.`

### fix_discriminator_capacity.py

修复判别器容量问题的工具脚本。

**两种方案：**
1. **解冻编码器顶层**（推荐）
   - 解冻最后2层Transformer
   - 增加约400M可训练参数
   - 预期Gap提升到0.35-0.45

2. **增大判别器头部**
   - 从1024维扩展到2048维
   - 增加约30M可训练参数
   - 预期Gap提升到0.25-0.30

## 📊 数据格式

### 输入（CommonGEN原始）

```json
{
  "source": "broccoli_N#cheese_N#chicken_N#pizza_N",
  "target": "A grilled pizza with chicken, broccoli and cheese."
}
```

### 输出（CoT轨迹）

```json
{
  "question": "Create a natural sentence incorporating: broccoli, cheese, chicken, pizza",
  "is_expert": true,
  "steps": [
    {
      "s1": "Step 1. Identify required concepts: broccoli, cheese, chicken, pizza",
      "a1": "broccoli, cheese, chicken, pizza"
    },
    {
      "s2": "Step 2. Plan sentence pattern: coordination structure",
      "a2": "coordination structure"
    },
    {
      "s3": "Step 3. Generate sentence: A grilled pizza with chicken, broccoli and cheese.",
      "a3": "A grilled pizza with chicken, broccoli and cheese."
    },
    {
      "s4": "Final Answer: A grilled pizza with chicken, broccoli and cheese.",
      "a4": "A grilled pizza with chicken, broccoli and cheese."
    }
  ]
}
```

## 🎓 理论基础

### 为什么CommonGEN适合GAIL？

| 维度 | CommonGEN | GSM8K | GAIL需求 |
|------|-----------|-------|---------|
| 答案唯一性 | 开放多样 | 唯一正确 | ✅ 需要多样性 |
| 评估方式 | 主观质量 | 客观指标 | ✅ 需要分布对齐 |
| 判别器任务 | 流畅度/语法 | 正确性 | ✅ 需要明显特征 |

### GAIL + PRM 混合奖励

```python
R_total = α * R_GAIL_seq + β * R_GAIL_step + γ * R_PRM_prefix

其中：
- R_GAIL_seq: 整体句子质量（全局）
- R_GAIL_step: 每步构建质量（局部）
- R_PRM_prefix: 前缀势能差（方向）
```

## 🐛 已知问题

### 1. 判别器容量不足

**现象：**
- D(expert)和D(fake)差距小（Gap < 0.2）
- 虽然准确率高（>90%）但confidence不足

**原因：**
- 7B编码器全冻结
- 判别器头部只有1100万参数

**解决方案：** 见 `code/fix_discriminator_capacity.py`

### 2. 数据规模小

**当前：** 100条训练数据
**问题：** 不足以训练判别器泛化

**解决方案：** 预处理完整数据（5000条）

## 📝 TODO

### 高优先级
- [ ] 测试解冻编码器方案
- [ ] 预处理完整CommonGEN数据
- [ ] 判别器预训练直到Gap > 0.35

### 中优先级
- [ ] 实现生成器采样（用于生成负样本）
- [ ] 编写完整GAIL训练循环
- [ ] 添加评估指标（概念覆盖率、困惑度）

### 低优先级
- [ ] 实验不同的混合奖励权重
- [ ] 消融实验（single/composite/hybrid）
- [ ] 与GSM8K对比实验
- [ ] 论文图表和case study

## 🔗 相关文件

### 主项目文件（上级目录）
- `../train.py` - 主训练脚本（需要修改以支持CommonGEN）
- `../discriminator.py` - 判别器实现
- `../generator.py` - 生成器实现

### GSM8K相关（对比实验）
- `../gail_api_trajectories.json` - GSM8K专家数据
- `../discriminator_test.log` - GSM8K判别器测试日志

## 📧 联系方式

如有问题，参考：
- `docs/COMMONGENV_EXPERIMENT_PLAN.md` - 完整实验指南
- `docs/NEXT_STEPS.md` - 当前问题和解决方案

---

**最后更新**：2024年12月19日  
**当前状态**：判别器容量优化中

