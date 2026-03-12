# 🎯 CommonGEN实验方案 - GAIL+PRM融合

## 📌 为什么换到CommonGEN？

### GSM8K的根本问题

#### ❌ 不适合GAIL的原因

| 维度 | GSM8K | GAIL需求 | 契合度 |
|------|-------|---------|--------|
| 答案唯一性 | 唯一正确答案 | 多样性分布 | ❌ 不匹配 |
| 评估方式 | 客观指标（EM） | 分布相似度 | ❌ 不匹配 |
| 专家价值 | 提供计算路径 | 定义"好"的风格 | ⚠️ 部分匹配 |
| 判别器任务 | 区分正确/错误 | 区分专家/生成风格 | ❌ 任务错位 |

**核心矛盾：**
```
GAIL假设: P_θ → P_expert (分布对齐)
GSM8K需求: f(question) → correct_answer (目标对齐)
```

#### 实验证据

1. **判别器测试失败：**
   - 即使人工构造明显劣质轨迹，D(expert)=0.58, D(fake)=0.42
   - Gap只有0.16（期望>0.3）

2. **训练日志显示：**
   - D(expert) ≈ 0.45-0.58，D(gen) ≈ 0.37-0.57
   - 判别器无法区分专家和生成轨迹

3. **根本原因：**
   - 数学推理的"好坏"不在于风格，在于正确性
   - 判别器看不到最终答案是否正确，只能看文本分布
   - 导致学习到的是表面特征（步骤长度、格式等），而非本质

---

### ✅ CommonGEN完美契合

#### 任务特性

| 维度 | CommonGEN | GAIL需求 | 契合度 |
|------|-----------|---------|--------|
| 答案唯一性 | 开放生成 | 多样性分布 | ✅ 完美 |
| 评估方式 | 主观（流畅度） | 分布相似度 | ✅ 完美 |
| 专家价值 | 高质量范例 | 定义"好"的风格 | ✅ 完美 |
| 判别器任务 | 区分流畅/不流畅 | 区分专家/生成风格 | ✅ 完美 |

#### 示例对比

**输入：** `island_N#sea_N#town_N`

**专家样本：**
```
"old town and sea on the island"  ✅ 流畅自然
"beautiful blue sea in town in the island"  ✅ 描述性
```

**差的生成：**
```
"island sea town"  ❌ 缺少连接词
"the town the island the sea"  ❌ 重复
"sea sea island town"  ❌ 不连贯
```

**判别器可学习的特征：**
- ✅ 语法结构（是否有完整句式）
- ✅ 词汇搭配（形容词+名词）
- ✅ 连接词使用（and, in, on, with）
- ✅ 流畅度（是否读起来自然）

---

## 🎯 实验目标：GAIL + PRM 融合

### 核心创新点

#### 1. **GAIL的隐式奖励** - 学习"什么是好句子"

```python
R_GAIL = -log(1 - D(s, a))
```
- 判别器D学习专家的风格分布
- 不需要显式定义"流畅度"的规则
- 自动学习词汇搭配、语法结构

#### 2. **PRM的过程信号** - 监督中间步骤

```python
R_PRM(step_i) = V_PRM(s_0:i)  # 前缀价值函数
```

**在CommonGEN中的应用：**
- Step 1检查点：是否识别了所有概念？
- Step 2检查点：是否开始构建合理结构？
- Step 3检查点：是否完成完整句子？

#### 3. **混合奖励** - 全局+局部

```python
R_total = α * R_GAIL_seq + β * R_GAIL_step + γ * R_PRM_prefix
```

- `R_GAIL_seq`: 整体句子质量（流畅度、自然度）
- `R_GAIL_step`: 每步构建质量（局部合理性）
- `R_PRM_prefix`: 前缀势能差（朝正确方向前进）

---

## 🔧 实验步骤

### Phase 1: 数据准备（1小时）

#### Step 1.1: 运行预处理脚本

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train

# 基础版本（3步CoT）
python preprocess_commongenv2.py \
  --train_size 5000 \
  --val_size 500 \
  --version v2

# 输出位置
# data_commongenv/commongenv_train_trajectories.json
# data_commongenv/commongenv_val_trajectories.json
```

#### Step 1.2: 检查数据质量

```bash
# 查看样例
python -c "
import json
with open('data_commongenv/commongenv_train_trajectories.json') as f:
    data = json.load(f)
    print(json.dumps(data[0], indent=2))
"
```

**预期输出：**
```json
{
  "question": "Create a natural sentence incorporating: island, sea, town",
  "is_expert": true,
  "steps": [
    {"s1": "Step 1. Identify required concepts: island, sea, town", "a1": "island, sea, town"},
    {"s2": "Step 2. Plan sentence pattern: location description", "a2": "location description"},
    {"s3": "Step 3. Generate sentence: old town and sea on the island", "a3": "old town and sea on the island"},
    {"s4": "Final Answer: old town and sea on the island", "a4": "old town and sea on the island"}
  ]
}
```

---

### Phase 2: 判别器独立验证（2-3小时）

#### Step 2.1: 修改判别器测试脚本

需要修改 `test_discriminator_only.py`，因为CommonGEN的句子更短、步骤更少。

```bash
# 我会为你创建专门的版本
python test_discriminator_commongenv.py
```

**期望结果：**
- D(expert) > 0.75（比GSM8K应该更高）
- D(fake) < 0.25
- Gap > 0.4

**为什么应该更好？**
- CommonGEN的好/坏句子区别更明显
- 短句子判别器更容易学习
- 语法结构特征更清晰

---

### Phase 3: 修改训练代码（1-2小时）

#### Step 3.1: 修改配置

```python
# train.py - Config类
class Config:
    def __init__(self, fast_debug: bool = False):
        # 数据路径
        self.data_path = "/home/zhoukaining/pro_cusor/GAIL_train/data_commongenv/commongenv_train_trajectories.json"
        
        # 模型参数（CommonGEN用更短的序列）
        self.max_length = 256  # 从512降到256（句子更短）
        self.ppo_max_steps = 80  # 从180降到80
        
        # 奖励配置
        self.reward_mode = 'hybrid'  # 使用混合奖励
        self.alpha_reward = 0.4  # seq权重
        self.beta_reward = 0.4   # step权重
        self.gamma_reward = 0.2  # prefix权重
```

#### Step 3.2: 修改评估指标

在CommonGEN中，不能用"Final Answer正确率"，需要：
1. **自动指标：**
   - 生成长度分布
   - 概念覆盖率（是否用了所有概念词）
   - 重复度（是否重复生成）

2. **人工采样评估：**
   - 流畅度（1-5分）
   - 自然度（1-5分）
   - 创意性（1-5分）

---

### Phase 4: 训练实验（3-5天）

#### Experiment 1: Baseline（判别器预训练）

```bash
# 先让判别器学会区分专家和随机生成
python train.py \
  --mode pretrain_discriminator \
  --epochs 5 \
  --batch_size 16
```

**目标：** D(expert) > 0.7, D(gen) < 0.3

#### Experiment 2: GAIL-only（单一奖励）

```bash
# 只用序列级GAIL奖励
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --reward_mode single \
  --num_epochs 10 \
  --batch_size 8
```

**监控指标：**
- D(expert) vs D(gen) 的gap
- 生成句子的长度分布
- 概念覆盖率
- 判别器不能过强（防止模式崩溃）

#### Experiment 3: GAIL+Step（加入步骤监督）

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --reward_mode composite \
  --alpha_reward 0.5 \
  --beta_reward 0.5 \
  --num_epochs 10
```

**新增监控：**
- 每步的判别器得分
- 步骤间的连贯性

#### Experiment 4: Full (GAIL+PRM完整版)

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --reward_mode hybrid \
  --alpha_reward 0.4 \
  --beta_reward 0.4 \
  --gamma_reward 0.2 \
  --num_epochs 15
```

---

## 📊 评估方案

### 自动评估指标

#### 1. **概念覆盖率（Concept Coverage）**

```python
def compute_coverage(generated_text, concepts):
    """检查是否所有概念都被使用"""
    covered = sum(1 for c in concepts if c in generated_text.lower())
    return covered / len(concepts)
```

**目标：** > 95%

#### 2. **句子流畅度（Perplexity）**

使用预训练语言模型计算困惑度：
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compute_perplexity(text):
    # 用GPT-2计算困惑度
    # 越低越流畅
    pass
```

**目标：** 接近专家样本的困惑度

#### 3. **多样性（Diversity）**

```python
def compute_diversity(generated_samples):
    """计算生成的多样性（不同的n-gram比例）"""
    unique_bigrams = set()
    total_bigrams = 0
    for text in generated_samples:
        bigrams = list(nltk.bigrams(text.split()))
        unique_bigrams.update(bigrams)
        total_bigrams += len(bigrams)
    return len(unique_bigrams) / total_bigrams
```

**目标：** > 0.7（避免模式崩溃）

#### 4. **判别器得分分布**

监控训练过程中：
- D(expert) 稳定在 0.7-0.8
- D(gen) 逐渐上升，但不能超过 0.6（否则判别器失效）
- Gap保持在 0.2-0.3

---

### 人工评估（抽样100个）

每个epoch结束时，随机抽100个生成样本，人工打分：

| 维度 | 1分 | 3分 | 5分 |
|------|-----|-----|-----|
| 流畅度 | 不可读 | 基本通顺 | 完美自然 |
| 准确度 | 缺概念 | 全覆盖但生硬 | 全覆盖且自然 |
| 创意性 | 简单堆砌 | 常规表达 | 有趣新颖 |

**对比基线：**
- MLE baseline（直接监督学习）
- 专家样本（上限）
- 随机采样（下限）

---

## 🎯 成功标准

### Tier 1: 基本可行性（1周内达到）

- ✅ 判别器能work：D(expert)>0.7, D(gen)<0.4
- ✅ 训练能跑通：10个epoch不崩溃
- ✅ 概念覆盖率 > 90%
- ✅ 生成的句子基本可读

### Tier 2: 有效性验证（2周内达到）

- ✅ 优于MLE baseline（人工评估至少+0.5分）
- ✅ 判别器得分上升（D(gen)从0.3涨到0.5）
- ✅ 困惑度接近专家（±20%以内）
- ✅ 多样性保持 > 0.6

### Tier 3: 论文级别（3-4周）

- ✅ GAIL+PRM显著优于GAIL-only（p<0.05）
- ✅ 步骤级奖励有明确作用（消融实验）
- ✅ 在test set上的泛化性能
- ✅ Case study展示质量提升

---

## 🔄 与GSM8K的对比实验（可选）

如果CommonGEN成功，可以写一个对比章节：

### 实验设计

在两个任务上都跑GAIL：
1. **CommonGEN** - 开放生成任务
2. **GSM8K** - 目标驱动任务

### 预期结果

| 指标 | CommonGEN | GSM8K |
|------|-----------|-------|
| 判别器Gap | > 0.3 ✅ | < 0.2 ❌ |
| GAIL vs MLE | +15% ✅ | -5% ❌ |
| 训练稳定性 | 稳定 ✅ | 震荡 ⚠️ |

### 论文贡献点

**Main Contribution:**
> 我们发现GAIL框架在开放生成任务（如CommonGEN）上显著优于目标驱动任务（如GSM8K），并提出了任务适配性分析框架。

**Technical Contribution:**
> 提出了GAIL+PRM的混合奖励机制，结合隐式风格学习和显式过程监督，在CommonGEN上达到SOTA。

---

## 📝 接下来立即行动

### 今天（2小时）

1. ✅ **运行数据预处理：**
   ```bash
   cd /home/zhoukaining/pro_cusor/GAIL_train
   python preprocess_commongenv2.py --version v2
   ```

2. ✅ **检查数据质量：**
   - 查看生成的样例
   - 确认格式正确
   - 统计信息合理

3. ✅ **我创建CommonGEN专用的判别器测试脚本**

### 明天（半天）

1. 运行判别器独立测试（CommonGEN版本）
2. 观察D(expert)和D(fake)能否拉开
3. 如果成功，修改train.py配置

### 后天（1天）

1. 跑第一个完整实验（GAIL-only, single reward）
2. 监控训练曲线
3. 人工查看生成样例

---

## 💪 为什么这次会成功？

### 1. 任务匹配度 ✅
- CommonGEN是开放生成，天然适合GAIL
- 判别器有明确的特征可学（流畅度、语法）

### 2. 技术路线清晰 ✅
- 数据预处理简单（只需构造CoT格式）
- 评估指标明确（覆盖率、困惑度、人工打分）
- 逐步实验（先判别器，再GAIL，再混合）

### 3. 改进空间大 ✅
- CommonGEN的baseline还不够强
- GAIL在这个任务上鲜有人做
- PRM结合是创新点

### 4. 失败也有价值 ✅
- 如果CommonGEN也不行 → 说明判别器架构问题
- 如果CommonGEN成功但GSM8K不行 → 很好的对比实验
- 任何结果都能写论文

---

## 🎓 论文思路（如果成功）

### 标题备选

1. "When Does GAIL Work for Text? A Task-Centric Analysis"
2. "Hybrid GAIL-PRM: Combining Implicit Style Learning with Explicit Process Supervision"
3. "Beyond Goal-Driven Tasks: GAIL for Open-Ended Text Generation"

### Abstract草稿

> Generative Adversarial Imitation Learning (GAIL) has shown promise in learning expert policies without explicit reward functions. However, its application to text generation remains under-explored, particularly regarding **which tasks benefit most from distributional alignment**. 
>
> In this work, we conduct a systematic study comparing GAIL's effectiveness on **goal-driven tasks (GSM8K)** vs **open-ended tasks (CommonGEN)**. We find that:
> 1. GAIL struggles on goal-driven tasks where correctness matters more than style
> 2. GAIL excels on open-ended tasks where diverse high-quality outputs exist
> 3. Combining GAIL with Process Reward Models (PRM) bridges global style learning and local step supervision
>
> On CommonGEN, our GAIL+PRM approach achieves X% improvement over MLE baselines, with discriminator learning clearer distinctions (D-gap=0.4) compared to GSM8K (D-gap=0.15).

---

让我们开始吧！先跑数据预处理，我相信这次方向是对的！🚀

