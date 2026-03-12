# 🎯 下一步行动计划

## 📊 当前状况总结

### ✅ 已完成
1. ✅ CommonGEN数据预处理（100条训练+20条验证）
2. ✅ 判别器测试脚本修复并运行成功
3. ✅ 发现核心问题：**判别器容量不足**

### 📉 实验结果

| 任务 | D(expert) | D(fake) | Gap | Accuracy | 结论 |
|------|-----------|---------|-----|----------|------|
| GSM8K | 0.58 | 0.42 | 0.16 | 72% | ❌ 判别器无法区分 |
| CommonGEN | 0.60 | 0.41 | **0.19** | 92% | ⚠️ 稍好但仍不够 |

**期望值:** Gap > 0.4, D(expert) > 0.75, D(fake) < 0.25

---

## 🔍 根本原因

### 判别器架构问题

```python
编码器（Qwen 7B）: 7,000,000,000 参数 → 全部冻结 ❌
判别器头部: 11,022,339 参数 → 容量太小 ❌
```

**问题：**
- 编码器冻结 → 无法学习任务特定的表示
- 头部太小 → 即使有好的表示也无法充分利用
- 就像**用小学生的理解力去评判博士论文**

**证据：**
- Accuracy达到92%（说明能分类）
- 但D值差距小（说明confidence不足）
- 学到的是表面特征，不是深层质量差异

---

## 🚀 解决方案（3个层次）

### 🔵 方案1：解冻编码器顶层（推荐⭐）

**原理：** 让编码器最后几层能适应判别任务

**修改：**
```python
# 在 discriminator.py 的 __init__ 中添加
def unfreeze_top_layers(self, num_layers=2):
    model = self.encoder.model
    total_layers = len(model.layers)
    
    # 解冻最后2层
    for i in range(total_layers - num_layers, total_layers):
        for param in model.layers[i].parameters():
            param.requires_grad = True
```

**预期效果：**
- 增加约400M可训练参数
- Gap应该能达到 0.3-0.4
- 显存增加约2-3GB

**运行：**
```bash
# 修改 test_discriminator_commongenv.py
# 在初始化判别器后添加：
# discriminator.unfreeze_top_layers(num_layers=2)

CUDA_VISIBLE_DEVICES=0 python test_discriminator_commongenv_v2.py
```

---

### 🟢 方案2：增大判别器头部

**原理：** 给判别器更强的分类能力

**修改：**
```python
# 在 discriminator.py 中
head_hidden_dim = 2048  # 从1024增加到2048

self.seq_head = nn.Sequential(
    nn.Linear(hidden_dim, 2048),
    nn.LayerNorm(2048),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(2048, 1024),  # 新增一层
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, 1)
)
```

**预期效果：**
- 增加约30M可训练参数
- Gap提升到 0.25-0.30
- 显存增加很少（<500MB）

**注意：** 需要重新训练（会丢弃已有权重）

---

### 🟡 方案3：使用LoRA微调编码器（最优但复杂）

**原理：** 低秩适配，以小参数量获得大提升

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # LoRA秩
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],  # 只微调注意力层
    lora_dropout=0.1,
)

self.encoder = get_peft_model(self.encoder, lora_config)
```

**预期效果：**
- 只增加约50M可训练参数
- Gap可能达到 0.4-0.5
- 显存增加约1-2GB

---

## 📋 立即执行（今晚/明天）

### Step 1: 测试方案1（解冻顶层）- 30分钟

<function_calls>
<invoke name="read_file">
<parameter name="target_file">/home/zhoukaining/pro_cusor/GAIL_train/test_discriminator_commongenv.py
