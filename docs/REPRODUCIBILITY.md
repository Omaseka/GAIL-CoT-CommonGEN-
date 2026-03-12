# 🔬 实验可重复性保证

## ❓ 你的担心是什么？

> "解冻了这么多参数，每次测试都会初始化吗？会不会每次实验参数都修改了导致模型性能发生变化？"

**这是一个非常好的问题！** 让我详细解释。

---

## ✅ 简短回答

**不会！** 每次运行脚本都从**相同的预训练权重**开始，不会互相影响。

### 核心机制

1. **每次运行都重新加载** → `AutoModelForCausalLM.from_pretrained()` 从磁盘读取原始权重
2. **解冻只改变标志** → `requires_grad=True` 不修改权重值，只允许训练时更新
3. **参数只在训练时变化** → 只有 `optimizer.step()` 才会修改权重
4. **脚本结束即重置** → 不保存到磁盘，下次运行重新开始

---

## 📊 详细解释

### 阶段1：加载模型（起点）

```python
# test_discriminator_commongenv.py 第160行
encoder = AutoModelForCausalLM.from_pretrained(
    model_path,  # 从磁盘加载
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
)
```

**发生了什么？**
- 从 `/home/zhoukaining/.cache/huggingface/models--Qwen2.5-7B-Instruct/` 读取权重文件
- 加载的是 **Qwen官方发布的预训练权重**
- **每次运行都是相同的起点**

### 阶段2：初始化判别器

```python
discriminator = HierarchicalDiscriminator(
    encoder=encoder,  # 使用刚加载的编码器
    tokenizer=tokenizer,
    device=device
)
```

**发生了什么？**
- 创建判别器头部（seq_head, step_head, prefix_head）
- 判别器头部使用 **随机初始化**
- 编码器权重 **保持不变**

### 阶段3：解冻操作

```python
discriminator.unfreeze_top_layers(num_layers=2)
```

**发生了什么？**
```python
# 只改变 requires_grad 标志
for param in model.layers[26].parameters():
    param.requires_grad = False  # 变成 True
```

- ✅ **权重值不变** - 仍然是预训练权重
- ✅ **只改变标志** - 允许梯度计算
- ✅ **不会修改数据** - 只是设置属性

### 阶段4：训练过程（**唯一会修改参数的阶段**）

```python
for epoch in range(epochs):
    loss.backward()        # 计算梯度
    optimizer.step()       # 🔥 修改参数！
```

**发生了什么？**
- ✅ **解冻的参数会被更新**
- ✅ **冻结的参数保持不变**
- ✅ **只在这个脚本内有效**

### 阶段5：脚本结束

```python
# 脚本结束，discriminator 对象被销毁
# 内存被释放
# 磁盘上的模型文件 **没有变化**
```

**发生了什么？**
- ❌ **不保存到磁盘**（除非明确调用 `torch.save()`）
- ❌ **内存中的修改丢失**
- ✅ **下次运行重新开始**

---

## 🧪 实验验证

### 验证1：对比两次加载的权重

运行这个脚本验证每次加载是否一致：

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code
python load_and_compare.py --mode verify
```

**输出示例：**
```
🔬 验证：每次加载是否从相同起点开始
[1/3] 第一次加载模型...
[2/3] 第二次加载模型...
[3/3] 对比权重...

✅ 验证通过！两次加载的权重完全相同
   说明：每次运行都从相同的预训练权重开始
```

### 验证2：运行多次实验，对比起始状态

```python
# 实验1
python test_discriminator_commongenv.py  # Gap=0.45

# 实验2（重新运行）
python test_discriminator_commongenv.py  # Gap=0.43（因为判别器头部随机初始化）

# 关键：两次实验的编码器起点完全相同！
```

**为什么Gap不完全一致？**
- 判别器头部（seq_head等）是 **随机初始化** 的
- 训练过程有 **随机性**（数据shuffle、dropout等）
- 但编码器的 **起点相同**

---

## ⚠️ 什么情况下会出问题？

### 情况1：同一个脚本内多次训练

```python
# ❌ 错误做法
encoder = load_model()  # 只加载一次

# 实验1
disc1 = HierarchicalDiscriminator(encoder, ...)
disc1.unfreeze_top_layers(2)
train(disc1, epochs=10)  # encoder参数被修改了！

# 实验2
disc2 = HierarchicalDiscriminator(encoder, ...)  # 😱 使用的是修改后的encoder
disc2.unfreeze_top_layers(2)
train(disc2, epochs=10)  # 起点已经不同了！
```

**解决方案：**
```python
# ✅ 正确做法
for exp in [exp1, exp2]:
    encoder = load_model()  # 每次重新加载
    disc = HierarchicalDiscriminator(encoder, ...)
    train(disc, ...)
```

### 情况2：保存checkpoint后继续训练

```python
# 训练并保存
discriminator = train_and_save()  # 保存了训练后的权重

# 下次加载
discriminator = load_from_checkpoint()  # 😱 加载的是训练后的，不是原始的
```

**解决方案：**
- 如果想继续训练：用 `load_from_checkpoint()`
- 如果想重新开始：用 `from_pretrained()`

### 情况3：修改了磁盘上的模型文件

```python
# 如果你手动修改了模型文件
# /home/zhoukaining/.cache/huggingface/models--Qwen2.5-7B-Instruct/

# 😱 以后所有实验的起点都变了！
```

**预防：**
- 不要修改预训练模型文件
- 训练后的权重保存到其他位置

---

## 💾 如何保存和加载checkpoint

### 保存checkpoint（已实现）

当前的 `test_discriminator_commongenv.py` 会自动保存：

```python
checkpoint = {
    'seq_head_state': discriminator.seq_head.state_dict(),
    'step_head_state': discriminator.step_head.state_dict(),
    'prefix_head_state': discriminator.prefix_head.state_dict(),
    'encoder_unfrozen_layers': {...},  # 解冻层的权重
    'config': {...},  # 训练配置
    'best_gap': 0.45,
}
torch.save(checkpoint, '../checkpoints/discriminator_commongenv.pt')
```

**保存内容：**
- ✅ 判别器头部权重
- ✅ 解冻的编码器层权重
- ✅ 训练配置和结果
- ❌ **不保存整个7B编码器**（太大，也没必要）

### 加载checkpoint

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code
python load_and_compare.py --mode load --checkpoint ../checkpoints/discriminator_commongenv.pt
```

**加载过程：**
1. 重新加载原始编码器（from_pretrained）
2. 创建新的判别器
3. 加载保存的头部权重
4. 解冻并加载解冻层权重
5. 可以继续训练或评估

---

## 🔍 最佳实践

### ✅ 推荐做法

#### 1. 独立运行每个实验

```bash
# 实验1：全冻结
python test_discriminator_commongenv.py > exp1.log

# 实验2：解冻2层（新进程，完全独立）
python test_discriminator_commongenv.py > exp2.log
```

**为什么好？**
- 每个实验完全独立
- 互不干扰
- 可并行运行

#### 2. 使用不同的checkpoint名称

```python
# 实验1
save_path = f"../checkpoints/disc_frozen_{timestamp}.pt"

# 实验2
save_path = f"../checkpoints/disc_unfrozen_2layers_{timestamp}.pt"
```

**为什么好？**
- 不会覆盖之前的结果
- 可以对比不同实验
- 可以回退到任何版本

#### 3. 记录随机种子

```python
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

**为什么好？**
- 判别器头部初始化相同
- 数据shuffle相同
- 结果可精确复现

#### 4. 保存完整实验信息

```python
experiment_info = {
    'timestamp': datetime.now(),
    'git_commit': get_git_commit(),
    'command': sys.argv,
    'config': config.__dict__,
    'results': results,
}
json.dump(experiment_info, open('experiment.json', 'w'))
```

### ❌ 应该避免的做法

#### 1. 重复使用同一个encoder对象

```python
# ❌ 不好
encoder = load_model()  # 全局变量
for i in range(10):
    disc = train(encoder)  # encoder被多次修改
```

#### 2. 不保存实验配置

```python
# ❌ 不好
torch.save(model.state_dict(), 'model.pt')  # 只保存权重，没有配置

# 几个月后...
# 😱 这是什么参数训练的？用了多少epoch？学习率多少？
```

#### 3. 覆盖checkpoint

```python
# ❌ 不好
torch.save(..., 'checkpoint.pt')  # 总是同一个名字

# 😱 之前的结果被覆盖了，无法对比！
```

---

## 📊 实验对比工具

### 对比多个checkpoint

```bash
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code

python load_and_compare.py --mode compare \
    --checkpoints \
    ../checkpoints/disc_frozen.pt \
    ../checkpoints/disc_unfrozen_2layers.pt \
    ../checkpoints/disc_unfrozen_4layers.pt
```

**输出示例：**
```
📊 Checkpoint对比
Checkpoint                     Gap        D(exp)     D(gen)     解冻       层数       
-------------------------------------------------------------------------------
disc_frozen.pt                 0.1900     0.60       0.41       ❌ 否      0          
disc_unfrozen_2layers.pt       0.4500     0.75       0.30       ✅ 是      2          
disc_unfrozen_4layers.pt       0.5200     0.80       0.28       ✅ 是      4          

🏆 最佳Checkpoint: disc_unfrozen_4layers.pt (Gap=0.5200)
```

---

## 🎓 总结

### 核心要点

| 问题 | 答案 |
|------|------|
| 每次运行都从相同起点开始吗？ | ✅ **是**（编码器从磁盘加载） |
| 解冻会修改参数吗？ | ❌ **不会**（只改变requires_grad标志） |
| 训练会修改参数吗？ | ✅ **会**（但只在内存中，不影响磁盘） |
| 下次运行会受影响吗？ | ❌ **不会**（重新加载原始权重） |
| checkpoint会影响后续实验吗？ | ❌ **不会**（除非明确加载） |

### 安全保证

1. **预训练权重不变**
   - 磁盘上的模型文件从不修改
   - 每次from_pretrained()加载相同权重

2. **实验完全独立**
   - 每个进程有自己的内存空间
   - 进程结束后所有修改消失

3. **checkpoint可选**
   - 保存训练结果供后续使用
   - 不影响新实验的起点

### 验证方法

```bash
# 运行这个验证每次加载是否一致
cd /home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/code
python load_and_compare.py --mode verify
```

---

## 📚 相关文件

- `test_discriminator_commongenv.py` - 主测试脚本（自动保存checkpoint）
- `load_and_compare.py` - 加载和对比工具
- `../checkpoints/` - checkpoint保存位置

---

**最后更新：** 2024-12-20  
**结论：** ✅ 当前实现是安全的，每次实验都从相同起点开始

