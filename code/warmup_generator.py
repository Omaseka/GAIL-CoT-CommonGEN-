import torch
import swanlab
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers.utils.quantization_config import BitsAndBytesConfig
import bitsandbytes as bnb
import sys

# 复用 train_commongen.py 中的 Dataset 类
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from discriminator import ReasoningTrajectoryDataset

# 配置
class SFTConfig:
    def __init__(self):
        self.model_name = "/home/zhoukaining/.cache/huggingface/models--Qwen2.5-7B-Instruct"
        self.data_path = "/home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/data/data_commongenv/commongenv_cot_train_llm.json"
        self.save_dir = "../checkpoints/sft_warmup"
        self.max_length = 256
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.num_epochs = 1
        self.lr = 2e-5
        self.dtype = torch.bfloat16
        
        # SwanLab
        self.use_swanlab = True
        self.swanlab_project = "gail-commongen"
        self.swanlab_experiment_name = "sft-warmup"
        # 服务器侧 api.swanlab.cn 不可达时，使用离线模式，避免 init/login 阻塞训练
        self.swanlab_mode = "offline"

def train_sft():
    config = SFTConfig()
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 1. 初始化 SwanLab
    if config.use_swanlab:
        try:
            swanlab.init(
                project=config.swanlab_project,
                experiment_name=config.swanlab_experiment_name,
                config=vars(config),
                mode=config.swanlab_mode,
            )
            print(f"SwanLab initialized (mode={config.swanlab_mode})")
        except Exception as e:
            print(f"Warning: SwanLab init failed: {e}")
            config.use_swanlab = False

    # 2. 加载模型
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=config.dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, # SFT 可以稍微大一点参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 数据加载
    def collate_fn(batch):
        questions = [item['question'] for item in batch]
        # 在 SFT 中，我们需要构造完整的 prompt: "Question\nSteps: ...\nAnswer: ..."
        # 但这里 ReasoningTrajectoryDataset 已经处理了吗？
        # dataset[i] 返回 {'question': q, 'input_ids': ids, ...}
        # input_ids 已经是完整的序列了，我们直接用它做 causal lm training
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
        # Labels: padding 部分为 -100
        labels = padded_input_ids.clone()
        labels[padded_attention_masks == 0] = -100
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': labels
        }

    dataset = ReasoningTrajectoryDataset(config.data_path, tokenizer, config.max_length)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 4. 优化器
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=config.lr)
    
    # 5. 训练循环
    print("Starting SFT Warmup...")
    model.train()
    
    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.item()
                total_loss += current_loss
                pbar.set_postfix(loss=current_loss)
                if config.use_swanlab:
                    swanlab.log({"train/loss": current_loss})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss}")
        
        # 保存
        save_path = os.path.join(config.save_dir, f"sft_epoch_{epoch+1}.pt")
        model.save_pretrained(os.path.join(config.save_dir, "final_adapter"))
        print(f"Saved adapter to {config.save_dir}")

if __name__ == "__main__":
    train_sft()
