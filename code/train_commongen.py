import torch
import swanlab
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import re
import huggingface_hub
from huggingface_hub import snapshot_download
from transformers.utils import hub
import requests
import urllib3
import gc
# NOTE: peft 的 __init__ 导出在不同版本/类型检查器下可能不一致，这里使用显式导入路径
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.peft_types import TaskType
from peft.utils.other import prepare_model_for_kbit_training
from peft.peft_model import PeftModel
from transformers.utils.quantization_config import BitsAndBytesConfig  
import datetime
import bitsandbytes as bnb  # 8-bit optimizers & quantization helpers
import torch.nn as nn # Added for MSELoss
import torch.nn.functional as F  # Add F import
from scipy.stats import spearmanr
from collections import deque
from torch.cuda.amp import autocast
from enum import Enum # Import Enum
import sys

# 添加父目录到sys.path以导入上层模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from discriminator import ReasoningTrajectoryDataset, HierarchicalDiscriminator, evaluate_discriminator
from generator import Generator
from compare_trajectories import compare_trajectories
from cas_utils import get_cas_reward

# 设置环境变量
os.environ["SWANLAB_API_KEY"] = "8WWPMw4umduE55xvtx25z" # 设置SwanLab API Key
os.environ["TRANSFORMERS_CACHE"] = "/home/zhoukaining/.cache/huggingface/transformers"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 设置较长的超时时间
urllib3.Timeout.DEFAULT_TIMEOUT = 300

def render_traj(trajectory):
    """将轨迹渲染为HTML格式以便在SwanLab中显示，使用表格格式提高可读性"""
    html = "<div style='font-family: monospace; white-space: pre-wrap; max-width: 100%; overflow-x: auto;'>\n"
    html += f"<h3>Question:</h3>\n<p>{trajectory['question']}</p>\n\n"
    
    # 创建表格头部
    html += "<table border='1' style='border-collapse: collapse; width: 100%;'>\n"
    html += "<tr style='background-color: #f2f2f2;'><th style='padding: 8px; text-align: left;'>Step</th>"
    html += "<th style='padding: 8px; text-align: left;'>State</th>"
    html += "<th style='padding: 8px; text-align: left;'>Action</th></tr>\n"
    
    # 添加每个步骤作为表格行
    for step in trajectory['steps']:
        # 提取步骤编号和内容
        keys = list(step.keys())
        step_num = keys[0][1:] if keys else "?"  # 获取步骤编号，移除's'前缀
        state_key = f"s{step_num}"
        action_key = f"a{step_num}"
        
        state = step.get(state_key, "")
        action = step.get(action_key, "")
        
        # 分析状态的最后一行，获取最新添加的内容
        state_lines = state.split('\n')
        latest_state = state_lines[-1] if state_lines else ""
        if len(state_lines) > 1:
            latest_state = f"...{latest_state}"
        
        # 在表格中添加一行
        html += f"<tr><td style='padding: 8px; vertical-align: top;'>{step_num}</td>"
        html += f"<td style='padding: 8px; vertical-align: top;'>{latest_state}</td>"
        html += f"<td style='padding: 8px; vertical-align: top;'>{action}</td></tr>\n"
    
    html += "</table>\n"
    
    # 添加完整轨迹的详细视图
    html += "<h3>Full Trajectory:</h3>\n"
    html += "<pre style='background-color: #f8f8f8; padding: 10px; border-radius: 5px; max-height: 400px; overflow-y: auto;'>\n"
    html += f"Question: {trajectory['question']}\n\n"
    
    full_state = trajectory['question']
    for step in trajectory['steps']:
        keys = list(step.keys())
        step_num = keys[0][1:] if keys else "?"
        action_key = f"a{step_num}"
        
        # 从完整状态中提取最新添加的步骤
        state_lines = full_state.split('\n')
        latest_step = state_lines[-1] if len(state_lines) > 1 else ""
        
        action = step.get(action_key, "")
        html += f"Step {step_num}: {latest_step} = {action}\n"
        
        # 更新完整状态
        full_state = step.get(f"s{step_num}", "") + f"\nStep {step_num} = {action}"
    
    html += "</pre>\n"
    html += "</div>"
    
    return html

class Config:
    def __init__(self, fast_debug: bool = False):
        # 使用本地模型路径
        self.model_name = "/home/zhoukaining/.cache/huggingface/models--Qwen2.5-7B-Instruct"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = 256 # CommonGEN 较短，不需要512
        
        # 🔥 统一奖励配置
        self.reward_head = "step"  # 只用一个head：固定使用step
        self.reward_variant = "softplus"
        # 奖励模式：single | composite | hybrid
        self.reward_mode = "hybrid"
        self.hybrid_blend_max = 0.5
        self.hybrid_warmup_batches = 100
        
        # 数据集路径 - 指向 CommonGEN CoT 数据
        self.data_path = "/home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/data/data_commongenv/commongenv_cot_train_llm.json"
        # 预训练判别器路径 (如果不为空，将尝试加载)
        self.pretrain_discriminator_path = "../checkpoints/discriminator_pretrained_gap0.9866.pt"
        # SFT 热身权重路径 (如果存在，Generator 将从此加载)
        self.sft_adapter_path = "../checkpoints/sft_warmup/final_adapter"
        # 断点续训：从某个 run 目录加载 latest 权重（命令行传入时覆盖）
        self.resume_from = ""
        self.resume_tag = "latest"  # latest | best | interrupt
        
        # 默认使用完整数据；fast_debug 模式会覆盖此值
        self.debug_data_size = -1 # 使用全部数据
        
        # 显存优化
        self.batch_size = 4
        self.gradient_accumulation_steps = 4 
        self.num_epochs = 5
        self.start_epoch = 0  # 断点续训时可通过 --start-epoch 覆盖
        
        # 判别器训练频率与强度
        self.disc_train_freq = 2  # 每2个batch训练一次D
        self.disc_steps_per_batch = 1  # 每次只训练1步，防止过拟合
        self.max_disc_train_freq = 4  
        self.min_disc_train_freq = 1  
        self.disc_update_threshold = 0.70  # 提高阈值，如果D准确率 > 0.7，则跳过D训练
        
        self.reward_discount = 0.95  
        self.label_smoothing = 0.1   # CommonGEN任务较难，开启平滑
        self.entropy_weight = 0.05   
        self.entropy_final_weight = 0.01  
        self.consistency_weight = 0.4  
        self.baseline_weight = 0.7    
        self.lr_generator = 1e-5      
        self.lr_discriminator = 1e-6  # 🔽 极低的学习率，因为已经预训练过了且很容易过拟合
        
        self.num_trajectories = 100
        self.eval_interval = 1        
        self.compare_interval = 2     
        
        run_tag = datetime.datetime.now().strftime("run_commongen_%Y%m%d_%H%M%S")
        self.save_dir_root = "../checkpoints"
        self.results_dir_root = "../results"
        self.save_dir = os.path.join(self.save_dir_root, run_tag)
        self.results_dir = os.path.join(self.results_dir_root, run_tag)

        # fast debug override
        if fast_debug:
            self.num_epochs = 1           
            self.debug_data_size = 50    
            self.batch_size = 2           
            self.eval_interval = 1        
            self.disc_train_freq = 1      
            self.max_length = 128        
            self.ppo_max_steps = 64      
            self.min_rollout_samples = 16 
            print(f"[Fast Debug] Enabled: epochs=1, dataset size={self.debug_data_size}")
        else:
            self.min_rollout_samples = 64  # CommonGEN句子短，可以稍微少一点
            
        self.use_swanlab = True 
        self.swanlab_project = "gail-commongen"
        self.swanlab_experiment_name = "commongen-hard-neg-finetune"
        # 说明：local 模式需要额外安装 swanboard；离线训练推荐用 offline
        self.swanlab_mode = "offline"
        self.local_files_only = True
        self.dtype = torch.bfloat16
        self.sample_trajectories = True  
        self.num_samples = 3             
        self.use_focal_loss = True       

        # New hierarchical reward weights
        self.alpha_reward = 0.5 
        self.beta_reward = 0.3  
        self.gamma_reward = 0.2 
        self.eta_reward = 0.1    
        
        # New discriminator loss weights
        self.lambda_loss_step = 0.5 
        self.mu_loss_prefix = 0.3   

        # Training stability
        self.reward_clip_value = 5.0  
        self.reward_norm_decay = 0.99  
        self.disc_update_ratio = 2 
        self.value_loss_weight = 0.25  
        self.max_grad_norm = 1.0  
        
        if not fast_debug:
            self.ppo_max_steps = 128 # CommonGEN不需要很长
        self.ppo_clip_value = 0.2  
        self.ppo_temperature = 1.0  
        self.eval_rms_bootstrap_trajectories = 64  
        self.entropy_topk = 4096  
        
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.fast_debug = fast_debug

        self.DiscState = Enum('DiscState', 'ACTIVE SKIP FROZEN')
        self.disc_state = self.DiscState.FROZEN # 初始冻结判别器
        self.disc_skip_batches_left = 0
        self.disc_frozen_steps_left = 1000 # 冻结前 1000 个 batch (约 1-2 个 epoch)
        self.d_too_strong_counter = 0
        self.dg_too_low_counter = 0
        
        os.makedirs(self.results_dir, exist_ok=True)
        
class RunningMeanStd:
    """统一的奖励标准化类"""
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        
    def update(self, x):
        if torch.is_tensor(x):
            x = x.item()
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.std = ((self.count - 1) * self.std**2 + delta * delta2) / self.count
        self.std = max(self.std**0.5, 1e-6)
        
    def normalize(self, x, update: bool = True):
        if torch.is_tensor(x):
            val = x.item()
        else:
            val = x
            
        if update:
            self.update(val)
            
        normalized = (val - self.mean) / self.std
        return max(-5.0, min(5.0, normalized))

class EmaRunningMeanStd:
    """基于EMA的奖励标准化"""
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update_stats(self, x: float):
        x = max(-5.0, min(5.0, float(x)))
        if self.count == 0:
            self.mean = x
            self.var = 1.0
        else:
            beta = self.decay
            self.mean = beta * self.mean + (1 - beta) * x
            m2 = beta * (self.var + self.mean * self.mean) + (1 - beta) * (x * x)
            self.var = max(1e-6, m2 - self.mean * self.mean)
        self.count += 1

    def normalize(self, x: float, update: bool = True) -> float:
        x = max(-5.0, min(5.0, float(x)))
        if update:
            self.update_stats(x)
        std = max(1e-6, self.var ** 0.5)
        z = (x - self.mean) / std
        if z > 5.0: return 5.0
        if z < -5.0: return -5.0
        return z

class GAILTrainer:
    def __init__(self, config):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self.config = config
        self.reward_normalizer = EmaRunningMeanStd(decay=getattr(config, 'reward_norm_decay', 0.99))
        self.reward_normalizer_eval = EmaRunningMeanStd(decay=getattr(config, 'reward_norm_decay', 0.99))
        self.use_eval_rms = False
        
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            self.generator_device_str = "cuda:1"
            self.discriminator_device_str = "cuda:0"
        else:
            self.generator_device_str = "cuda:0"
            self.discriminator_device_str = "cuda:0"
        self.generator_device = torch.device(self.generator_device_str)
        self.discriminator_device = torch.device(self.discriminator_device_str)
        self.device = self.generator_device
        self.dtype = config.dtype
        
        self._train_step = 0
        self.skip_disc_update = False
        self.reward_buffer = []
        self.entropy_warmup_steps = 50
        self.current_batch = 0
        self.total_batches_est = 1000
        self.recent_dgen_means = deque(maxlen=20)
        
        self.DiscState = Enum('DiscState', 'ACTIVE SKIP FROZEN')
        self.disc_state = self.DiscState.FROZEN # 初始冻结判别器
        self.disc_skip_batches_left = 0
        self.disc_frozen_steps_left = 1000 # 冻结前 1000 个 batch (约 1-2 个 epoch)
        self.d_too_strong_counter = 0
        self.dg_too_low_counter = 0
        
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        print(f"Starting initialization with device={self.device}, dtype={self.dtype}")
        
        # Load Tokenizer
        print("Loading tokenizer...")
        model_path = config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=config.local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        print("Successfully loaded tokenizer")
        
        # 1. Generator Model
        print(f"Initializing Generator model… (pinned on {self.generator_device_str})")
        torch.cuda.set_device(self.generator_device)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
        )
        generator_base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map=self.generator_device_str,
            local_files_only=config.local_files_only
        )
        generator_base_model.config.use_cache = False
        generator_base_model = prepare_model_for_kbit_training(
            generator_base_model, 
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # ✅ 正确的 adapter 加载方式：
        # - 若存在 SFT warmup 或 resume 的 adapter 目录，使用 PeftModel.from_pretrained 自动匹配 LoRA rank/结构
        # - 否则创建一个默认 LoRA（rank 可根据任务调整）
        adapter_path = getattr(config, "sft_adapter_path", None)
        if adapter_path and os.path.exists(adapter_path):
            print(f"🔄 Loading PEFT adapter from: {adapter_path}")
            peft_model = PeftModel.from_pretrained(generator_base_model, adapter_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=["lm_head"]
            )
            peft_model = get_peft_model(generator_base_model, lora_config)
        try: peft_model.base_model.model.config.use_cache = False
        except: pass
        self.generator = Generator(peft_model, self.tokenizer, device=self.generator_device_str)
        
        # 2. Discriminator Model
        print(f"Initializing Discriminator model (pinned on {self.discriminator_device_str})…")
        torch.cuda.set_device(self.discriminator_device)
        discriminator_quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
        )
        discriminator_encoder = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=discriminator_quantization_config,
            device_map=self.discriminator_device_str,
            local_files_only=config.local_files_only,
            cache_dir=config.local_files_only and None or f"{os.environ.get('TRANSFORMERS_CACHE', '')}_discriminator"
        )
        discriminator_encoder.config.use_cache = False
        for param in discriminator_encoder.parameters():
            param.requires_grad = False
            
        self.discriminator = HierarchicalDiscriminator(
            discriminator_encoder,
            self.tokenizer,
            device=self.discriminator_device_str,
            max_length=self.config.max_length,
        )
        
        # Load Pretrained Discriminator Head
        if hasattr(config, 'pretrain_discriminator_path') and config.pretrain_discriminator_path and os.path.exists(config.pretrain_discriminator_path):
            print(f"🔄 Loading pretrained discriminator from: {config.pretrain_discriminator_path}")
            try:
                # 只加载 head 的权重
                state_dict = torch.load(config.pretrain_discriminator_path, map_location=self.discriminator_device)
                # 过滤掉 encoder 的 keys (如果有)
                head_state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
                self.discriminator.load_state_dict(head_state_dict, strict=False)
                print("✅ Successfully loaded pretrained discriminator head!")
            except Exception as e:
                print(f"❌ Failed to load pretrained discriminator: {e}")
        else:
            print("⚠️ No pretrained discriminator found, starting from scratch.")

        # ✅ 断点续训：如存在 resume_from，则加载对应的 discriminator_{tag}.pt
        resume_from = getattr(config, "resume_from", "") or ""
        resume_tag = getattr(config, "resume_tag", "latest") or "latest"
        if resume_from and os.path.isdir(resume_from):
            disc_resume_path = os.path.join(resume_from, f"discriminator_{resume_tag}.pt")
            if os.path.exists(disc_resume_path):
                try:
                    self.discriminator.load(disc_resume_path)
                except Exception as e:
                    print(f"⚠️ Failed to load resume discriminator from {disc_resume_path}: {e}")

        self.generator.to(self.generator_device)
        self.discriminator.to(self.discriminator_device)
        
        self.generator_optimizer = bnb.optim.PagedAdamW8bit(
            self.generator.parameters(), lr=config.lr_generator, weight_decay=0.01
        )
        self.discriminator_optimizer = bnb.optim.AdamW8bit(
            self.discriminator.get_trainable_parameters(), lr=config.lr_discriminator, weight_decay=0.01
        )
        
        self.generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.generator_optimizer, mode='max', factor=0.5, patience=2
        )
        self.discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.discriminator_optimizer, mode='min', factor=0.5, patience=2
        )
        
        os.makedirs(config.save_dir, exist_ok=True)
        
        if config.use_swanlab:
            try:
                swanlab.init(
                    project=config.swanlab_project,
                    experiment_name=config.swanlab_experiment_name,
                    config=dict(vars(config)),
                    mode=getattr(config, 'swanlab_mode', 'cloud')
                )
                print("Successfully initialized SwanLab.")
            except Exception as e:
                print(f"Warning: Failed to initialize SwanLab: {str(e)}")
                config.use_swanlab = False
            
        print("Initialization complete")
    
    def pick_head_outputs(self, discriminator_outputs, head=None):
        if head is None: head = self.config.reward_head
        seq_logits, step_logits, prefix_logits = discriminator_outputs
        if head == "seq": return seq_logits
        elif head == "step": return step_logits
        elif head == "prefix": return prefix_logits
        else: raise ValueError(f"Unknown head: {head}")

    def compute_step_spans(self, question, steps_text, max_length=None):
        if steps_text is None: steps_text = []
        if max_length is None: max_length = self.config.max_length
        spans = []
        try:
            question_len = len(self.tokenizer.encode(question, add_special_tokens=False))
            current_len = question_len
            for step in steps_text:
                step_len = len(self.tokenizer.encode(f"\n{step}", add_special_tokens=False))
                start = min(current_len, max_length - 1)
                end = min(current_len + step_len, max_length - 1)
                if start < end: spans.append((start, end))
                current_len += step_len
        except Exception as e:
            print(f"[SpanWarn] failed to compute step spans: {e}")
        return spans
    
    def gail_reward_from_logits(self, logits, variant=None):
        if variant is None: variant = self.config.reward_variant
        if variant == "softplus": return F.softplus(logits)
        elif variant == "logD": return F.logsigmoid(logits)
        else: raise ValueError(f"Unknown reward variant: {variant}")
    
    def get_unified_reward(self, trajectory_text, prefix_texts=None, is_training=True):
        traj_encoding = self.discriminator.tokenizer(
            trajectory_text, max_length=self.config.max_length, truncation=True, return_tensors='pt'
        )
        traj_encoding = {k: v.to(self.discriminator_device) for k, v in traj_encoding.items()}
        
        with torch.no_grad():
            discriminator_outputs = self.discriminator(traj_encoding['input_ids'], traj_encoding['attention_mask'])
        
        logits_single = self.pick_head_outputs(discriminator_outputs)
        raw_single = self.gail_reward_from_logits(logits_single).mean()

        seq_logits, step_logits, prefix_logits = discriminator_outputs
        R_seq = self.gail_reward_from_logits(seq_logits).mean()
        r_step = self.gail_reward_from_logits(step_logits).mean()
        delta_D_prefix = (self.gail_reward_from_logits(prefix_logits).mean() - R_seq)
        raw_composite = (
            self.config.alpha_reward * R_seq +
            self.config.beta_reward * r_step +
            self.config.gamma_reward * delta_D_prefix
        )

        if self.config.reward_mode == "hybrid":
            blend = min(self.config.hybrid_blend_max, max(0.0, (self.current_batch) / max(1, self.config.hybrid_warmup_batches)))
            raw_reward = (1 - blend) * raw_single + blend * raw_composite
        elif self.config.reward_mode == "composite":
            raw_reward = raw_composite
        else:
            raw_reward = raw_single

        raw_val = float(raw_reward.detach().cpu().item())
        if is_training:
            norm_val = self.reward_normalizer.normalize(raw_val, update=True)
        else:
            if self.use_eval_rms:
                norm_val = self.reward_normalizer_eval.normalize(raw_val, update=False)
            else:
                norm_val = self.reward_normalizer.normalize(raw_val, update=False)
        
        del traj_encoding, discriminator_outputs
        torch.cuda.empty_cache()
        return norm_val, raw_val
    
    def _create_collate_fn(self):
        def collate_fn(batch):
            questions = [item['question'] for item in batch]
            input_ids = [item['input_ids'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            step_spans = [item['step_spans'] for item in batch]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
            return {
                'question': questions,
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_masks,
                'step_spans': step_spans
            }
        return collate_fn
    
    def ppo_step(self, trajectories, log_probs, rewards, old_log_probs, value_preds):
        with autocast(dtype=torch.bfloat16):
            rewards_normalized = torch.tanh(rewards.float() / 3.0)
            values_squeezed = value_preds.squeeze(-1).float()
            advantages = rewards_normalized - values_squeezed
            
            if advantages.numel() > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                advantages = torch.clamp(advantages, -10.0, 10.0)
            else:
                advantages = torch.zeros_like(advantages)
            
            log_ratio = (log_probs.float() - old_log_probs.float()).clamp(-20, 20)
            ratio = torch.exp(log_ratio)
            clip_ratio = self.config.ppo_clip_value
            
            policy_loss_1 = ratio * advantages
            policy_loss_2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            min_surrogate = torch.min(policy_loss_1, policy_loss_2)
            policy_loss = -min_surrogate.mean()
            
            approx_kl = (old_log_probs.float() - log_probs.float()).mean().detach()
            clip_frac = (torch.abs(ratio - 1.0) > clip_ratio).float().mean().detach()
            ratio_mean = ratio.mean().detach()
            ratio_std = ratio.std().detach()
            
            v_pred = values_squeezed
            old_values = value_preds.squeeze(-1).float().detach()
            v_clipped = old_values + (v_pred - old_values).clamp(-0.2, 0.2)

            vf_loss1 = F.mse_loss(v_pred, rewards_normalized)
            vf_loss2 = F.mse_loss(v_clipped, rewards_normalized)
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
            
            policy_valid = torch.isfinite(policy_loss) and torch.abs(policy_loss) < 100.0
            value_valid = torch.isfinite(value_loss) and torch.abs(value_loss) < 100.0

            if not (policy_valid and value_valid):
                print(f"⚠️ PPO loss overflow or invalid!")
                return None, None

        # NOTE: 这些统计目前未被下游使用；避免给 Tensor 动态挂属性以通过静态检查
        return policy_loss, value_loss
    
    def log_metrics(self, metrics):
        if self.config.use_swanlab:
            try: swanlab.log(metrics)
            except Exception as e: print(f"Warning: Failed to log metrics to SwanLab: {str(e)}")
    
    def load_data(self):
        print("Loading dataset...")
        # ReasoningTrajectoryDataset 已经更新以支持 string list 类型的 steps
        dataset = ReasoningTrajectoryDataset(
            self.config.data_path,
            self.tokenizer,
            self.config.max_length
        )
        
        subset_size = getattr(self.config, "debug_data_size", -1)
        if subset_size and subset_size > 0 and subset_size < len(dataset):
            print(f"Debug: limiting dataset from {len(dataset)} to {subset_size} samples")
            dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
        
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self._create_collate_fn()
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, collate_fn=self._create_collate_fn()
        )
    
    def compare_and_save_trajectories(self, epoch):
        print("Comparing generated trajectories with expert trajectories...")
        output_path = os.path.join(self.config.results_dir, f"trajectory_comparison_epoch_{epoch}.json")
        # 如果已经存在结果文件，直接复用，避免重复跑 2000 条生成导致 100+ 小时耗时
        if os.path.exists(output_path):
            print(f"[compare_trajectories] Found existing comparison: {output_path} (skip recompute)")
            return output_path
        compare_trajectories(
            self.config.data_path,
            self.config.model_name,
            output_path,
            device=str(self.device),
            generator=self.generator
        )
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Trajectory comparison results saved to {output_path}")
        return output_path
    
    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        try:
            self.reward_normalizer_eval.mean = self.reward_normalizer.mean
            self.reward_normalizer_eval.var = getattr(self.reward_normalizer, 'var', 1.0)
            self.reward_normalizer_eval.count = self.reward_normalizer.count
            self.use_eval_rms = True
        except Exception: self.use_eval_rms = False
        
        total_disc_loss = 0
        total_gen_reward = 0
        num_batches = 0
        total_generated_tokens = 0
        total_sequences = 0
        
        with torch.no_grad():
            for expert_batch in tqdm(self.test_loader, desc="Evaluating"):
                expert_input_ids = expert_batch['input_ids'].to(self.discriminator_device)
                expert_attention_mask = expert_batch['attention_mask'].to(self.discriminator_device)
                
                generated_trajectories = []
                for question in expert_batch['question']:
                    # 🔥 保持 Prompt 格式一致性: 专家数据 question 已经是 "Generate a sentence with: ..."
                    traj = self.generator.generate_single_trajectory(question)
                    generated_trajectories.append(traj)
                    if traj and 'full_text' in traj:
                        total_generated_tokens += len(self.tokenizer(traj['full_text']).input_ids)
                        total_sequences += 1
                    
                gen_texts = [traj['full_text'] for traj in generated_trajectories if traj and 'full_text' in traj]
                if not gen_texts: continue

                gen_encodings = self.tokenizer(gen_texts, padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt')
                gen_step_spans = [self.compute_step_spans(traj.get('question', ''), traj.get('steps_text', [])) for traj in generated_trajectories]
                
                gen_batch = {
                    'input_ids': gen_encodings['input_ids'].to(self.discriminator_device),
                    'attention_mask': gen_encodings['attention_mask'].to(self.discriminator_device),
                    'step_spans': gen_step_spans
                }
                expert_batch_on_device = {
                    'input_ids': expert_input_ids,
                    'attention_mask': expert_attention_mask,
                    'step_spans': expert_batch['step_spans']
                }

                loss_dict = self.discriminator.compute_loss(expert_batch_on_device, gen_batch)
                total_disc_loss += loss_dict['total_loss'].item()

                for traj in generated_trajectories:
                    if not traj or 'full_text' not in traj: continue
                    normalized_reward, raw_reward = self.get_unified_reward(traj['full_text'], is_training=False)
                    total_gen_reward += normalized_reward
                
                num_batches += 1

        avg_disc_loss = total_disc_loss / num_batches if num_batches > 0 else 0
        from typing import Sized
        dataset_obj = getattr(self.test_loader, 'dataset', None)
        dataset_len = len(dataset_obj) if isinstance(dataset_obj, Sized) else 1
        avg_gen_reward = total_gen_reward / dataset_len
        avg_len = (total_generated_tokens / max(1, total_sequences))
        
        eval_stats = {
            'eval_disc_loss': avg_disc_loss,
            'eval_gen_reward': avg_gen_reward,
            'eval_avg_gen_len': avg_len
        }
        self.generator.train()
        self.discriminator.train()
        return eval_stats
    
    def train(self, skip_initial_eval=False):
        print("Starting training...")
        self.load_data()
        best_reward = float('-inf') 
        patience = 5
        no_improvement = 0
        
        if not skip_initial_eval:
            self.compare_and_save_trajectories(0)
        
        try:
            start_epoch = getattr(self.config, 'start_epoch', 0) or 0
            for epoch in range(start_epoch, self.config.num_epochs):
                epoch_stats = {
                    'disc_loss': [], 'disc_acc': [], 'gen_loss': [],
                    'reward_mean': [], 'entropy_loss': [], 'batch_sizes': []
                }
                
                self.generator.train()
                self.discriminator.train()
                
                pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
                
                for i, expert_batch in pbar:
                    # 1. 判别器训练
                    self.discriminator.train()
                    
                    # Check gate
                    should_skip_d = False
                    if self.disc_state == self.DiscState.FROZEN:
                        self.disc_frozen_steps_left -= 1
                        if self.disc_frozen_steps_left <= 0:
                            self.disc_state = self.DiscState.ACTIVE
                        should_skip_d = True
                    elif self.disc_state == self.DiscState.SKIP:
                        self.disc_skip_batches_left -= 1
                        if self.disc_skip_batches_left <= 0:
                            self.disc_state = self.DiscState.ACTIVE
                        should_skip_d = True
                    
                    if not should_skip_d:
                        self.discriminator_optimizer.zero_grad()
                        expert_input_ids = expert_batch['input_ids'].to(self.discriminator_device)
                        expert_attention_mask = expert_batch['attention_mask'].to(self.discriminator_device)
                        expert_step_spans = expert_batch['step_spans']
                        
                        # Generate Fake Data
                        generated_trajectories = []
                        with torch.no_grad():
                            self.generator.eval()
                            for question in expert_batch['question']:
                                traj = self.generator.generate_single_trajectory(question)
                                generated_trajectories.append(traj)
                                torch.cuda.empty_cache()
                            self.generator.train()
                        
                        gen_texts = [traj['full_text'] for traj in generated_trajectories]
                        gen_step_spans = [self.compute_step_spans(traj.get('question', ''), traj.get('steps_text', [])) for traj in generated_trajectories]
                        gen_encodings = self.tokenizer(gen_texts, padding=True, truncation=True, max_length=self.config.max_length, return_tensors='pt')
                        
                        gen_batch = {
                            'input_ids': gen_encodings['input_ids'].to(self.discriminator_device),
                            'attention_mask': gen_encodings['attention_mask'].to(self.discriminator_device),
                            'step_spans': gen_step_spans
                        }
                        expert_batch_on_device = {
                            'input_ids': expert_input_ids,
                            'attention_mask': expert_attention_mask,
                            'step_spans': expert_step_spans
                        }
                        
                        avg_disc_loss = 0.0
                        for disc_step in range(self.config.disc_steps_per_batch):
                            self.discriminator_optimizer.zero_grad()
                            loss_dict = self.discriminator.compute_loss(
                                expert_batch_on_device, gen_batch,
                                lambda_step=self.config.lambda_loss_step,
                                mu_prefix=self.config.mu_loss_prefix,
                                use_label_smoothing=True
                            )
                            disc_loss = loss_dict['total_loss']
                            disc_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.discriminator.get_trainable_parameters(), max_norm=1.0)
                            self.discriminator_optimizer.step()
                            avg_disc_loss += disc_loss.item()
                        
                        avg_disc_loss /= max(1, self.config.disc_steps_per_batch)
                        
                        # 统计 Acc 并更新 Gate
                        with torch.no_grad():
                            expert_outputs = self.discriminator(expert_batch_on_device['input_ids'], expert_batch_on_device['attention_mask'])
                            gen_outputs = self.discriminator(gen_batch['input_ids'], gen_batch['attention_mask'])
                            expert_logits = self.pick_head_outputs(expert_outputs)
                            gen_logits = self.pick_head_outputs(gen_outputs)
                            expert_acc = (expert_logits.sigmoid() > 0.5).float().mean().item()
                            gen_acc = (gen_logits.sigmoid() < 0.5).float().mean().item()
                            disc_acc = (expert_acc + gen_acc) / 2
                            
                            print(f"  [Gate] D Loss: {avg_disc_loss:.4f}, Acc: {disc_acc:.4f} (Exp: {expert_acc:.2f}, Gen: {gen_acc:.2f})")
                            
                            if disc_acc > self.config.disc_update_threshold:
                                self.d_too_strong_counter += 1
                                if self.d_too_strong_counter >= 3:
                                    self.disc_state = self.DiscState.SKIP
                                    self.disc_skip_batches_left = 3
                                    self.d_too_strong_counter = 0
                                    print("  [Gate] D too strong -> SKIP 3 batches")
                            else:
                                self.d_too_strong_counter = 0
                    else:
                        avg_disc_loss = 0.0
                        disc_acc = 0.0

                    # 2. 生成器训练 (PPO)
                    self.generator.train()
                    self.generator_optimizer.zero_grad()
                    
                    all_obs_ids, all_obs_masks, all_action_ids, all_log_probs, all_values, all_rewards = [], [], [], [], [], []
                    
                    for question in expert_batch['question']:
                        current_obs_ids = self.tokenizer(question, return_tensors='pt').input_ids.to(self.generator_device)
                        traj_steps = []
                        
                        for t in range(getattr(self.config, 'ppo_max_steps', 128)):
                            obs_mask = torch.ones_like(current_obs_ids)
                            with torch.no_grad():
                                action_id, log_prob, _, value_pred = self.generator.sample_action(current_obs_ids, obs_mask)
                            
                            traj_steps.append({
                                'obs_ids': current_obs_ids.squeeze(0),
                                'obs_masks': obs_mask.squeeze(0),
                                'action_ids': action_id,
                                'log_probs': log_prob,
                                'values': value_pred
                            })
                            current_obs_ids = torch.cat([current_obs_ids, action_id.unsqueeze(0)], dim=1)
                            if action_id.item() == self.tokenizer.eos_token_id or current_obs_ids.shape[1] >= self.config.max_length:
                                break
                                
                        full_text = self.tokenizer.decode(current_obs_ids.squeeze(0), skip_special_tokens=True)
                        normalized_reward, raw_reward = self.get_unified_reward(full_text, is_training=True)
                        print(f"    Gen Reward: {raw_reward:.4f} (Norm: {normalized_reward:.4f}) | {full_text[:50]}...")
                        
                        # Compute Returns
                        R = normalized_reward
                        for step in reversed(traj_steps):
                            all_obs_ids.append(step['obs_ids'])
                            all_obs_masks.append(step['obs_masks'])
                            all_action_ids.append(step['action_ids'])
                            all_log_probs.append(step['log_probs'])
                            all_values.append(step['values'])
                            all_rewards.append(torch.tensor([R], device=self.generator_device, dtype=self.dtype))
                            R = -0.001 + self.config.reward_discount * R # Step penalty
                    
                    # PPO Update
                    if len(all_obs_ids) > 0:
                        # Pad batch
                        obs_ids_pad = torch.nn.utils.rnn.pad_sequence(all_obs_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                        obs_masks_pad = torch.nn.utils.rnn.pad_sequence(all_obs_masks, batch_first=True, padding_value=0)
                        actions_t = torch.cat(all_action_ids)
                        log_probs_t = torch.cat(all_log_probs)
                        values_t = torch.cat(all_values)
                        rewards_t = torch.cat(all_rewards)
                        
                        # Minibatch PPO
                        ppo_batch_size = 8
                        accum_steps = (len(actions_t) + ppo_batch_size - 1) // ppo_batch_size
                        gen_loss_accum = 0
                        
                        for j in range(0, len(actions_t), ppo_batch_size):
                            end = min(j + ppo_batch_size, len(actions_t))
                            
                            # Re-compute log probs
                            new_log_probs, new_values, logits_last = self.generator.get_log_probs_and_values(
                                obs_ids_pad[j:end], obs_masks_pad[j:end], actions_t[j:end]
                            )
                            
                            with autocast(dtype=torch.bfloat16):
                                policy_loss, value_loss = self.ppo_step(None, new_log_probs, rewards_t[j:end], log_probs_t[j:end], new_values)
                                if policy_loss is not None:
                                    # Entropy
                                    probs = torch.softmax(logits_last, dim=-1)
                                    log_probs = torch.log_softmax(logits_last, dim=-1)
                                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                                    
                                    loss = policy_loss + self.config.value_loss_weight * value_loss - self.config.entropy_weight * entropy
                                    (loss / accum_steps).backward()
                                    gen_loss_accum += loss.item()
                        
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                        self.generator_optimizer.step()
                        
                        epoch_stats['gen_loss'].append(gen_loss_accum / accum_steps)
                        epoch_stats['reward_mean'].append(rewards_t.mean().item())
                        epoch_stats['disc_loss'].append(avg_disc_loss)
                        epoch_stats['disc_acc'].append(disc_acc)
                        epoch_stats['batch_sizes'].append(len(expert_batch['question']))
                    
                    self.current_batch += 1
                    self._train_step += 1
                
                # Epoch End Logging
                avg_reward = np.mean(epoch_stats['reward_mean'])
                print(f"Epoch {epoch+1} finished. Avg Reward: {avg_reward:.4f}")
                self.log_metrics({'epoch_reward': avg_reward, 'epoch_disc_acc': np.mean(epoch_stats['disc_acc'])})
                
                # Evaluate & Save
                eval_stats = self.evaluate()
                print(f"Eval: {eval_stats}")
                self.log_metrics(eval_stats)
                self.save_models('latest')
                if eval_stats['eval_gen_reward'] > best_reward:
                    best_reward = eval_stats['eval_gen_reward']
                    self.save_models('best')
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print("Early stopping")
                        break
        except KeyboardInterrupt:
            # ✅ 断电/手动中断：尽最大可能保存当前进度（至少保住上一轮 epoch 的最新权重）
            try:
                print("\n🛑 Caught KeyboardInterrupt. Saving interrupt checkpoints...")
                self.save_models('interrupt')
            except Exception as e:
                print(f"⚠️ Failed to save interrupt checkpoint: {e}")
            raise

    def save_models(self, tag):
        self.generator.save(os.path.join(self.config.save_dir, f'generator_{tag}.pt'))
        self.discriminator.save(os.path.join(self.config.save_dir, f'discriminator_{tag}.pt'))
        print(f"Saved {tag} models")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-initial-eval', action='store_true')
    parser.add_argument('--fast-debug', action='store_true')
    parser.add_argument('--resume-from', type=str, default="", help="Resume training from an existing run directory (CommonGEN/checkpoints/run_...)")
    parser.add_argument('--resume-tag', type=str, default="latest", choices=["latest", "best", "interrupt"])
    parser.add_argument('--start-epoch', type=int, default=0, help="Start from this epoch index (0-based). E.g. --start-epoch 1 means skip epoch 0.")
    parser.add_argument('--num-epochs', type=int, default=None, help="Override total number of epochs (default: use config value)")
    args = parser.parse_args()
    
    config = Config(fast_debug=args.fast_debug)
    # 覆盖 epoch 相关参数（优先命令行）
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    config.start_epoch = args.start_epoch
    # 断点续训参数注入：resume-from 会复用该 run 目录作为保存目录，并默认跳过 compare_trajectories
    if args.resume_from:
        config.resume_from = args.resume_from
        config.resume_tag = args.resume_tag
        config.save_dir = args.resume_from
        # 推断 results_dir：把 /checkpoints/ 替换为 /results/
        try:
            config.results_dir = args.resume_from.replace("/checkpoints/", "/results/")
        except Exception:
            pass
        # generator adapter 也从该 run 的 generator_{tag}.pt 目录加载
        gen_resume_path = os.path.join(args.resume_from, f"generator_{args.resume_tag}.pt")
        if os.path.exists(gen_resume_path):
            config.sft_adapter_path = gen_resume_path
        # resume 时默认跳过初始 compare
        args.skip_initial_eval = True
    trainer = GAILTrainer(config)
    trainer.train(skip_initial_eval=args.skip_initial_eval)
