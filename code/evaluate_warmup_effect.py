import argparse
import datetime
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from tqdm import tqdm

import sys

# allow importing top-level GAIL_train modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from discriminator import ReasoningTrajectoryDataset, HierarchicalDiscriminator
from generator import Generator


@dataclass
class EvalConfig:
    model_path: str
    data_path: str
    adapter_path: str
    discriminator_head_path: str
    max_length: int = 256
    num_samples: int = 50
    seed: int = 42
    output_dir: str = "../results"
    max_new_tokens: int = 64
    reward_mode: str = "hybrid"
    alpha_reward: float = 0.5
    beta_reward: float = 0.3
    gamma_reward: float = 0.2
    hybrid_blend: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_concepts(question: str) -> List[str]:
    # CommonGEN prompt format: "Generate a sentence with: a, b, c"
    lower = question.lower()
    if ":" in lower:
        tail = lower.split(":", 1)[1]
    else:
        tail = lower
    concepts = [c.strip(" ,.;") for c in tail.split(",") if c.strip(" ,.;")]
    return [c for c in concepts if c]


def coverage_ratio(text: str, concepts: List[str]) -> float:
    if not concepts:
        return 0.0
    lower = text.lower()
    covered = sum(1 for c in concepts if c in lower)
    return covered / len(concepts)


def repetition_stats(tokens: List[int]) -> Tuple[float, float]:
    if len(tokens) < 2:
        return 0.0, 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    if not bigrams:
        return 0.0, 0.0
    unique_bigrams = len(set(bigrams))
    bigram_repeat_ratio = 1.0 - (unique_bigrams / len(bigrams))

    # Degenerate run ratio: fraction of tokens that are part of runs >= 3
    run_tokens = 0
    run_len = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run_len += 1
        else:
            if run_len >= 3:
                run_tokens += run_len
            run_len = 1
    if run_len >= 3:
        run_tokens += run_len
    degenerate_ratio = run_tokens / len(tokens)
    return bigram_repeat_ratio, degenerate_ratio


def structure_stats(text: str, steps: List[Dict]) -> Tuple[int, int]:
    step_marker = bool(re.search(r"\bStep\s*\d+", text))
    step_count = len(steps) if steps is not None else 0
    return step_count, int(step_marker)


def load_adapter_config(adapter_path: str) -> Dict:
    if not adapter_path:
        return {}
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_generator(model_path: str, device: str, max_length: int, adapter_cfg: Dict, max_new_tokens: int) -> Generator:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map={"": device},
        local_files_only=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=int(adapter_cfg.get("r", 8)),
        lora_alpha=int(adapter_cfg.get("lora_alpha", 16)),
        target_modules=adapter_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=float(adapter_cfg.get("lora_dropout", 0.05)),
        bias=adapter_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=adapter_cfg.get("modules_to_save", ["lm_head"]),
    )
    peft_model = get_peft_model(base_model, lora_config)
    try:
        peft_model.base_model.model.config.use_cache = False
    except Exception:
        pass

    generator = Generator(peft_model, tokenizer, device=device)
    generator.gen_kwargs["max_new_tokens"] = min(generator.gen_kwargs.get("max_new_tokens", 180), max_new_tokens)
    return generator


def attach_adapter(generator: Generator, adapter_path: str) -> None:
    if adapter_path and os.path.exists(adapter_path):
        generator.model.load_adapter(adapter_path, adapter_name="default")
        try:
            generator.model.set_adapter("default")
        except Exception:
            pass


def build_discriminator(model_path: str, device: str, max_length: int, head_path: str) -> Tuple[HierarchicalDiscriminator, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    encoder = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map={"": device},
        local_files_only=True,
    )
    encoder.config.use_cache = False
    discriminator = HierarchicalDiscriminator(encoder, tokenizer, device=device, max_length=max_length)
    if head_path and os.path.exists(head_path):
        state_dict = torch.load(head_path, map_location=device)
        head_state = {k: v for k, v in state_dict.items() if "encoder" not in k}
        discriminator.load_state_dict(head_state, strict=False)
    discriminator.eval()
    return discriminator, tokenizer


def reward_from_logits(seq_logits, step_logits, prefix_logits, cfg: EvalConfig) -> torch.Tensor:
    r_seq = F.softplus(seq_logits).mean()
    r_step = F.softplus(step_logits).mean()
    r_prefix = F.softplus(prefix_logits).mean()
    raw_composite = cfg.alpha_reward * r_seq + cfg.beta_reward * r_step + cfg.gamma_reward * (r_prefix - r_seq)
    if cfg.reward_mode == "hybrid":
        return (1 - cfg.hybrid_blend) * r_seq + cfg.hybrid_blend * raw_composite
    if cfg.reward_mode == "composite":
        return raw_composite
    return r_seq


def score_text(discriminator: HierarchicalDiscriminator, tokenizer: Any, text: str, cfg: EvalConfig) -> float:
    enc = tokenizer(text, max_length=cfg.max_length, truncation=True, return_tensors="pt")
    enc = {k: v.to(discriminator.device) for k, v in enc.items()}
    with torch.no_grad():
        seq_logits, step_logits, prefix_logits = discriminator(enc["input_ids"], enc["attention_mask"])
        reward = reward_from_logits(seq_logits, step_logits, prefix_logits, cfg)
    return float(reward.detach().cpu().item())


def summarize_rewards(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def run_eval(cfg: EvalConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)

    gpu_count = torch.cuda.device_count()
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    disc_device = "cuda:1" if gpu_count >= 2 else gen_device

    adapter_cfg = load_adapter_config(cfg.adapter_path)
    generator = build_generator(cfg.model_path, gen_device, cfg.max_length, adapter_cfg, cfg.max_new_tokens)
    discriminator, disc_tokenizer = build_discriminator(
        cfg.model_path, disc_device, cfg.max_length, cfg.discriminator_head_path
    )

    dataset = ReasoningTrajectoryDataset(cfg.data_path, disc_tokenizer, cfg.max_length)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: min(cfg.num_samples, len(indices))]

    def eval_condition(label: str, use_adapter: bool) -> Dict[str, Any]:
        if use_adapter:
            attach_adapter(generator, cfg.adapter_path)

        samples = []
        gen_rewards = []
        exp_rewards = []
        coverages = []
        step_counts = []
        step_markers = []
        lengths = []
        repeat_ratios = []
        degenerate_ratios = []

        for idx in tqdm(indices, desc=f"Generating ({label})", leave=False):
            item = dataset[idx]
            question = item["question"]
            expert_text = item["full_text"]
            concepts = parse_concepts(question)

            traj = generator.generate_single_trajectory(question)
            gen_text = traj.get("full_text", "")
            steps = traj.get("steps", [])

            cov = coverage_ratio(gen_text, concepts)
            tokens = disc_tokenizer.encode(gen_text, add_special_tokens=False)
            rep_ratio, deg_ratio = repetition_stats(tokens)
            step_count, step_marker = structure_stats(gen_text, steps)
            length = len(tokens)

            gen_reward = score_text(discriminator, disc_tokenizer, gen_text, cfg)
            exp_reward = score_text(discriminator, disc_tokenizer, expert_text, cfg)

            samples.append(
                {
                    "question": question,
                    "concepts": concepts,
                    "gen_text": gen_text,
                    "expert_text": expert_text,
                    "coverage_ratio": cov,
                    "step_count": step_count,
                    "step_marker": step_marker,
                    "length": length,
                    "bigram_repeat_ratio": rep_ratio,
                    "degenerate_ratio": deg_ratio,
                    "gen_reward": gen_reward,
                    "expert_reward": exp_reward,
                    "margin_expert_minus_gen": exp_reward - gen_reward,
                }
            )

            gen_rewards.append(gen_reward)
            exp_rewards.append(exp_reward)
            coverages.append(cov)
            step_counts.append(step_count)
            step_markers.append(step_marker)
            lengths.append(length)
            repeat_ratios.append(rep_ratio)
            degenerate_ratios.append(deg_ratio)

        summary = {
            "reward": summarize_rewards(gen_rewards),
            "expert_reward": summarize_rewards(exp_rewards),
            "margin_expert_minus_gen": summarize_rewards([e - g for e, g in zip(exp_rewards, gen_rewards)]),
            "coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
            "step_count_mean": float(np.mean(step_counts)) if step_counts else 0.0,
            "step_marker_rate": float(np.mean(step_markers)) if step_markers else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "bigram_repeat_mean": float(np.mean(repeat_ratios)) if repeat_ratios else 0.0,
            "degenerate_ratio_mean": float(np.mean(degenerate_ratios)) if degenerate_ratios else 0.0,
        }
        return {"label": label, "summary": summary, "samples": samples}

    baseline = eval_condition("no_warmup", use_adapter=False)
    warmup = eval_condition("warmup", use_adapter=True)

    return {"config": asdict(cfg), "baseline": baseline, "warmup": warmup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/zhoukaining/.cache/huggingface/models--Qwen2.5-7B-Instruct")
    parser.add_argument("--data-path", default="/home/zhoukaining/pro_cusor/GAIL_train/CommonGEN/data/data_commongenv/commongenv_cot_train_llm.json")
    parser.add_argument("--adapter-path", default="../checkpoints/sft_warmup/final_adapter")
    parser.add_argument("--discriminator-head-path", default="../checkpoints/discriminator_pretrained_gap0.9866.pt")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="../results")
    args = parser.parse_args()

    cfg = EvalConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        adapter_path=args.adapter_path,
        discriminator_head_path=args.discriminator_head_path,
        max_length=args.max_length,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    results = run_eval(cfg)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_dir, f"warmup_effect_eval_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_path}")
    print("Summary (no_warmup):", results["baseline"]["summary"])
    print("Summary (warmup):", results["warmup"]["summary"])


if __name__ == "__main__":
    main()
