import json
import os
import time
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")

# ================= 配置区域 =================
API_KEY = "sk-5ccb81ede30f49bcadac6b5dcf13334b"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"  # 升级为 qwen-max

# Qwen-max 比较强，我们可以让它生成更自然的 CoT
SYSTEM_PROMPT = """You are an expert in commonsense reasoning and sentence generation.
Your task is to generate a Chain-of-Thought (CoT) reasoning process for the CommonGEN task.
Given a set of concepts, you need to construct a natural sentence that includes all of them.

Output Format:
You must output the reasoning process in the following strictly structured steps:

Step 1. Identify required concepts: [List the concepts]
Step 2. Plan sentence pattern: [Describe the grammatical structure or scenario]
Step 3. Generate sentence: [The final sentence containing all concepts]
Final Answer: [Repeat the final sentence only]

Example:
Concepts: dog, frisbee, catch, throw
Step 1. Identify required concepts: dog, frisbee, catch, throw
Step 2. Plan sentence pattern: A person throws a frisbee and a dog catches it.
Step 3. Generate sentence: The boy throws the frisbee and the dog runs to catch it in the air.
Final Answer: The boy throws the frisbee and the dog runs to catch it in the air.
"""

def generate_cot(concepts, client, model):
    """调用 LLM 生成 CoT"""
    if isinstance(concepts, list):
        concepts_str = ", ".join(concepts)
    else:
        concepts_str = str(concepts)
        
    user_prompt = f"Concepts: {concepts_str}\n\nPlease generate a sentence using these concepts with detailed reasoning steps."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            # max_tokens=512 # Qwen API 可能不需要严格限制，或者根据需要调整
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_cot_response(response_text):
    """解析 LLM 的输出为结构化格式"""
    if not response_text:
        return []
        
    lines = response_text.strip().split('\n')
    steps = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 兼容一些常见的变体
        if (line.startswith("Step") or 
            line.startswith("Final Answer") or 
            line.startswith("1.") or 
            line.startswith("2.") or 
            line.startswith("3.")):
            steps.append(line)
            
    return steps

def process_item(item, client, model):
    """处理单条数据"""
    # CommonGEN 原始数据格式: {"source": "dog_N#run_V", "target": "..."}
    raw_concepts = item.get('source', '')
    if not raw_concepts:
        # 尝试兼容其他格式
        raw_concepts = item.get('concept_set', '')
    
    if not raw_concepts:
        return None
        
    # 清洗 concepts: 去掉 _N, _V 等后缀，并按 # 分割
    # 例如: "broccoli_N#cheese_N" -> ["broccoli", "cheese"]
    concepts = []
    for c in raw_concepts.split('#'):
        # 去掉最后的 _X 部分
        if '_' in c:
            c = c.rsplit('_', 1)[0]
        concepts.append(c)
    
    if not concepts:
        return None
        
    cot_text = generate_cot(concepts, client, model)
    
    if cot_text:
        steps = parse_cot_response(cot_text)
        if len(steps) >= 3: 
            return {
                "question": f"Generate a sentence with: {', '.join(concepts)}",
                "concepts": concepts,
                "is_expert": True,
                "steps": steps,
                "raw_llm_output": cot_text,
                "gold_references": [item.get('target', '')] # CommonGEN 只有单条 target
            }
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw CommonGEN jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save generated JSON")
    parser.add_argument("--limit", type=int, default=2000, help="Limit number of items (default 2000 to save cost)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel threads") # API 可以并发高一点
    args = parser.parse_args()

    # 初始化 Client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 读取原始数据
    print(f"Reading from {args.input_file}...")
    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    total_data = len(data)
    if args.limit and args.limit < total_data:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} items (out of {total_data})")
    else:
        print(f"Processing all {len(data)} items")
    
    print(f"Starting generation with {args.workers} workers using {MODEL_NAME}...")
    
    results = []
    # 使用线程池并发调用
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, item, client, MODEL_NAME) for item in data]
        
        for future in tqdm(futures, total=len(data)):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Worker error: {e}")
                
    # 保存结果
    print(f"Generated {len(results)} valid trajectories.")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
