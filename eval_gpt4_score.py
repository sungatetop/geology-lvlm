import os
import json
import argparse
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import llm
import util
INSTRUCT_PROMPT = """我们想请您对上述用户问题的AI助手回答进行评估。用户在观察图片时提出了问题。为了您的参考，图片的内容通过图片说明来表示。
# 请评估回答的以下几个方面：
# 1. 专业性（professionality）：回答是否较好的使用专业词汇，是否引用规范等专业材料
# 2. 准确性（Accuracy）：回答的内容是否准确，与参考答案相比是否正确


请首先输出一行，仅包含一个0到10的分数，其中更高的分数表示更好的整体性表现。
在随后的几行中，请提供您评估的详细解释，说明为什么给出这个分数。"""

# INSTRUCT_PROMPT = """我们想请您对上述用户问题的AI助手回答进行评估。用户在观察图片时提出了问题。为了您的参考，图片的内容通过图片说明来表示。

# 请评估回答的以下几个方面：
# 1. 帮助性（Helpfulness）：回答是否有效解决了用户的问题
# 2. 相关性（Relevance）：回答是否紧扣问题主题
# 3. 准确性（Accuracy）：回答的内容是否准确，与参考答案相比是否正确


# 请首先输出一行，仅包含3个1到10的分数，其中更高的分数表示更好的整体表现。
# 在随后的几行中，请提供您评估的详细解释，说明为什么给出这些分数。"""

ROLE = 'Assistant'

def conv_to_str(image_path, question, label, prediction):
    return (f'[Context]\n'
            f'Image Path: {image_path}\n\n'
            f'[Question]\n{question}\n\n'
            f'[Reference Answer]\n{label}\n\n'
            f'[{ROLE}]\n{prediction}\n\n[End of {ROLE}]\n\n'
            f'[System]\n{INSTRUCT_PROMPT}\n\n')

def compare_messages_gen(image_path, question, label, prediction):
    messages = [
        {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
    ]
    messages.append({"role": "user", "content": conv_to_str(image_path, question, label, prediction)})
    return messages

def chunk(lst, n):
    """Split list into chunks of size n"""
    for i in range(0, len(lst), n):
        if i + (1.5 * n) < len(lst):
            end = i + n
        else:
            end = len(lst)
        yield lst[i:end]
        if end == len(lst):
            return

def process_file(input_file, output_file, model_inst):
    """Process a single evaluation file"""
    print(f'Processing file: {input_file}')
    
    # Load the input data
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    results = []
    BATCH_SIZE = 10

    for sample in tqdm(samples):
        # 处理每个样本中的所有对话轮次
        for i, conv in enumerate(sample['conversations']):
            # 创建当前轮次的结果对象
            turn_result = {
                'id': f"{sample['id']}_turn_{i+1}",  # 添加轮次编号
                'original_id': sample['id'],
                'turn_number': i + 1,
                'total_turns': len(sample['conversations']),
                'conversation': conv  # 保存当前轮次的对话内容
            }
            
            image_path = conv['images'][0] if conv.get('images') else "No image"
            question = conv['user_input']
            prediction = conv['prediction']
            label = conv.get('label', '')
            
            input_msg = compare_messages_gen(image_path, question, label, prediction)
            
            # 对当前轮次进行评估
            inference_results = [x.strip() for chunk_messages in chunk([input_msg], BATCH_SIZE) 
                               for x in model_inst.infer(chunk_messages)]
            
            # 保存评估结果
            turn_result['evaluation'] = inference_results[0] if inference_results else ''
            results.append(turn_result)
            
        print(f"Processed {len(sample['conversations'])} conversations for sample {sample['id']}")

    print(f"Processed {len(results)} total turns")

    # Save results - each turn as a separate line
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for turn_result in results:
            f.write(json.dumps(turn_result, ensure_ascii=False) + '\n')

def main(args):
    # Initialize GPT-4 model
    model_inst = llm.GPT("gpt-4o-mini")  # 使用gpt-4o-mini模型
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each evaluation file
    eval_files = [
        # 'eval_cog_merge_data_v1.jsonl',
        # 'eval_complex_reasoning.jsonl',
        # 'eval_intro_conv_v1.jsonl',
        # 'eval_single_feature_judge.jsonl',
        # 'eval_support_params_v1.jsonl',
        'eval_tunnel_knowledge.jsonl'
    ]
    
    for eval_file in eval_files:
        input_file = os.path.join(args.input_dir, eval_file)
        output_file = os.path.join(args.output_dir, f'scored_{eval_file}')
        
        if os.path.exists(input_file):
            process_file(input_file, output_file, model_inst)
        else:
            print(f"Warning: Input file not found: {input_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GPT-4 Answer Scoring", add_help=True)
    parser.add_argument("--input-dir", default="eval_result/llava-1.5-7b-hf-sft-5ep", 
                      help="directory containing evaluation files")
    parser.add_argument("--output-dir", default="eval_result/llava-1.5-7b-hf-sft-5ep/gpt4_scores", 
                      help="directory to save GPT-4 score files")
    args = parser.parse_args()
    main(args)

# python eval_gpt4_score.py --input-dir eval_result/llava-1.5-13b-hf-sft-5ep --output-dir eval_result/llava-1.5-13b-hf-sft-5ep/gpt4_scores
#python eval_gpt4_score.py --input-dir eval_result/llava-1.5-7b-hf-sft-5ep --output-dir eval_result/llava-1.5-7b-hf-sft-5ep/gpt4_scores
# python eval_gpt4_score.py --input-dir eval_result/GPT4Vresults --output-dir eval_result/GPT4Vresults/gpt4_scores