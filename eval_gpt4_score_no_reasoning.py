import os
import json
import argparse
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import llm
import util
INSTRUCT_PROMPT = """你是一位理性的评价者,需要对用户的答题进行打分。请根据正确答案对用户的答题结果进行评分，评分范围是0-10分,分数越高表示用户的回答越准确，请首先输出一行,仅包含一个0到10的分数,表示答题的得分。接下来清提供详细的评估说明，避免任何潜在偏见，并确保回答的呈现顺序不会影响您的判断。"""

# INSTRUCT_PROMPT = """我们想请您对上述用户问题的AI助手回答进行评估。用户在观察图片时提出了问题。为了您的参考，图片的内容通过图片说明来表示。
# # 请评估回答的以下几个方面：
# # 1. 专业性（professionality）：回答是否较好的使用专业词汇，是否引用规范等专业材料
# # 2. 准确性（Accuracy）：回答的内容是否准确，与参考答案相比是否正确


# 请首先输出一行，仅包含一个0到10的分数，其中更高的分数表示更好的整体性表现。
# 在随后的几行中，请提供您评估的详细解释，说明为什么给出这个分数。"""

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
    BATCH_SIZE = 100
    invalid_samples = []  # 用于记录无效样本的ID
    
    # Prepare batches of messages
    current_batch = []
    current_batch_results = []
    
    for sample in samples:
        for i, conv in enumerate(sample['conversations']):
            try:
                turn_result = {
                    'id': f"{sample['id']}_turn_{i+1}",
                    'original_id': sample['id'],
                    'turn_number': i + 1,
                    'total_turns': len(sample['conversations']),
                }
                
                image_path = conv['images'][0] if conv.get('images') else "No image"
                question = conv['user_input']
                prediction = conv['prediction']
                # Extract content after \n</think>\n
                processed_prediction = prediction
                think_marker = "\n</think>\n"
                think_pos = prediction.find(think_marker)
                print(f"\n=== Debug Info for {sample['id']}_turn_{i+1} ===")
                print(f"Original prediction: {prediction}")
                print(f"Think marker position: {think_pos}")
                if think_pos != -1:
                    processed_prediction = prediction[think_pos + len(think_marker):].strip()
                    print(f"Processed prediction: {processed_prediction}")
                else:
                    print("Warning: Could not find think marker!")
                    invalid_samples.append(f"{sample['id']}_turn_{i+1}")
                    continue  # 跳过此样本
                label = conv.get('label', '')
                
                # Create conversation without prediction
                conversation = {
                    'images': conv['images'],
                    'user_input': conv['user_input'],
                    'label': label
                }
                turn_result['conversation'] = conversation
                turn_result['prediction'] = processed_prediction
                
                input_msg = compare_messages_gen(image_path, question, label, processed_prediction)
                current_batch.append(input_msg)
                current_batch_results.append(turn_result)
                
                # Process batch when it reaches BATCH_SIZE
                if len(current_batch) >= BATCH_SIZE:
                    inference_results = model_inst.infer(current_batch)
                    
                    # Update results with evaluations
                    for result, evaluation in zip(current_batch_results, inference_results):
                        result['evaluation'] = evaluation.strip()
                        results.append(result)
                    
                    print(f"Processed batch of {len(current_batch)} messages")
                    current_batch = []
                    current_batch_results = []
            except Exception as e:
                print(f"Error processing sample {sample['id']}_turn_{i+1}: {str(e)}")
                invalid_samples.append(f"{sample['id']}_turn_{i+1}")
                continue
    
    # Process remaining messages if any
    if current_batch:
        inference_results = model_inst.infer(current_batch)
        for result, evaluation in zip(current_batch_results, inference_results):
            result['evaluation'] = evaluation.strip()
            results.append(result)
        print(f"Processed final batch of {len(current_batch)} messages")

    print(f"Processed {len(results)} total turns")
    if invalid_samples:
        print(f"Found {len(invalid_samples)} invalid samples:")
        for sample_id in invalid_samples:
            print(f"  - {sample_id}")

    # Save results - each turn as a separate line
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for turn_result in results:
            f.write(json.dumps(turn_result, ensure_ascii=False) + '\n')

def main(args):
    # Initialize GPT-4 model
    model_inst = llm.GPT("gpt-4o-mini")  # 使用gpt-4o-mini模型
    # model_inst = llm.GPT("gpt-4o")  # 使用gpt-4o-mini模型

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each evaluation file
    eval_files = [
        # 'eval_cog_merge_data_v1.jsonl',
        # 'eval_complex_reasoning.jsonl',
        # # 'eval_intro_conv_v1.jsonl',
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
#python eval_gpt4_score.py --input-dir eval_result/llava-1.5-7b-hf --output-dir eval_result/llava-1.5-7b-hf/gpt4_scores
# python eval_gpt4_score.py --input-dir eval_result/llava-1.5-13b-hf --output-dir eval_result/llava-1.5-13b-hf/gpt4_scores
# python eval_gpt4_score.py --input-dir eval_result/qwen2_vl_full_sft_2ep --output-dir eval_result/qwen2_vl_full_sft_2ep/gpt4_scores
# python eval_gpt4_score.py --input-dir eval_result/qwen2_vl_lora_sft_15ep --output-dir eval_result/qwen2_vl_lora_sft_15ep/gpt4_scores
# python eval_gpt4_score.py --input-dir eval_result/qwen2-vl-5ep --output-dir eval_result/qwen2-vl-5ep/gpt4_scores


# python eval_gpt4_score.py --input-dir eval_result/llava-1.5-13b-hf-sft-5ep --output-dir eval_result/llava-1.5-13b-hf-sft-5ep/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/llava-1.5-7b-hf-sft-5ep --output-dir eval_result/llava-1.5-7b-hf-sft-5ep/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/GPT4Vresults --output-dir eval_result/GPT4Vresults/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/llava-1.5-7b-hf --output-dir eval_result/llava-1.5-7b-hf/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/llava-1.5-13b-hf --output-dir eval_result/llava-1.5-13b-hf/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/qwen2_vl_full_sft_2ep --output-dir eval_result/qwen2_vl_full_sft_2ep/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/qwen2_vl_lora_sft_15ep --output-dir eval_result/qwen2_vl_lora_sft_15ep/gpt4_scores_gpt4
# python eval_gpt4_score.py --input-dir eval_result/qwen2-vl-5ep --output-dir eval_result/qwen2-vl-5ep/gpt4_scores_gpt4

# python eval_gpt4_score_no_reasoning.py --input-dir eval_result/deepseek_r1_distill_qwen2_vl_7b --output-dir eval_result/deepseek_r1_distill_qwen2_vl_7b/gpt4_scores_gpt4
