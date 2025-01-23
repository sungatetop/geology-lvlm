import argparse
from pathlib import Path
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_jsonl(file_path):
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_score(gpt_eval):
    """从GPT评估结果中提取分数"""
    try:
        # 获取第一行的分数
        score = float(gpt_eval.split('\n')[0].strip())
        return score
    except:
        print(f"Warning: Could not parse score from: {gpt_eval}")
        return None

def analyze_scores(scores_dir):
    """分析所有评分文件的结果"""
    results = defaultdict(list)
    
    # 处理目录中的所有评分文件
    for file_name in os.listdir(scores_dir):
        if file_name.startswith('scored_') and file_name.endswith('.jsonl'):
            file_path = os.path.join(scores_dir, file_name)
            task_name = file_name[7:-6]  # 移除 'scored_' 和 '.jsonl'
            
            data = load_jsonl(file_path)
            for item in data:
                if 'gpt_eval' in item:
                    score = extract_score(item['gpt_eval'])
                    if score is not None:
                        results['Task'].append(task_name)
                        results['Score'].append(score)
                        # 保存评价内容以供后续分析
                        results['Evaluation'].append(item['gpt_eval'])
    
    return pd.DataFrame(results)

def visualize_scores(df, output_dir):
    """可视化评分结果"""
    # plt.style.use('seaborn')
    
    # 1. 箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Task', y='Score', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Score Distribution by Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_boxplot.png'))
    plt.close()
    
    # 2. 小提琴图
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Task', y='Score', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Score Distribution (Violin Plot) by Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_violin.png'))
    plt.close()
    
    # 3. 生成统计摘要
    summary = df.groupby('Task')['Score'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    summary.to_csv(os.path.join(output_dir, 'score_summary.csv'))
    
    # 打印统计结果
    print("\n=== Score Summary ===")
    print(summary)
    
    # 4. 保存详细的评价内容
    task_evaluations = defaultdict(list)
    for task, eval_text in zip(df['Task'], df['Evaluation']):
        task_evaluations[task].append(eval_text)
    
    with open(os.path.join(output_dir, 'detailed_evaluations.txt'), 'w', encoding='utf-8') as f:
        for task, evals in task_evaluations.items():
            f.write(f"\n\n=== {task} ===\n")
            for i, eval_text in enumerate(evals, 1):
                f.write(f"\n--- Sample {i} ---\n{eval_text}\n")

def main(args):
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(args.scores_dir), 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析分数
    df = analyze_scores(args.scores_dir)
    
    # 可视化并保存结果
    visualize_scores(df, output_dir)
    
    print(f"\nAnalysis results saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("GPT-4 Score Analysis", add_help=True)
    parser.add_argument("--scores-dir", default="eval_result/llava-1.5-7b-hf-sft-5ep/gpt4_scores",
                      help="directory containing GPT-4 score files")
    args = parser.parse_args()
    main(args)

# python summarize_gpt_review.py --scores-dir eval_result/llava-1.5-7b-hf-sft-5ep/gpt4_scores