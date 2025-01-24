import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def extract_score(evaluation):
    """从评估文本中提取分数"""
    if not evaluation:
        return None
    try:
        # 获取第一行的数字作为分数
        score = float(evaluation.split('\n')[0])
        return score
    except:
        return None

def analyze_scores(scores_dir, output_dir):
    """分析评分结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_scores = []
    all_evaluations = []  # 存储所有评估结果
    conversation_scores = defaultdict(list)  # 按对话ID存储所有轮次的分数
    
    # 处理每个评分文件
    for filename in os.listdir(scores_dir):
        if not filename.startswith('scored_'):
            continue
            
        task_name = filename.replace('scored_', '').replace('.jsonl', '')
        file_path = os.path.join(scores_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                score = extract_score(result['evaluation'])
                if score is not None:
                    # 记录分数信息
                    all_scores.append({
                        'Task': task_name,
                        'Score': score,
                        'Original_ID': result['original_id'],
                        'Turn': result['turn_number']
                    })
                    # 收集对话级别的分数
                    conversation_scores[result['original_id']].append(score)
                    # 保存评估信息
                    all_evaluations.append({
                        'Task': task_name,
                        'ID': result['original_id'],
                        'Turn': result['turn_number'],
                        'Question': result['conversation']['user_input'],
                        'Evaluation': result['evaluation']
                    })
    
    if not all_scores:
        print("No scores found!")
        return
        
    # 转换为DataFrame
    df = pd.DataFrame(all_scores)
    
    # 生成对话级别统计
    conversation_stats = []
    for conv_id, scores in conversation_scores.items():
        conversation_stats.append({
            'Conversation_ID': conv_id,
            'Avg_Score': round(sum(scores) / len(scores), 2),
            'Min_Score': min(scores),
            'Max_Score': max(scores),
            'Turn_Count': len(scores)
        })
    
    # 创建DataFrame并按平均分降序排序
    conv_df = pd.DataFrame(conversation_stats)
    conv_df = conv_df.sort_values('Avg_Score', ascending=False)
    
    # 处理每个评估文件
    for filename in os.listdir(scores_dir):
        if not filename.startswith('scored_'):
            continue
        task_name = filename.replace('scored_', '').replace('.jsonl', '')
        
        # 筛选当前任务的数据
        task_df = df[df['Task'] == task_name]
        
        # 1. 保存文件级别统计
        task_stats = task_df['Score'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        task_stats = pd.DataFrame(task_stats).T
        task_stats.insert(0, 'Task', task_name)
        stats_file = os.path.join(output_dir, f'{task_name}_score_summary.csv')
        task_stats.to_csv(stats_file, index=False)
        print(f"Saved score summary to: {stats_file}")
        
        # 2. 保存对话统计CSV
        output_csv = os.path.join(output_dir, f'{task_name}_conversation_summary.csv')
        task_convs = conv_df[conv_df['Conversation_ID'].str.startswith(task_name.split('_')[1])]
        task_convs.to_csv(output_csv, index=False)
        print(f"Saved conversation summary to: {output_csv}")
        
        # 3. 保存详细评估TXT
        output_txt = os.path.join(output_dir, f'{task_name}_detailed_evaluations.txt')
        with open(output_txt, 'w', encoding='utf-8') as f:
            # 按对话ID排序输出
            sorted_evals = sorted(all_evaluations, key=lambda x: (x['ID'], x['Turn']))
            for eval_info in sorted_evals:
                if eval_info['Task'] == task_name:  # 只输出当前任务的评估
                    f.write(f"ID: {eval_info['ID']}\n")
                    f.write(f"Turn: {eval_info['Turn']}\n")
                    f.write(f"Question: {eval_info['Question']}\n")
                    f.write(f"Evaluation:\n{eval_info['Evaluation']}\n")
                    f.write("-" * 80 + "\n\n")
        print(f"Saved detailed evaluations to: {output_txt}")
        
        # 4. 生成可视化
        # 箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=task_df, x='Task', y='Score')
        plt.xticks(rotation=45, ha='right')
        plt.title('Score Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task_name}_scores_boxplot.png'))
        plt.close()
        
        # 小提琴图
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=task_df, x='Task', y='Score')
        plt.xticks(rotation=45, ha='right')
        plt.title('Score Distribution (Violin Plot)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task_name}_scores_violin.png'))
        plt.close()
        
        # 轮次分析图
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=task_df, x='Turn', y='Score')
        plt.title('Score Distribution by Conversation Turn')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task_name}_scores_by_turn.png'))
        plt.close()
    
    # 打印统计摘要
    print("\n=== Score Summary ===")
    summary = df.groupby('Task')['Score'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    print(summary)
    
    print("\n=== Conversation Statistics ===")
    print("Conversations by average score:")
    print(conv_df[['Conversation_ID', 'Avg_Score', 'Turn_Count']].to_string())

def main():
    # scores_dir = "eval_result/llava-1.5-7b-hf-sft-5ep/gpt4_scores"
    # output_dir = "eval_result/llava-1.5-7b-hf-sft-5ep/analysis"
    scores_dir = "eval_result/llava-1.5-13b-hf-sft-5ep/gpt4_scores"
    output_dir = "eval_result/llava-1.5-13b-hf-sft-5ep/analysis"
    analyze_scores(scores_dir, output_dir)

if __name__ == "__main__":
    main()