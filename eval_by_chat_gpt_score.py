'''
 参考llava-med

'''
import os
import json
import asyncio
import backoff
import argparse
from copy import deepcopy
from collections import defaultdict
import pandas as pd
import openai
import util

# 评估提示语
INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks questions about tunnel construction, rock characteristics, and geological conditions based on observing an image.

Please rate the helpfulness, relevance, accuracy, and level of detail in their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

ROLE = 'Assistant'

class GPTEvaluator:
    """GPT-4评估器类"""
    def __init__(self, api_key, base_url):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    def create_messages(self, question, ans1, ans2):
        """生成评估消息"""
        content = (f'[Question]\n{question}\n\n'
                  f'[{ROLE} 1]\n{ans1}\n\n[End of {ROLE} 1]\n\n'
                  f'[{ROLE} 2]\n{ans2}\n\n[End of {ROLE} 2]\n\n'
                  f'[System]\n{INSTRUCT_PROMPT}\n\n')
        print(content)
        return [
            {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
            {"role": "user", "content": content}
        ]

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    async def evaluate_single(self, question, ans1, ans2):
        """评估单个样本"""
        messages = self.create_messages(question, ans1, ans2)
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return None

    async def evaluate_batch(self, samples, batch_size=5):
        """批量评估样本"""
        results = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            tasks = [
                self.evaluate_single(
                    sample['question'],
                    sample['llava_answer'],
                    sample['label']
                )
                for sample in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            
            for sample, result in zip(batch, batch_results):
                if result:
                    sample['gpt_eval'] = result
                    print(result)
                    results.append(sample)
                    
        return results

def load_theme_data(file_path):
    """加载单个主题文件的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def process_conversations(conv_data,theme):
    """处理对话数据,提取问题和回答"""
    conversations=conv_data['conversations']
    
    data=[]
    id=conv_data['id']
    for conv in conversations:
        question = conv['user_input'].replace('<image>', '')
        sample={}
        sample['id']=id
        sample['theme']=theme
        sample['images']=conv['images']
        sample['question']=question
        sample['llava_answer']=conv['prediction']
        sample['label']=conv['label']
        data.append(sample)
    return id,data

async def main(args):
    # 初始化GPT-4评估器
    evaluator = GPTEvaluator(
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 加载所有主题的LLaVA回答
    inferresult = {}
    for file_name in os.listdir(args.llava_dir):
        if file_name.endswith('.jsonl'):
            theme = file_name.split('.')[0]
            file_path = os.path.join(args.llava_dir, file_name)
            inferresult[theme] = load_theme_data(file_path)
            
        
    # 准备评估样本
    samples = []

    for theme in inferresult:            
        for item in inferresult[theme]:
            id,data=process_conversations(item,theme)
            samples.extend(data)                
    # 使用GPT-4进行评估
    results = await evaluator.evaluate_batch(samples)
    
    # 保存结果
    with open(args.scores_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
    # 统计不同主题的得分
    score_theme_dict = defaultdict(lambda: defaultdict(list))
    for x in results:
        theme = x['theme']
        scores = x['gpt_eval'].split('\n')[0].split(' ')
        if len(scores) >= 2:
            llava_score = float(scores[0])
            gpt_score = float(scores[1])
            score_theme_dict[theme][1].append(llava_score)  # LLaVA scores
            score_theme_dict[theme][2].append(gpt_score)    # GPT scores
            score_theme_dict['overall'][1].append(llava_score)
            score_theme_dict['overall'][2].append(gpt_score)
    
    # 计算各项统计指标
    result = defaultdict(dict)
    for theme, score_dict in score_theme_dict.items():
        result[theme]['llava_score'] = util.get_avg(score_dict[1])
        result[theme]['gpt_score'] = util.get_avg(score_dict[2])
        result[theme]['relative_score'] = util.get_avg([float(s2)/float(s1) for s1, s2 in zip(score_dict[1], score_dict[2])])*100
        result[theme]['data_size'] = len(score_dict[1])
        
    # 输出结果
    df = pd.DataFrame.from_dict(result).T
    print("\nModel Performance Comparison by Theme:")
    print(df)
    
    # 输出每个主题的样本数量
    print("\nSample counts by theme:")
    for theme in result:
        print(f"{theme}: {result[theme]['data_size']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("LLaVA vs GPT Model Comparison", add_help=True)
    parser.add_argument("--llava-dir", type=str, required=True, help="directory containing LLaVA answer files")
    parser.add_argument("--scores-file", default="scores.jsonl", help="path to save score file")
    parser.add_argument("--api-key", required=False,default="", help="OpenAI API key")
    parser.add_argument("--base_url", required=False,default="https://api.deepseek.com", help="base_url")
    args = parser.parse_args()
    
    asyncio.run(main(args))