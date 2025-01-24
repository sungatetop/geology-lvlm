from util import load_file_jsonl, get_avg
import os

data=load_file_jsonl("tests\llava-1.5-7b-hf-sft-5ep\eval_tunnel_knowledge_scores.jsonl")
total=0
for item in data:
    score=item['gpt_eval'].split('\n')[0]
    score=float(score)
    total+=score
    print(item['id'],item['gpt_eval'],score)
print("total:",total,"100:",10*total/len(data))
