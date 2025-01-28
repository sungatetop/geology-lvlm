from util import load_file_jsonl, get_avg
import os

root='tests/gpt4v/'
files=os.listdir(root)
for file in files:
    data=load_file_jsonl(f"{root}{file}")
    total=0
    for item in data:
        score=item['gpt_eval'].split('\n')[0]
        score=float(score)
        total+=score
        #print(item['id'],item['gpt_eval'],score)
    print(file,"total:",total,"100:",10*total/len(data))
