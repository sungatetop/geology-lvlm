import json


def load_file_jsonl(path):
   with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(row) for row in f]

def get_avg(x):
  return sum([float(y) for y in x])/len(x)