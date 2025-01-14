import os
import base64
import json
from openai import OpenAI
from transformers.utils.versions import require_version
def readImageBase64(image_path):
    with open(image_path,"rb") as f:
        # 读取图片数据
        image_data = f.read()
        # 将图片数据编码为base64字符串
        return base64.b64encode(image_data).decode()

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")
image_root="./eval_origin/"

def query_llm(conversations:dict,save_path="./prediction.jsonl"):
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
    messages = []
    convs=conversations["messages"]
    images=conversations.get("images",[])
    id=conversations["id"]
    result={"id":id,"conversations":[]}
    print(id)
    for idx in range(len(convs)//2):
        user_item=convs[idx*2]

        user_input=user_item["content"]
        message={
                "role":"user",
                "content":[
                    {"type":"text","text":user_input}
                ]
            }
        if "<image>" in user_input:
            for imp in images:
                image=readImageBase64(image_root+imp)
                message["content"].append({"type":"image_url","image_url":{
                        "url":"data:image/png;base64,"+image
                }})
        else:
            images=[]
        #user
        messages.append(message)
        assitant_item=convs[idx*2+1]
        label=assitant_item["content"]
        response = client.chat.completions.create(messages=messages, model="test")
        resp=response.choices[0].message.content
        #assistant
        messages.append({
            "role":"assistant",
            "content":[
                {"type":"text","text":resp}
            ]
        })
        result["conversations"].append({"images":images,"user_input":user_input,"prediction":resp,"label":label})
    with open(save_path,'a+',encoding="utf-8") as f:
        f.write(json.dumps(result,ensure_ascii=False)+"\n")


def main():
    import glob
    from tqdm import tqdm
    save_root='./qwen2-vl-5ep/'
    evalfiles=glob.glob("eval_origin/eval_*.json",recursive=True)
    print(evalfiles)

    for file in tqdm(evalfiles):
        with open(file,'r',encoding="utf-8") as f:
            data=json.load(f)
        filename=os.path.basename(file)
        if os.path.exists(filename+'l'):
            continue
        for d in tqdm(data):
            query_llm(d,save_root+filename+"l")


if __name__ == "__main__":
    main()
