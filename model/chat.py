from model import*
import torch
from prompt_toolkit import prompt
from transformers import AutoTokenizer, TextStreamer
from model.model import *
from train_utils.lora import *

if __name__ == "__main__":
    # 初始化
    model_type = "lora"
    if model_type == "lora":
        path = '/myfile/data/checkpoints/merge_model.pt'  
        model = Model(ModelConfig(2000, 50))
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)#torch.load默认加载到数据保存时的设备，显式指定加载到cpu
    else:
        path = '/myfile/data/checkpoints/sft_model.pt'  
        model = Model(ModelConfig(2000, 50))
        model.load_state_dict(torch.load(path, map_location="cpu")['model_state_dict'], strict=True)#torch.load默认加载到数据保存时的设备，显式指定加载到cpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    enc = AutoTokenizer.from_pretrained("./")

    print("\n" + "="*50)
    print("对话系统已启动（输入 'quit' 退出，'clear' 清空历史）")
    print("="*50)
    streamer = TextStreamer(tokenizer=enc, skip_prompt=True, skip_special_tokens=True)
    while True:
        user_input = prompt("\n你: ")

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        elif user_input.lower() == 'clear':
            continue

        try:
            with torch.inference_mode():
                inputs = enc.apply_chat_template(user_input, add_generate_prompt=True, tokenizer=True, return_dict=True, return_tensors="pt")
                _, response, kv_cache = model.generate(input_ids=inputs["input_ids"], max_new_tokens=1000, streamer=streamer, KV_cache=kv_cache)
                # responese = response.cpu()
                # response = enc.decode(response.tolist(), skip_special_tokens=True)[0]
                # print(f"\n助手: {response}")
        except Exception as e:
                print(f"错误: {e}")