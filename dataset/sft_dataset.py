import torch
from datasets import load_dataset
import torch.utils.data as data
from transformers import AutoTokenizer

def post_processing_chat(prompt_content):
    prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
  
class SFTDataset(data.Dataset):
  def __init__(self, data_path, tokenizer, max_seq_len):
    super().__init__()
    self.ori_data = load_dataset("json", data_files=data_path, split="train")
    self.bos_id = tokenizer.encode(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False)
    self.eos_id = tokenizer.encode(f"{tokenizer.eos_token}\n", add_special_tokens=False)
    self.pad_id = tokenizer.pad_token_id
    self.max_seq_len = max_seq_len
    self.tokenizer = tokenizer
    self.type = type
    
  def __getitem__(self, index: int):
    line = self.ori_data[index]
    input = self.tokenizer.apply_chat_template(line["messages"], tokenize=False)
    input = post_processing_chat(input)
    input_ids = self.tokenizer.encode(input, add_special_tokens=False, max_length=self.max_seq_len+1, truncation=True)
    if len(input_ids) < self.max_seq_len+1:
      input_ids += [self.pad_id] * (self.max_seq_len+1-len(input_ids))
    labels = [-100] * (self.max_seq_len+1)
    for i in range(self.max_seq_len+1):
      if input_ids[i:i+len(self.bos_id)] == self.bos_id:
        i = i+len(self.bos_id)
        while i < self.max_seq_len+1:
          labels[i] = input_ids[i]
          i += 1
        break
    input_ids = torch.tensor(input_ids, dtype=torch.long)  
    labels = torch.tensor(labels, dtype=torch.long)
    labels[input_ids==self.pad_id] = -100  
    return input_ids[:-1], labels[1:]
  
  def __len__(self):
    return len(self.ori_data)
  
enc = AutoTokenizer.from_pretrained("../model")
print(enc.apply_chat_template([{"role":"assistant","content":"你好吗"}], tokenize=False))








