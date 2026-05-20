import torch
from datasets import load_dataset
import torch.utils.data as data


class PretrainDataset(data.Dataset):
  def __init__(self, data_path, tokenizer, max_seq_len):
    super().__init__()
    self.ori_data = load_dataset("json", data_files=data_path, split="train")
    self.tokenizer = tokenizer
    self.eos_id = tokenizer.eos_token_id
    self.bos_id = tokenizer.bos_token_id
    self.pad_id = tokenizer.pad_token_id
    self.max_seq_len = max_seq_len
    
  def __getitem__(self, index: int):
    self.line = self.ori_data[index]
    input_ids = self.tokenizer.encode(self.line["text"], add_special_tokens=False, trunction=True, max_length=self.max_seq_len-1)
    input_ids = [self.bos_id] + input_ids + [self.eos_id]
    if len(input_ids) < self.max_seq_len+1:
      input_ids = input_ids + [self.pad_id] * (self.max_seq_len+1-len(input_ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids==self.pad_id] = -100
    return input_ids[:-1], labels[1:]
  
  def __len__(self):
    return len(self.ori_data)







