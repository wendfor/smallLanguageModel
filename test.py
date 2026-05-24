import numpy as np
import torch
from train_utils.lora import *
import math
import torch
import torch.nn as  nn


class simpleDecoderLayer(nn.Module):
  def __init__(self, hidden_dim, head_num, dropout_rate=0.1):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.head_num = head_num
    
    self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    # self.atten_drop = nn.Dropout(dropout_rate)
    self.atten_ln = nn.LayerNorm(hidden_dim, eps=0.000001)

    self.ffn_layer1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
    self.ffn_layer2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
    self.act = nn.GELU()
    # self.ffn_drop = nn.Dropout(dropout_rate)
    self.ffn_ln = nn.LayerNorm(hidden_dim, eps=0.000001)

  def mha(self, x, mask=None):
    batch_size, seq_len,_ = x.shape
    head_dim = self.hidden_dim // self.head_num

    Q = self.q_proj(x).view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
    K = self.k_proj(x).view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)
    V = self.v_proj(x).view(batch_size, seq_len, self.head_num, -1).transpose(1, 2)

    attention_weight = torch.matmul(
      Q, K.transpose(-1, -2)
    )
    attention_weight = attention_weight / math.sqrt(head_dim)
    if mask is not None:
      mask = mask.tril()
    else:
      mask = torch.ones_like(attention_weight)
      mask = mask.tril()
    attention_weight = attention_weight.masked_fill(
      mask==0, float('-inf')
    )
    attention_weight = torch.softmax(
      attention_weight, dim=-1
    )
    # attention_weight = self.atten_drop(attention_weight)
    
    attention_res = torch.matmul(
      attention_weight, V
    )

    attention_res = attention_res.transpose(1, 2).contiguous()
    attention_res = attention_res.view(batch_size, seq_len, self.hidden_dim)
    attention_res = self.out_proj(attention_res)

    output = self.atten_ln(x + attention_res)
    return output
  

  def ffn(self, x):
    x1 = self.ffn_layer1(x)
    x1 = self.act(x1)
    x2 = self.ffn_layer2(x1)
    # x2 = self.ffn_drop(x2)

    output = self.ffn_ln(x+x2)
    return output 

  def forward(self, x):
    x = self.mha(x)
    x = self.ffn(x)

    return x
  

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.ModuleList(
      [simpleDecoderLayer(12, 1) for i in range(1)]
    )
    self.embedding = nn.Embedding(12, 12)
    self.out_proj = nn.Linear(12, 12, bias=False)
    self.apply(self._init_weights)
    
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.zeros_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.zeros_(module.weight)
    
    
  def forward(self, x):
    #[b, s, 1/2/4]->[b, s, h]
    x = self.embedding(x)
    print(x)
    #[b,s,h]->[b,s,h]
    for layer in self.layers:
      x = layer(x)
    #[b,s,h]->[b,s,v]
    x = self.out_proj(x)
    x = torch.softmax(x, dim=-1)
    return x

decode = Decoder()
load_lora(decode, "./")
for name, module in decode.named_modules():
  l = ["q_proj", "k_proj", "out_proj"]
  if isinstance(module, nn.Linear):
    if any(x in name for x in l):
      print(name)
      print(module.state_dict().items())

lora_config = LoraConfig(4, 2, ["q_proj", "k_proj"])

# apply_lora(decode, lora_config, "cpu")

# save_lora(decode, "./", lora_config)
# merge_lora(decode, "./")

           

  
 

