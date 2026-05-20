import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dataclasses


@dataclasses
class ModelConfig:
  def __init__(self, max_seq_len, batch_size):
    self.max_seq_len: int = max_seq_len #文本的最大长度
    self.batch_size: int = batch_size
    self.n_layer: int = 10
    self.n_head: int = 8
    self.n_embd: int = 768
    self.hidden_dim: int = self.n_embd
    self.dropout: float = 0.1
    self.rms_eps: float = 1e-6
    self.n_kv_head: int = 6
    self.head_dim: int = self.n_embd // self.n_head
    self.vocab_size: int = 6400
    self.flash_attn: bool = True


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones([dim]))

  def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

  def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
      
def freqs_get(head_dim, seq_len, theta: float = 1e6):
  freqs = torch.arange(0, head_dim, 2)[:(head_dim//2)].float()#对head_dim为奇数的处理
  freqs = 1.0 / (theta ** (freqs/head_dim))
  m = torch.arange(seq_len, device=freqs.device)
  freqs = torch.outer(m, freqs).float()
  freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
  freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
  return freqs_cos, freqs_sin

def rotary_half(x):
  return torch.cat([-x[...,x.shape[-1]//2:], x[...,:x.shape[-1]//2]], dim=-1)


def apply_rotary(q, k, cos, sin):
  q_embd = ((q * cos.unsqueeze(1)) + (rotary_half(q) * sin.unsqueeze(1))).to(q.dtype)
  k_embd = ((k * cos.unsqueeze(1)) + (rotary_half(k) * sin.unsqueeze(1))).to(k.dtype)
  return q_embd, k_embd


class MultiheadAttention(nn.Module):
  """
    带 RoPE 的多头注意力
  """
  def __init__(self, config: ModelConfig):
    super().__init__()

    self.n_embd = config.n_embd
    self.num_heads = config.n_head
    self.head_dim = config.head_dim

    self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    self.q_norm = RMSNorm(self.head_dim, config.rms_eps)
    self.k_norm = RMSNorm(self.head_dim, config.rms_eps)
    self.dropout = nn.Dropout(config.dropout)
    self.dropout_p = config.dropout
    #flash attn
    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

  def forward(self, x: torch.Tensor, rope_states: torch.tensor, use_cache: bool = False, kv_cache: torch.tensor = None, attn_mask: torch.Tensor = None):
    """
      Args:
          x: 输入，shape (batch, seq_len, config.n_embd)
          atten_mask: 注意力掩码，shape (b, s)
          use_cache: kv cache是否使用
          kv_cache: 缓存
    """
    batch, seq_len, _ = x.shape

    # 线性投影
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # 重塑为多头形式
    q = q.view(batch, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch, seq_len, self.num_heads, self.head_dim)
    v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        
    #normalize
    q = self.q_norm(q)
    k = self.k_norm(k)
        
    # 应用 RoPE（只对 Q 和 K）
    cos, sin = rope_states
    q, k = apply_rotary(q, k, cos, sin)

    # 转置用于矩阵乘法：(batch, num_heads, seq_len, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
        
    if kv_cache is not None:
      k = torch.cat([kv_cache[0], k], dim=2)
      v = torch.cat([kv_cache[1], v], dim=2)

    kv_cache = (k, v) if use_cache else None
    if self.flash and (seq_len > 1) and (attn_mask is None or torch.all(attn_mask == 1)):
      attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
    else:
      attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
      #d = 0 if kv_cache is None else kv_cache[0].shape[2]
      attn_scores[...,-seq_len:] += torch.full([seq_len, seq_len], float("-inf"), device=attn_scores.device).triu(1)#seq_len=1时相当于不操作
            
      if attn_mask is not None:#attn_mask [b, s]
        attn_scores += (1.0 - attn_mask.unsqueeze(1).unsqueeze(2)) * -1e9

      attn_probs = torch.softmax(attn_scores, dim=-1)
      attn_probs = self.dropout(attn_probs)
      # 向量加权求和
      attn_output = torch.matmul(attn_probs, v)
        
    # 重塑并输出投影
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch, seq_len, self.n_embd)
    output = self.o_proj(attn_output)

    return output, kv_cache

  
class FFN(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    mid_dim = (config.hidden_dim*8 + 2)//3
    self.up = nn.Linear(config.hidden_dim, mid_dim, bias=False)
    self.gate = nn.Linear(config.hidden_dim, mid_dim, bias=False)
    self.down = nn.Linear(mid_dim, config.hidden_dim, bias=False)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x: torch.tensor):
    w1 = F.silu(self.gate(x))
    w2 = self.up(x)
    gate = w1*w2
    w3 = self.down(gate)
    out = self.dropout(w3)
    return out
  
class Block(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.mha = MultiheadAttention(config)
    self.ffn = FFN(config)
    self.rn1 = RMSNorm(config.hidden_dim, config.rms_eps)
    self.rn2 = RMSNorm(config.hidden_dim, config.rms_eps)
    #self.ln1 = nn.LayerNorm(config.n_embd)
    #self.ln2 = nn.LayerNorm(config.n_embd)
  
  def forward(self, x: torch.tensor, rope_states: torch.tensor, use_cache: bool = False, kv_cache: torch.tensor = None, attn_mask: torch.tensor = None):
    # 前归一化
    hidden_states, kv_cache = self.mha(self.rn1(x), rope_states, use_cache=use_cache, kv_cache=kv_cache, attn_mask=attn_mask)
    hidden_states = x + hidden_states
    output = hidden_states + self.ffn(self.rn2(hidden_states))
    return output, kv_cache
  
class Model(nn.Module):
  def __init__(self, config: ModelConfig):
    super().__init__()
    self.block_size = config.block_size
    self.eos_token_id = 2
    self.KV_cache = None
    self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
    #self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.ModuleList(
      [Block(config) for _ in range(config.n_layer)]
    )
    self.final_rn = RMSNorm(config.hidden_dim, config.rms_eps)
    #self.final_ln = nn.LayerNorm(config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    #tie weight来减少参数 // linear layer weight实际形状是倒的
    self.token_embedding_table.weight = self.lm_head.weight
    #rope buffer
    cos, sin = freqs_get(config.head_dim, config.block_size)
    self.register_buffer("freqs_cos", cos, persistent=False)
    self.register_buffer("freqs_sin", sin, persistent=False)
    #init weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, input_ids: torch.tensor, labels: torch.tensor = None, use_cache: bool = False, KV_cache: torch.tensor = None, attn_mask: torch.tensor = None):
    """
      Args:
          input_ids 是输入token_ids [b， s]
          labels 是输出token_ids
          shape要一样
    """
    batch_size, seq_len = input_ids.shape
    hidden_states = self.token_embedding_table(input_ids)
    #存储在同样的设备
    #position_embd = self.position_embedding_table(
    #    torch.arange(seq_len, device=input_ids.device)
    #)
    #embd = token_embd + position_embd
    #hidden_states = self.blocks(embd)
    prev = 0
    if KV_cache is not None:
        prev = KV_cache[0][0].shape[2]#0层k_cache的seq_len
    rope_states = (self.freqs_cos[prev:prev+seq_len], self.freqs_sin[prev:prev+seq_len])
    KV_cache = KV_cache or [None] * len(self.blocks)
    present = []
    for block, kv_cache in zip(self.blocks, KV_cache):
        hidden_states, new_kv_cache = block(hidden_states, rope_states, use_cache=use_cache, kv_cache=kv_cache, attn_mask=attn_mask)
        present.append(new_kv_cache)
    hidden_states = self.final_rn(hidden_states)
    logits = self.lm_head(hidden_states)

    if labels is None:
      loss = None
    else:
      _, _, vocab_size = logits.shape
      logits = logits.view(batch_size * seq_len, vocab_size)
      labels = labels.view(batch_size * seq_len)
      loss = F.cross_entropy(logits, labels, ignore_index=-100)

    return logits, loss, present
  
  
  def generate(self, input_ids: torch.tensor, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.9, top_k: int = 50):
    KV_cache = None
    ids = input_ids
    gen_ids = torch.zeros((input_ids.size(0), 0), dtype=torch.long, device=input_ids.device)
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
      # 如果序列太长，只取最后 block_size 个token
      if input_ids.size(1) > self.block_size and KV_cache is not None:
        KV_cache = [(k[:,:,-self.block_size:,:], v[:,:,-self.block_size:,:])
                for k, v in KV_cache]
      # 前向计算
      logits, loss, KV_cache = self(ids, use_cache=True, KV_cache=KV_cache)
      # 采样策略
      logits = logits[:, -1, :] / temperature  # becomes (B, vocab_size)
            
      #对生成eos的prompt继续生成eos
      logits[finished, :] = float("-inf")
      logits[finished, self.eos_token_id] = 0
      #取概率最大的前k个
      if top_k > 0: 
        logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
      #取概率累积为p的前几个      
      if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
        logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
      # 应用softmax获取概率
      probs = F.softmax(logits, dim=-1)
      # 采样下一个token
      ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
      finished = finished | (ids.squeeze(-1) == self.eos_token_id)
      # 附加到序列上
      input_ids = torch.cat([input_ids, ids], dim=1)  # (B, T+1)
      gen_ids = torch.cat([gen_ids, ids], dim=1)
    
    return input_ids, gen_ids



  



