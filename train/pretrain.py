import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
import sys, os
__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.pretrain_dataset import *
from model.model import *
from train_utils.utils import *
import time
import torch.distributed as dist


def train(model, optimizer, scheduler, train_loader, device):
  model.train()
  total_loss = 0
  start_time = time.time()
  for batch_idx, (x, y) in enumerate(train_loader):
    # 将数据移到设备上
    x, y = x.to(device), y.to(device)
        
    # 前向传播，混合精度
    with autocast_ctx:
      logits, loss, _ = model(x, labels=y)
      loss = loss / accumulation_steps
        
    #反向传播得到梯度
    loss.backward()
        
    #梯度累积
    if (batch_idx+1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):  
      #梯度裁切
      torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
      )
      #更新权重
      optimizer.step()
      # 调整学习率
      scheduler.step()
      #清空梯度
      optimizer.zero_grad()
        
    total_loss += loss.item()
        
    if batch_idx % 100 == 0 and is_main_process():
      elapsed_min = (time.time() - start_time) / 60
      print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()*accumulation_steps:.4f}, time: {elapsed_min:.2f}min')
        
    if batch_idx % 1000 == 0 and is_main_process():
      orig_model = model.module if isinstance(model, DistributedDataParallel) else model#提取ddp模型的原始模型， 1是model.save_pretrained这种方法DDP模型没有， 2是从compile包装过的模型提取原始模型。
      orig_model = getattr(orig_model, '_orig_mod', orig_model)#提取compile包装过的原始模型，orig_model必须是原始模型
      checkpoint = {
        'epoch': epoch,
        'model_state_dict': orig_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
      }
      # 保存每个epoch的模型
      torch.save(checkpoint, save_file+"pretrain_model.pt")

  return total_loss

if __name__ == "__main__":
  accumulation_steps: int = 8 #梯度累积步数
  epoch_num: int = 2  
  dtype = torch.bfloat16 #混合精度类型
  data_path: str = "/myfile/data/dataset/pretrain_t2t.jsonl"
  save_file: str = "/myfile/data/checkpoints/"
  num_workers: int = 8 #多线程加载数据
  batch_size: int = 50
  max_seq_len: int = 2048
  
  local_rank = init_distributed_mode()
  if dist.is_initialized(): 
    device = f"cuda:{local_rank}"
  else: 
    device = "cuda"
  setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

  '''
  # split traindataset to train and val
  train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
  train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, pin_memory=True, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  '''

  print("************init model**************")
  model = Model(ModelConfig(batch_size=batch_size, max_seq_len=max_seq_len))
  #device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  tokenizer = AutoTokenizer.from_pretrained("/myfile/data/minimind")

  # 打印模型一共有多少参数
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_params / 1e6} M")
  
  print("**************load data***************")
  train_dataset = PretrainDataset(data_path=data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
  train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
  
  #adamw
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  # 设置 cosine 学习率
  base_lr = 3e-4
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=base_lr * 0.1)
  
  #混合精度训练 
  autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
  
  #ddp model  
  if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
    
  print("**************train start**************")
  for epoch in range(epoch_num):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(42 + epoch); indices = torch.randperm(len(train_dataset)).tolist()
    batch_sampler = SkipBatchSampler(train_sampler or indices, batch_size)
    loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True)
    train_loss = train(model, optimizer, scheduler, loader, device)
    #val_loss = eval(model, val_loader, device)
    #print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

  if dist.is_initialized(): dist.destroy_process_group()
  

