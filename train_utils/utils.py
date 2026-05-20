import os
import torch.distributed as dist
from torch.utils.data import Sampler
import torch
import random
import numpy as np

def is_main_process():
  return not dist.is_initialized() or dist.get_rank() == 0

def init_distributed_mode():
  if int(os.environ.get("RANK", -1)) == -1:
    return 0  # 非DDP模式

  dist.init_process_group(backend="nccl")
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  return local_rank

def setup_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


class SkipBatchSampler(Sampler):
  def __init__(self, sampler, batch_size, skip_batches=0):
    self.sampler = sampler
    self.batch_size = batch_size
    self.skip_batches = skip_batches

  def __iter__(self):
    batch = []
    skipped = 0
    for idx in self.sampler:
      batch.append(idx)
      if len(batch) == self.batch_size:
        if skipped < self.skip_batches:
          skipped += 1
          batch = []
          continue
        yield batch
        batch = []
      if len(batch) > 0 and skipped >= self.skip_batches:
        yield batch

  def __len__(self):
    total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
    return max(0, total_batches - self.skip_batches)
