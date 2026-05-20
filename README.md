# smallLanguageModel
参考minimind实现一个轻量级的模型，以及预训练，微调流程。

## Repository Layout
'''text
dataset/
  pretrain_dataset.py   预训练数据dataset类
  sft_dataset.py        sft数据dataset类
model/  
  model.py               定义模型
  chat.py                运行模型
train/
  pretrain.py           预训练过程 
  sft.py                微调过程
train_utils/
  utils.py                工具
'''