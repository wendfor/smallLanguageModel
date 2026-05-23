import torch
import torch.nn as nn
import json

class LoraConfig:
  def __init__(self, alpha: int, r: int, target_modules: list[str]):
    alpha: int = alpha,
    r: int = r,
    target_modules: list[str] = target_modules

#增加lora模块
def apply_lora(model, lora_config):
  scaling = lora_config.alpha / lora_config.r
  r = lora_config.r
  alpha = lora_config.alpha
  target_modules = lora_config.target_modules
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and any(x in name for x in target_modules):
      lora = Lora(module.weight.shape[1], module.weight.shape[0], r, module.weight.dtype).to(model.device)#非线性层形状不是这样
      #冻结权重
      module.weight.requires_grad = False
      if module.bias is not None: module.bias.requires_grad = False
      #绑定lora
      setattr(module, "lora", lora)
      orig_forward = module.forward
      
      def forward_with_lora(x, f1=orig_forward, f2=lora): 
        return f1(x) + scaling * f2(x)
      module.forward = forward_with_lora


class Lora(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, rank: int, dtype = torch.bfloat16):
    super().__init__()
    self.A = nn.Linear(in_dim, rank, bias=False, dtype=dtype)
    self.B = nn.Linear(rank, out_dim, bias=False, dtype=dtype)
    nn.init.normal_(self.A.weight, mean=0.0, std=0.02)   
    nn.init.zeros_(self.B.weight)

    
  def forward(self, x: torch.tensor):
    return self.B(self.A(x))
    

def load_lora(model, path: str):
  state_dict = torch.load(path+"lora_model.pt", map_location=model.device)["state_dict"]
  for name, module in model.named_modules():
    if hasattr(module, "lora"):
      lora_state = {k.replace(f"{name}.lora.", "") : v for k, v in state_dict.items() if f"{name}.lora." in k}
      module.lora.load_state_dict(lora_state)
  

def save_lora(model, save_path: str, lora_config: LoraConfig):
  orig_model = getattr(model, "_orig_mod", model)#原始模型
  state_dict = {}
  for name, module in orig_model.named_modules():
    if hasattr(module, "lora"):
      mix_name = name[len("module."):] if name.startswith("module.") else name
      lora_state = {f"{mix_name}.lora.{k}" : v.cpu().half() for k, v in module.lora.state_dict().items()}#{A.weight:tensor, B.weight:tensor}
      state_dict.update(lora_state)
  with open(save_path+"lora_config.json", 'w') as f:
    json.dump({"alpha":lora_config.alpha, "rank": lora_config.r, "target_modules": lora_config.target_modules}, f)
  torch.save({"state_dict": state_dict}, save_path+"lora_model.pt")
  
  
def merge_lora(model, lora_path: str, save_path: str):
  with open(lora_path+"lora_config.json", 'r') as f:
    config = json.load(f)
  r = config["rank"]
  alpha = config["alpha"]
  target_modules = config["target_modules"]
  
  load_lora(model, lora_path)
  orig_model = getattr(model, "_orig_mod", model)
  state_dict = {k : v.cpu().half() for k, v in orig_model.state_dict().items() if ".lora." not in k}
  for name, module in orig_model.named_modules():
    if isinstance(module, nn.Linear) and hasattr(module, "lora"):
      state_dict[f"{name}.weight"] =(module.weight.detach().cpu().half() + (module.lora.B.weight @ module.lora.A.weight).detach().cpu().half())
      module.forward = module.orig_forward
      delattr(module, "orig_forward")
      delattr(module, "lora")
      
  torch.save(state_dict, save_path)
      
  