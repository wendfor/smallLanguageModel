import torch
import torch.nn as nn

class Lora(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: int):
    super().__init__()
    self.A = nn.Linear(in_dim, rank, bias=False)
    self.B = nn.Linear(rank, out_dim, bias=False)
    nn.init.normal_(self.A.weight, mean=0.0, std=0.02)   
    nn.init.zeros_(self.B.weight)
    self.alpha = alpha
    self.r = rank
    
  def forward(self, x: torch.tensor):
    return self.alpha/self.r * self.B(self.A(x))
    
def apply_lora(model, rank: int, alpha: int, name_list: list[str]):
  """
    Args:
      name_list: q_proj, k_proj, v_proj, o_proj, up, gate, down
  """
  
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and any(x in name for x in name_list):
      lora = Lora(module.weight.shape[1], module.weight.shape[0], rank, alpha).to(model.device)#非线性层形状不是这样
      setattr(module, "lora", lora)
      module.orig_forward = module.forward
      
      def forward_with_lora(x, f1=module.orig_forward, f2=lora): 
        return f1(x) + f2(x)
      module.forward = forward_with_lora
      

def load_lora(model, path: str):
  state_dict = torch.load(path, map_location=model.device)["state_dict"]
  rank = torch.load(path, map_location=model.device)["config"]["rank"]
  alpha = torch.load(path, map_location=model.device)["config"]["alpha"]
  
  for name, module in model.named_modules():
    if hasattr(module, "lora"):
      lora_state = {k.replace(f"{name}.lora.", "") : v for k, v in state_dict.items() if f"{name}.lora." in k}
      module.lora.load_state_dict(lora_state)
  

def save_lora(model, save_path: str, rank: int, alpha: int):
  orig_model = getattr(model, "_orig_mod", model)#原始模型
  state_dict = {}
  for name, module in orig_model.named_modules():
    if hasattr(module, "lora"):
      mix_name = name[len("module."):] if name.startswith("module.") else name
      lora_state = {f"{mix_name}.lora.{k}" : v.cpu().half() for k, v in module.lora.state_dict().items()}#{A.weight:tensor, B.weight:tensor}
      state_dict.update(lora_state)
  torch.save({"config": {
          "rank": rank,
          "alpha": alpha,
        },"state_dict": state_dict}, save_path)
  
  
def merge_lora(model, lora_path: str, save_path: str):
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
      
  