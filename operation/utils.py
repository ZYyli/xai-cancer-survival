import torch
import numpy as np
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
  #  """
   # hazards: 模型预测的风险概率
 #   #S: 生存函数（可选，若为 None 会根据 hazards 计算）
 #   Y: 离散时间 bin
 #   c: 事件标记（1 = 死亡 / 事件发生，0 = 删失）
 #   """

  #  batch_size = len(Y)
  #  Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
  #  c = c.view(batch_size, 1).float() #censorship status, 0 or 1
  #  if S is None:
  #      S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
  #  S_padded = torch.cat([torch.ones_like(c), S], 1)
  #  uncensored_loss = - c * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
  #  censored_loss = - (1 - c) * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
  #  neg_l = censored_loss + uncensored_loss
  #  loss = (1-alpha) * neg_l + alpha * uncensored_loss
  #  loss = loss.mean()
  #  return loss

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

def l1_reg_all(module_to_regularize):
    l1_accumulator = None    ## 初始化L1正则化项累加器
    for param in module_to_regularize.parameters():    ## 遍历所有模型参数
        if l1_accumulator is None:
            l1_accumulator = torch.abs(param).sum()
        else:
            l1_accumulator += torch.abs(param).sum()
    
    if l1_accumulator is None:
        return torch.tensor(0.0, device=device)
    return l1_accumulator

def l1_reg_omic(model):
    if hasattr(model, 'fc_omic'):
        return l1_reg_all(model.fc_omic)
    else:
        return l1_reg_all(model)