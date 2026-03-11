import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##########################
#### Genomic FC Model ####
##########################
# 定义一个神经网络模块，包含一个全连接层、一个SELU激活函数和一个AlphaDropout层。
class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim1, dim2),   #一个全连接层（线性层），它将输入维度从 dim1 转换为 dim2
            nn.SELU(),      #激活函数，使神经元的输出在一定条件下自动趋向于零均值和单位方差，从而实现网络的自归一化
            nn.AlphaDropout(p=dropout, inplace=False)     #与SELU配合使用，能保持输入的均值和方差不变，从而不破坏网络的自归一化特性。p=dropout 是丢弃神经元的概率
        )
    def forward(self, x):
        return self.block(x)

#初始化神经网络的权重
def init_max_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear): #检查模块 m 是否是线性层
            stdv = 1. / math.sqrt(m.weight.size(1))  #计算标准差，基于 fan_in (输入神经元数量)
            m.weight.data.normal_(0, stdv)     #从均值为0，计算得到的标准差的正态分布中采样，初始化权重
            if m.bias is not None: # 检查偏置项是否存在
                m.bias.data.zero_() # 将偏置项初始化为0


class SNN_RISK(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super().__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size_omic]

        # fc_omic
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i in range(1, len(hidden)):
            fc_omic.append(SNN_Block(dim1=hidden[i-1], dim2=hidden[i]))
        self.fc_omic = nn.Sequential(*fc_omic)

        # 用于计算 hazard 的线性层
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, x_omic, return_logits: bool = False, return_hazard: bool = False):
        """
        x_omic: [B, num_features]
        return_logits: True -> 返回 classifier logits [B, n_classes]
        return_hazard: True -> 返回 hazards (sigmoid 后的概率) [B, n_classes]
        默认 -> 返回风险分数 risk [B,1]
        """
        h = self.fc_omic(x_omic)
        logits = self.classifier(h)  # [B, n_classes]

        # 校验形状
        if logits.dim() != 2:
            raise ValueError(f"Logits 维度应为2，但实际为 {logits.dim()}，形状：{logits.shape}")

        if return_logits:
            return logits

        hazards = torch.sigmoid(logits)  # [B, n_classes]
        if return_hazard:
            return hazards

        # 计算生存函数并得到风险分数
        survival = torch.cumprod(1 - hazards, dim=1)       #torch.cumprod(1 - hazards, dim=1)  # 累积生存概率
        risk = -torch.sum(survival, dim=1, keepdim=True)
        return risk
        #logits = self.classifier(h_omic)  # shape [B, n_classes]

        # 转换成风险分数
        #hazards = torch.sigmoid(logits)            # [B, n_classes]
        #log_survival = torch.sum(torch.log(1 - hazards + 1e-7), dim=1, keepdim=True)  # [B, 1]
        #risk = -log_survival                      # [B, 1]