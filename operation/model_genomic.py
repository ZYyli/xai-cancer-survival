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


class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()

        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}

        hidden = self.size_dict_omic[model_size_omic]
        ## 4 个 SNN_Block 模块被放入 fc_omic 列表
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        ##将4个模块放入一个Sequential容器中，相当于forward函数  
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        #在模型层构建完毕后，调用前面定义的权重初始化函数来初始化模型的所有线性层权重
        init_max_weights(self)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        h_omic = self.fc_omic(x)
        h  = self.classifier(h_omic) # logits needs to be a [B x 4] vector
        assert len(h.shape) == 2 and h.shape[1] == self.n_classes
        return h