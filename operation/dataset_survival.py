import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class RNAseqSurvivalDataset(Dataset):
    def __init__(self, data, label_col='survival_months', print_info=True, seed=None):
        self.seed = seed
        # 设置随机数生成器的种子
        self._set_seed()
        df = data.copy()
        # 确保 case_id 列存在
        if 'case_id' not in df.columns:
            raise ValueError("CSV文件中缺少 'case_id' 列！")
        
        # 提取 case_id 作为样本标识符
        self.case_ids = df['case_id'].values
        rnaseq_cols = [col for col in df.columns if '_rnaseq' in col]
        self.features = df[rnaseq_cols].copy()
        self.labels = df['disc_label'].values
        self.times = df[label_col].values
        self.censorship = df['censorship'].values

    def _set_seed(self):
        """设置随机数生成器的种子"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features.iloc[idx].values, dtype=torch.float)
        label = torch.tensor([self.labels[idx]], dtype=torch.float)
        time = torch.tensor([self.times[idx]], dtype=torch.float)
        c = torch.tensor([self.censorship[idx]], dtype=torch.float)
        return x, label, time, c