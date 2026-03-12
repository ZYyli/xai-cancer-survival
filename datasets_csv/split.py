import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

def generate_survival_five_fold_split(cancer_data, split_file, seed=7):
    """
    生成五折交叉验证分割
    
    参数:
    cancer_data: 癌症数据文件
    split_file: 分割结果保存的文件名
    seed: 随机种子
    """ 
    # 提取必要信息
    case_ids = cancer_data['case_id'].values
    labels = cancer_data['label'].values
            
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            
    for fold, (train_idx, val_idx) in enumerate(skf.split(case_ids, labels)):
        train_ids = case_ids[train_idx].tolist()
        val_ids = case_ids[val_idx].tolist()

        split_df = pd.DataFrame({
            'case_id': train_ids + val_ids,
            'fold': [fold] * (len(train_ids) + len(val_ids)),
            'set': ['train'] * len(train_ids) + ['val'] * len(val_ids)
        })
                
        result_file = os.path.join(split_file, f'splits_{fold}.csv')
        split_df.to_csv(result_file, index=False)
                
        print(f"Fold {fold} 分割完成:")
        print(f"训练集大小: {len(train_ids)}, 验证集大小: {len(val_ids)}")
             
    print("五折交叉验证分割生成完成！")