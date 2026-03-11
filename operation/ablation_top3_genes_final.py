"""
Top-3 基因独立消融实验 (LGG) - 自动化多XAI版
1. 自动根据 XAI 结果目录统计并选出 Consensus Top-3 基因
2. 循环验证多种 XAI 方法
3. 完全复刻原始训练参数和流程
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
import warnings
import math

warnings.filterwarnings('ignore')

# ================= 用户配置区 =================
CANCER = 'LGG'
OUTPUT_DIR = '/home/zuoyiyi/SNN/TCGA/ablation_study_LGG_final'

# 您可以在这里添加您想要验证的 XAI 方法
# 脚本会自动去对应的目录找 ranking 文件，选出 Top-3 进行消融
XAI_METHODS_TO_TEST = ['DeepLIFT', 'IG', 'G-SHAP', 'LRP', 'PFI', 'D-SHAP']

# XAI 方法对应的目录配置 (根据您的 individual_analysis.py 输出设定)
XAI_CONFIG = {
    'DeepLIFT': {
        'base_dir': '/home/zuoyiyi/SNN/TCGA/DeepLIFT_results_2',
        'subdir': 'deeplift_feature_importance',
        'file_pattern': '*deeplift_feature_importance_ranking.csv'
    },
    'IG': {
        'base_dir': '/home/zuoyiyi/SNN/TCGA/IG_results_2',
        'subdir': 'ig_feature_importance',
        'file_pattern': '*ig_feature_importance_ranking.csv'
    },
    'G-SHAP': {
        'base_dir': '/home/zuoyiyi/SNN/TCGA/shap_results_2',
        'subdir': 'shap_feature_importance',
        'file_pattern': '*shap_feature_importance_ranking.csv'
    },
    'LRP': {
        'base_dir': '/home/zuoyiyi/SNN/TCGA/LRP_results_2',
        'subdir': 'lrp_feature_importance',
        'file_pattern': '*lrp_feature_importance_ranking.csv'
    },
    'PFI': {
        'base_dir': '/home/zuoyiyi/SNN/TCGA/PFI_results_2',
        'subdir': 'pfi_feature_importance',
        'file_pattern': '*pfi_feature_importance_ranking.csv'
    },
    'D-SHAP': {
        'base_dir': '/home/zuoyiyi/SNN/TCGA/deepshap_results_2',
        'subdir': 'deepshap_feature_importance',
        'file_pattern': '*deepshap_feature_importance_ranking.csv'
    }
}

# 训练参数 (复刻 main.py)
BATCH_SIZE = 32
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-5
L1_REG = 1e-5
INNER_VAL_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实验设置 (Nested CV)
N_REPEATS = 10
N_FOLDS = 5
SEED = 1

# ================= 自动选择 Top 基因逻辑 =================
def get_top_consensus_genes(cancer, method_name, k=3):
    """
    自动读取指定 XAI 方法的 ranking 文件，选出 Top-K 共识基因
    策略：优先选择 Top-10 出现频率高的；频率相同时，选择 Rank 1 次数多的。
    """
    if method_name not in XAI_CONFIG:
        print(f"❌ 未知的方法: {method_name}，请在 XAI_CONFIG 中配置")
        return []

    config = XAI_CONFIG[method_name]
    ranking_dir = os.path.join(config['base_dir'], cancer, config['subdir'])
    
    if not os.path.exists(ranking_dir):
        print(f"❌ 目录不存在: {ranking_dir}")
        return []
    
    search_pattern = os.path.join(ranking_dir, config['file_pattern'])
    files = glob.glob(search_pattern)
    
    if not files:
        # 尝试放宽搜索条件，比如不带 _feature_importance
        search_pattern_alt = os.path.join(ranking_dir, '*ranking.csv')
        files = glob.glob(search_pattern_alt)
        if not files:
            print(f"❌ 在 {ranking_dir} 未找到 ranking 文件")
            return []

    print(f"🔍 [自动选特征] 正在分析 {len(files)} 个 {method_name} 的 ranking 文件...")

    gene_stats = {} # gene -> {'top10_hits': 0, 'rank1_hits': 0, 'sum_rank': 0}

    for f in files:
        try:
            df = pd.read_csv(f)
            if 'feature_name' not in df.columns:
                continue
            
            # 获取前10名
            top10_df = df.head(10)
            
            for idx, row in top10_df.iterrows():
                gene = row['feature_name']
                rank = idx + 1
                
                if gene not in gene_stats:
                    gene_stats[gene] = {'top10_hits': 0, 'rank1_hits': 0, 'sum_rank': 0}
                
                gene_stats[gene]['top10_hits'] += 1
                gene_stats[gene]['sum_rank'] += rank
                
                if rank == 1:
                    gene_stats[gene]['rank1_hits'] += 1
                    
        except Exception as e:
            print(f"⚠️ 读取 {os.path.basename(f)} 失败: {e}")

    # 排序策略：
    # 1. Top 10 出现次数 (稳定性)
    # 2. Rank 1 出现次数 (爆发力)
    # 3. 平均排名 (Sum Rank 越小越好)
    sorted_genes = sorted(
        gene_stats.items(), 
        key=lambda x: (
            -x[1]['top10_hits'],    # 降序
            -x[1]['rank1_hits'],    # 降序
            x[1]['sum_rank']        # 升序
        )
    )
    
    top_k_genes = [g[0] for g in sorted_genes[:k]]
    
    print(f"🏆 {method_name} 的 Top-{k} 共识基因:")
    for i, gene in enumerate(top_k_genes):
        stats = gene_stats[gene]
        print(f"   {i+1}. {gene:<15} (Top10频率: {stats['top10_hits']}/{len(files)}, Rank1次数: {stats['rank1_hits']})")
        
    return top_k_genes

# ================= 模型定义 (SNN_RISK) =================
class SNN_Block(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.SELU(),
            nn.AlphaDropout(p=dropout, inplace=False)
        )
    def forward(self, x):
        return self.block(x)

def init_max_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            if m.bias is not None:
                m.bias.data.zero_()

class SNN_RISK(nn.Module):
    def __init__(self, omic_input_dim, model_size_omic='small', n_classes=4):
        super().__init__()
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size_omic]

        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i in range(1, len(hidden)):
            fc_omic.append(SNN_Block(dim1=hidden[i-1], dim2=hidden[i]))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, x_omic):
        h = self.fc_omic(x_omic)
        logits = self.classifier(h)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1, keepdim=True)
        return risk

# ================= 数据集 =================
class RNAseqSurvivalDataset(Dataset):
    def __init__(self, data_df, feature_cols):
        self.data = data_df
        self.survival_months = data_df['survival_months'].values
        self.censorship = data_df['censorship'].values
        self.event = 1 - self.censorship
        self.features = data_df[feature_cols].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float32),
            't': torch.tensor(self.survival_months[idx], dtype=torch.float32),
            'e': torch.tensor(self.event[idx], dtype=torch.float32)
        }

# ================= 训练相关函数 =================
def cox_loss(risk_pred, y_time, y_event):
    val, idx = torch.sort(y_time, descending=True)
    risk_pred = risk_pred[idx]
    y_event = y_event[idx]
    exp_risk = torch.exp(risk_pred)
    log_cumsum = torch.log(torch.cumsum(exp_risk, dim=0))
    loss = -torch.sum((risk_pred - log_cumsum) * y_event.unsqueeze(1)) / (torch.sum(y_event) + 1e-8)
    return loss

def regularize_weights(model):
    l1_reg = torch.tensor(0., requires_grad=True).to(DEVICE)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_reg = l1_reg + torch.norm(param, 1)
    return l1_reg

def train_fold(train_df, test_df, feature_cols, seed):
    # 1. 内部划分验证集
    inner_train_df, inner_val_df = train_test_split(
        train_df, 
        test_size=INNER_VAL_RATIO,
        stratify=train_df['censorship'],
        random_state=seed
    )
    
    # 2. 构建Dataset/DataLoader
    train_ds = RNAseqSurvivalDataset(inner_train_df, feature_cols)
    val_ds = RNAseqSurvivalDataset(inner_val_df, feature_cols)
    test_ds = RNAseqSurvivalDataset(test_df, feature_cols)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. 初始化模型
    input_dim = len(feature_cols)
    model = SNN_RISK(omic_input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_cindex = 0
    patience = 7 
    counter = 0
    best_state = None
    
    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x = batch['x'].to(DEVICE)
            t = batch['t'].to(DEVICE)
            e = batch['e'].to(DEVICE)
            
            optimizer.zero_grad()
            risk = model(x)
            loss = cox_loss(risk, t, e)
            
            if L1_REG > 0:
                loss += L1_REG * regularize_weights(model)
                
            loss.backward()
            optimizer.step()
            
        # 验证
        model.eval()
        x_val = torch.tensor(val_ds.features, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            risk_val = model(x_val).cpu().numpy().flatten()
        
        try:
            val_cindex = concordance_index(val_ds.survival_months, -risk_val, val_ds.event)
        except:
            val_cindex = 0.5
            
        if val_cindex > best_cindex:
            best_cindex = val_cindex
            counter = 0
            best_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break
    
    # 5. 测试
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    x_test = torch.tensor(test_ds.features, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        risk_test = model(x_test).cpu().numpy().flatten()
        
    try:
        test_cindex = concordance_index(test_ds.survival_months, -risk_test, test_ds.event)
    except:
        test_cindex = 0.5
        
    return test_cindex

# ================= 实验主流程 =================
def run_nested_cv_experiment(exp_name, drop_gene=None):
    """运行完整的 10 Repeats * 5 Folds Nested CV"""
    print(f"\n🚀 [实验执行] {exp_name} (Drop: {drop_gene if drop_gene else 'None'})")
    
    cindices = []
    dataset_cancer = CANCER.lower()
    
    for repeat in range(N_REPEATS):
        current_seed = SEED + repeat * 1000
        
        for fold in range(N_FOLDS):
            # 读取 Nested CV 分片数据 (与 run_SNN.py 路径一致)
            base_path = '/home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_1'
            train_file = f'{base_path}/{dataset_cancer}_{fold}_train.csv'
            test_file = f'{base_path}/{dataset_cancer}_{fold}_val.csv'
            
            if not os.path.exists(train_file):
                print(f"❌ 找不到数据文件: {train_file}")
                return None
                
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            # 特征选择：只选数值类型列，并排除标签列
            exclude_cols = ['survival_months', 'censorship']
            numeric_cols = train_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
            all_features = [c for c in numeric_cols if c not in exclude_cols]
            
            # === 消融逻辑（支持单个基因或基因列表） ===
            if drop_gene:
                # 统一转为列表处理
                genes_to_drop = drop_gene if isinstance(drop_gene, list) else [drop_gene]
                features_to_use = [c for c in all_features if c not in genes_to_drop]
            else:
                features_to_use = all_features
            
            # 训练 (保持 seed 一致)
            seed_for_split = current_seed + fold
            score = train_fold(train_df, test_df, features_to_use, seed_for_split)
            cindices.append(score)
            
    result_array = np.array(cindices)
    print(f"   >>> 平均 C-index: {np.mean(result_array):.4f}")
    return result_array

def main():
    print(f"开始消融实验流程 (CANCER: {CANCER})")
    print("========================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 运行 Baseline (只跑一次，所有方法共用)
    baseline_path = os.path.join(OUTPUT_DIR, "Baseline_cindex.npy")
    if os.path.exists(baseline_path):
        print("📊 [Baseline] 检测到已存在，直接加载...")
        baseline_res = np.load(baseline_path)
    else:
        print("⚙️ [Baseline] 开始训练基线模型 (Full Features)...")
        baseline_res = run_nested_cv_experiment("Baseline", drop_gene=None)
        if baseline_res is not None:
            np.save(baseline_path, baseline_res)
            
    if baseline_res is None:
        print("❌ Baseline 训练失败，程序退出")
        return

    print(f"📊 Baseline Mean C-index: {np.mean(baseline_res):.4f}\n")

    # 2. 循环验证每种 XAI 方法
    for method in XAI_METHODS_TO_TEST:
        print(f"\n{'='*20} Processing XAI: {method} {'='*20}")
        
        # 自动获取 Top 基因
        targets = get_top_consensus_genes(CANCER, method, k=3)
        if not targets:
            print(f"⚠️ 无法获取 {method} 的 Top 基因，跳过")
            continue
            
        # 为该方法创建结果目录
        method_out_dir = os.path.join(OUTPUT_DIR, method)
        os.makedirs(method_out_dir, exist_ok=True)
        
        results_summary = []
        
        # 对每个 Top 基因进行单独消融
        for gene in targets:
            exp_name = f"Ablation_{gene}"
            res_file = os.path.join(method_out_dir, f"{exp_name}.npy")
            
            # 检查是否已跑过
            if os.path.exists(res_file):
                print(f"📦 [Skip] {exp_name} 已存在，加载结果...")
                ablation_res = np.load(res_file)
            else:
                ablation_res = run_nested_cv_experiment(exp_name, drop_gene=gene)
                if ablation_res is not None:
                    np.save(res_file, ablation_res)
            
            # 统计检验
            if ablation_res is not None:
                t_stat, p_val = ttest_rel(baseline_res, ablation_res)
                delta = np.mean(baseline_res) - np.mean(ablation_res)
                
                print(f"   📉 Delta: {delta:.4f} (P-val: {p_val:.2e})")
                
                results_summary.append({
                    'Method': method,
                    'Gene': gene,
                    'Baseline_Mean': np.mean(baseline_res),
                    'Ablation_Mean': np.mean(ablation_res),
                    'Delta': delta,
                    'P_value': p_val
                })
        
        # ===== 联合消融：同时删除 Top-3 基因 =====
        joint_exp_name = "Ablation_Top3_Joint"
        joint_res_file = os.path.join(method_out_dir, f"{joint_exp_name}.npy")
        
        print(f"\n🔗 [联合消融] 同时移除 Top-3: {targets}")
        
        if os.path.exists(joint_res_file):
            print(f"📦 [Skip] {joint_exp_name} 已存在，加载结果...")
            joint_ablation_res = np.load(joint_res_file)
        else:
            joint_ablation_res = run_nested_cv_experiment(joint_exp_name, drop_gene=targets)
            if joint_ablation_res is not None:
                np.save(joint_res_file, joint_ablation_res)
        
        if joint_ablation_res is not None:
            t_stat, p_val = ttest_rel(baseline_res, joint_ablation_res)
            delta = np.mean(baseline_res) - np.mean(joint_ablation_res)
            
            print(f"   📉 Joint Delta: {delta:.4f} (P-val: {p_val:.2e})")
            
            results_summary.append({
                'Method': method,
                'Gene': 'Top3_Joint',
                'Baseline_Mean': np.mean(baseline_res),
                'Ablation_Mean': np.mean(joint_ablation_res),
                'Delta': delta,
                'P_value': p_val
            })
        
        # 保存该方法的汇总表
        if results_summary:
            df = pd.DataFrame(results_summary)
            csv_path = os.path.join(method_out_dir, "ablation_summary.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ {method} 汇总表格已保存: {csv_path}")

    print("\n================ 所有实验完成 ================")

if __name__ == "__main__":
    main()
