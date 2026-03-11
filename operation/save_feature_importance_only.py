import shap
print(shap.__version__)
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle
from model_genomic import SNN
from dataset_survival import RNAseqSurvivalDataset
from captum.attr import IntegratedGradients
from lifelines.utils import concordance_index
from typing import List
import warnings
warnings.filterwarnings('ignore')

# ------------------ 自定义 LRP 类 ------------------
class LRP:
    """支持 SNN（前馈 MLP，激活函数 ReLU/SELU/Tanh）"""
    def __init__(self, model: nn.Module, epsilon: float = 1e-6):
        self.model = model.eval()
        self.epsilon = epsilon
        self.layers: List[nn.Module] = []
        for m in self.model.modules():
            if m is self.model:
                continue
            if isinstance(m, (nn.Sequential,)):
                continue
            if isinstance(m, (nn.Linear, nn.ReLU, nn.SELU, nn.Tanh, nn.Dropout, nn.AlphaDropout)):
                self.layers.append(m)
        self.activations: List[torch.Tensor] = []

    def forward_collect(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = [x]
        out = x
        for layer in self.layers:
            out = layer(out)
            self.activations.append(out)
        return out

    @torch.no_grad()
    def _linear_lrp(self, x: torch.Tensor, layer: nn.Linear, R_out: torch.Tensor) -> torch.Tensor:
        W = layer.weight  # (out,in)
        z = x @ W.t()     # 不加 bias 保持守恒
        stabilizer = self.epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
        z_stable = z + stabilizer
        S = R_out / (z_stable + 1e-12)
        R_in = (S @ W) * x
        return R_in

    def propagate(self, x: torch.Tensor, root: torch.Tensor = None) -> torch.Tensor:
        y = self.forward_collect(x)
        R = y.detach().clone()
        if R.dim() == 1:
            R = R.unsqueeze(-1)
        if root is not None:
            R = root.view(R.shape)
        for li in range(len(self.layers)-1, -1, -1):
            layer = self.layers[li]
            x_l = self.activations[li]
            if isinstance(layer, nn.Linear):
                R = self._linear_lrp(x_l, layer, R)
            elif isinstance(layer, (nn.ReLU, nn.SELU, nn.Tanh, nn.Dropout, nn.AlphaDropout)):
                # 激活/Dropout：直接传递
                pass
            else:
                raise NotImplementedError(f"Unsupported layer for LRP: {type(layer)}")
        return R

# ------------------ IG模型包装器 ------------------
class RiskModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x_omic=x)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1, keepdim=True)  # shape [B, 1]
        return risk

# ------------------ PFI辅助函数 ------------------
def compute_risk_score(model, X, device):
    """预测风险分数"""
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_omic=X_tensor)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).cpu().numpy()
    return risk

def permutation_feature_importance(model, X_val, y_val, feature_names, baseline_cindex, device, n_repeats=5):
    """计算Permutation Feature Importance"""
    n_features = X_val.shape[1]
    importances = np.zeros(n_features)

    for i in range(n_features):
        deltas = []
        for _ in range(n_repeats):
            X_val_perm = X_val.copy()
            X_val_perm[:, i] = np.random.permutation(X_val_perm[:, i])

            risk_perm = compute_risk_score(model, X_val_perm, device)
            cindex_perm = concordance_index(y_val["survival_months"], -risk_perm, y_val["censorship"])

            delta = baseline_cindex - cindex_perm
            deltas.append(delta)

        importances[i] = np.mean(deltas)

    return importances

def save_feature_importance_only(repeat, fold, args, num_features, method='shap'):
    """只保存单个实验的特征重要性排序文件
    
    参数:
    - repeat: 重复次数
    - fold: 折数
    - args: 参数对象
    - num_features: 特征数量
    - method: XAI方法 ('shap', 'ig', 'lrp', 'pfi')
    
    返回:
    - True: 成功保存
    - False: 保存失败
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== 保存{method.upper()}特征重要性 Repeat {repeat} Fold {fold} ===")
    
    # 检查模型文件
    model_path = os.path.join(args.cancer_results_dir, f"repeat{repeat}_s_{fold}_final_test_model.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return False
    
    # 所有方法都使用SNN模型
    model = SNN(omic_input_dim=num_features, model_size_omic='small', n_classes=4)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # 加载数据
    val_csv = os.path.join(args.csv_path, f"{args.cancer_lower}_{fold}_val.csv")
    train_csv = os.path.join(args.csv_path, f"{args.cancer_lower}_{fold}_train.csv")
    
    if not os.path.exists(val_csv) or not os.path.exists(train_csv):
        print(f"数据文件不存在")
        return False
    
    # 验证集
    val_df = pd.read_csv(val_csv)
    val_dataset = RNAseqSurvivalDataset(val_df, label_col='survival_months', seed=args.seed)
    X_val = val_dataset.features
    
    if isinstance(X_val, pd.DataFrame):
        feature_names = X_val.columns.tolist()
        X_val_np = X_val.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
        X_val_np = X_val.numpy()
    
    # 训练集作为background
    train_df = pd.read_csv(train_csv)
    train_dataset = RNAseqSurvivalDataset(train_df, label_col='survival_months', seed=args.seed)
    X_train = train_dataset.features
    
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
    else:
        X_train_np = X_train.numpy()
    
    # 转换为tensor
    background_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    
    # 根据不同方法计算特征重要性
    print(f"计算{method.upper()}值...")
    
    if method == 'shap':
        # SHAP方法
        class RiskWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                # SNN模型输出logits，需要转换为风险分数
                logits = self.model(x_omic=x)
                hazards = torch.sigmoid(logits)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1, keepdim=True)  # shape [B, 1]
                return risk
        
        explainer = shap.GradientExplainer(RiskWrapper(model), background_t)
        shap_exp = explainer(X_val_t)
        shap_values = shap_exp.values
        
        if shap_values.ndim == 3 and shap_values.shape[2] == 1:
            shap_values = shap_values[:, :, 0]
        
        importance = np.abs(shap_values).mean(axis=0)
        
    elif method == 'ig':
        # IG方法
        model_wrapper = RiskModelWrapper(model).to(device)
        baseline = background_t.mean(dim=0, keepdim=True).to(X_val_t.device)
        
        ig = IntegratedGradients(model_wrapper)
        IG_all = ig.attribute(X_val_t, baselines=baseline, n_steps=50).detach().cpu().numpy()
        
        importance = np.abs(IG_all).mean(axis=0)
        
    elif method == 'lrp':
        # LRP方法
        lrp = LRP(model, epsilon=1e-6)
        relevance = lrp.propagate(X_val_t)
        relevance_np = relevance.detach().cpu().numpy()
        
        importance = np.abs(relevance_np).mean(axis=0)
        
    elif method == 'pfi':
        # PFI方法
        # 准备生存数据
        val_dataset = RNAseqSurvivalDataset(val_df, label_col='survival_months', seed=args.seed)
        y_val = pd.DataFrame({
            "survival_months": val_dataset.times,
            "censorship": val_dataset.censorship
        }, index=val_dataset.case_ids)
        
        # 计算baseline C-index
        risk_scores = compute_risk_score(model, X_val_np, device)
        baseline_cindex = concordance_index(y_val["survival_months"], -risk_scores, y_val["censorship"])
        
        # 计算PFI importance
        importance = permutation_feature_importance(
            model, X_val_np, y_val, feature_names, baseline_cindex, device, n_repeats=5
        )
        
    else:
        raise ValueError(f"不支持的方法: {method}")
        
    print(f"{method.upper()}值计算完成")
    
    # 创建特征重要性排序DataFrame
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_score': importance,
        'rank': range(1, len(feature_names) + 1)  # 初始排名
    })
    
    # 按重要性得分降序排序
    feature_importance_df = feature_importance_df.sort_values('importance_score', ascending=False).reset_index(drop=True)
    # 重新分配排名
    feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)
    
    # 保存特征重要性排序文件
    # 根据方法确定保存目录
    if method == 'shap':
        base_dir = args.output_dir
    elif method == 'ig':
        base_dir = args.output_dir
    elif method == 'lrp':
        base_dir = args.output_dir
    elif method == 'pfi':
        base_dir = args.output_dir
    else:
        base_dir = args.output_dir
        
    importance_dir = os.path.join(base_dir, args.cancer, f'{method}_feature_importance')
    os.makedirs(importance_dir, exist_ok=True)
    
    importance_file = os.path.join(importance_dir, f"repeat{repeat}_fold{fold}_{method}_feature_importance_ranking.csv")
    feature_importance_df.to_csv(importance_file, index=False)
    
    print(f"✅ {method.upper()}特征重要性排序已保存: {importance_file}")
    print(f"   总特征数: {len(feature_importance_df)}")
    
    return True

def process_cancer_type(args, num_features, method='shap', num_repeats=10, num_folds=5):
    """处理单个癌症类型的所有实验"""
    success_count = 0
    total_count = 0
    
    for repeat in range(num_repeats):
        for fold in range(num_folds):
            total_count += 1
            success = save_feature_importance_only(repeat, fold, args, num_features, method)
            if success:
                success_count += 1
    
    print(f"\n{args.cancer} {method.upper()}特征重要性保存完成:")
    print(f"成功保存: {success_count}/{total_count} 个实验")
    
    return success_count, total_count

def main(args):
    """主函数 - 只保存特征重要性排序"""
    method = args.method.lower()
    
    print(f"🎯 {method.upper()}特征重要性排序保存工具")
    print(f"   ✅ 计算每个实验的{method.upper()}特征重要性")
    print("   ✅ 保存2000个特征的完整排序")
    print("   ✅ 跳过其他分析步骤，只执行特征重要性保存")
    print(f"   📁 支持的方法: SHAP, IG, LRP, PFI")
    print(f"   🔥 当前使用方法: {method.upper()}\n")
    
    cancer_list = sorted(os.listdir(args.results_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_success = 0
    total_experiments = 0
    
    for cancer in cancer_list:
        if not os.path.isdir(os.path.join(args.results_dir, cancer)):
            continue
            
        args.cancer = cancer
        args.cancer_lower = cancer.lower()
        args.cancer_results_dir = os.path.join(args.results_dir, cancer)
        
        # 固定特征数量
        num_features = 2000
        print(f"\n{'='*60}")
        print(f"正在处理癌症类型: {cancer} ({method.upper()})")
        print(f"{'='*60}")
        
        success, total = process_cancer_type(args, num_features=num_features, method=method)
        total_success += success
        total_experiments += total
    
    print(f"\n{'='*60}")
    print(f"所有癌症类型{method.upper()}特征重要性保存完成:")
    print(f"{'='*60}")
    print(f"总成功保存: {total_success}/{total_experiments} 个实验")
    print(f"保存目录: {args.output_dir}")
    print(f"文件结构: [癌症类型]/{method}_feature_importance/repeat[X]_fold[Y]_{method}_feature_importance_ranking.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只保存每个实验的特征重要性排序文件")
    parser.add_argument('--csv_path', type=str, required=True, help='CSV数据路径')
    parser.add_argument('--results_dir', type=str, required=True, help='Nested CV结果目录')
    parser.add_argument('--output_dir', type=str, required=True, help='特征重要性结果保存目录')
    parser.add_argument('--method', type=str, required=True, choices=['shap', 'ig', 'lrp', 'pfi'], 
                        help='XAI方法: shap, ig, lrp, pfi')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='LRP epsilon 参数')
    
    args = parser.parse_args()
    main(args) 