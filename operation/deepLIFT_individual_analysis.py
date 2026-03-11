import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from captum.attr import DeepLift
from model_genomic import SNN
from dataset_survival import RNAseqSurvivalDataset
from scipy.stats import ttest_1samp, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# 模型包装器 - 适配SNN模型（用于DeepLIFT）
class RiskWrapper(nn.Module):
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

def load_cindex_from_results(results_file):
    """从结果文件中加载cindex"""
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            return results['test_cindex']
    except Exception as e:
        print(f"读取结果文件失败: {e}")
        return None

def perform_cox_analysis(X_val, val_df, top_100_indices, feature_names):
    """对选定特征进行批量单因素Cox分析
    
    参数:
    - X_val: 特征数据（现在使用合并数据集）
    - val_df: 数据框（现在包含训练集+验证集的合并数据）
    - top_100_indices: 要分析的Top 100特征索引
    - feature_names: 特征名称列表
    
    返回:
    - cox_results: Cox分析结果列表
    - protective_count_fdr: FDR校正后保护因子数量
    - risk_count_fdr: FDR校正后风险因子数量
    - protective_count: 原始p值保护因子数量
    - risk_count: 原始p值风险因子数量
    """
    print(f"📊 Cox分析数据统计：")
    print(f"   样本总数: {len(val_df)}")
    print(f"   特征总数: {X_val.shape[1] if hasattr(X_val, 'shape') else 'Unknown'}")
    print(f"   将分析Top 100特征")
    
    cox_results = []
    protective_count = 0
    risk_count = 0
    
    if 'survival_months' in val_df.columns and 'censorship' in val_df.columns:
        survival_time = val_df['survival_months'].values
        censorship = val_df['censorship'].values
        # 转换编码：原数据 censorship=0代表死亡，censorship=1代表删失
        # Cox分析需要：event=1代表事件发生（死亡），event=0代表删失
        event = 1 - censorship  # 翻转编码
        print(f"使用列: survival_months, censorship")
        print(f"原始censorship编码: 0=死亡, 1=删失")
        print(f"转换后event编码: 1=死亡, 0=删失")
    
    print(f"Cox分析：生存时间范围 {np.min(survival_time):.2f} - {np.max(survival_time):.2f}")
    print(f"Cox分析：事件数量 {np.sum(event)} / {len(event)} ({np.mean(event)*100:.1f}%)")
    
    # 对每个特征进行单因素Cox分析
    skipped_features = []
    analyzed_count = 0
    
    for idx in top_100_indices:
        feature_name = feature_names[idx]
        
        try:
            # 准备Cox分析数据
            if isinstance(X_val, pd.DataFrame):
                feature_values = X_val.iloc[:, idx].values
            else:
                feature_values = X_val[:, idx]
            
            # 创建Cox分析数据框
            cox_data = pd.DataFrame({
                'T': survival_time,
                'E': event,
                'feature': feature_values
            })
            
            # 去除缺失值
            original_samples = len(cox_data)
            cox_data = cox_data.dropna()
            samples_after_dropna = len(cox_data)
            
            if len(cox_data) < 10:  # 样本太少
                skipped_features.append({
                    'feature': feature_name,
                    'reason': f'样本不足: {samples_after_dropna}/10 (原始:{original_samples})'
                })
                continue
                
            # 进行Cox分析
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='T', event_col='E')
            
            # 提取结果
            coef = cph.params_['feature']
            hr = np.exp(coef)
            p_value = cph.summary.loc['feature', 'p']
            
            # 🔧 修复：兼容不同版本的lifelines置信区间列名
            ci_cols = cph.confidence_intervals_.columns.tolist()
            if '95% lower-bound' in ci_cols and '95% upper-bound' in ci_cols:
                # 新版本lifelines
                ci_lower = np.exp(cph.confidence_intervals_.loc['feature', '95% lower-bound'])
                ci_upper = np.exp(cph.confidence_intervals_.loc['feature', '95% upper-bound'])
            elif '95% CI lower' in ci_cols and '95% CI upper' in ci_cols:
                # 旧版本lifelines
                ci_lower = np.exp(cph.confidence_intervals_.loc['feature', '95% CI lower'])
                ci_upper = np.exp(cph.confidence_intervals_.loc['feature', '95% CI upper'])
            else:
                # 未知版本，使用第一列和第二列
                ci_lower = np.exp(cph.confidence_intervals_.loc['feature', ci_cols[0]])
                ci_upper = np.exp(cph.confidence_intervals_.loc['feature', ci_cols[1]])
                print(f"⚠️ 未知的置信区间列名: {ci_cols}，使用前两列")
            
            cox_result = {
                'gene': feature_name,
                'coef': coef,
                'hr': hr,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': len(cox_data),
                'n_events': int(cox_data['E'].sum())
            }
            
            # 根据系数判断因子类型
            if p_value < 0.05:  # 显著性阈值
                if coef > 0:
                    cox_result['type'] = 'risk'  # HR > 1, 风险因子
                    risk_count += 1
                elif coef < 0:
                    cox_result['type'] = 'protective'  # HR < 1, 保护因子
                    protective_count += 1
                else:
                    cox_result['type'] = 'neutral'
            else:
                cox_result['type'] = 'not_significant'
            
            cox_results.append(cox_result)
            analyzed_count += 1
            
        except Exception as e:
            skipped_features.append({
                'feature': feature_name,
                'reason': f'Cox分析失败: {str(e)}'
            })
            # 输出详细错误信息用于调试
            print(f"❌ 特征 {feature_name} Cox分析失败:")
            print(f"   错误: {str(e)}")
            print(f"   错误类型: {type(e).__name__}")
            continue
    
    # FDR校正
    if cox_results:
        p_values = [r['p_value'] for r in cox_results]
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        
        # 重新计算因子数量（基于FDR校正）
        protective_count_fdr = 0
        risk_count_fdr = 0
        
        for i, (result, adj_p, sig) in enumerate(zip(cox_results, p_adj, reject)):
            result['p_adj'] = adj_p
            result['significant_fdr'] = sig  # 添加FDR显著性标记
            
            if sig:  # FDR显著
                if result['coef'] > 0:
                    result['type_fdr'] = 'risk'
                    risk_count_fdr += 1
                elif result['coef'] < 0:
                    result['type_fdr'] = 'protective'
                    protective_count_fdr += 1
                else:
                    result['type_fdr'] = 'neutral'
            else:
                result['type_fdr'] = 'not_significant'
        
        print(f"Cox分析完成：分析了Top 100特征中的{analyzed_count}个特征")
        if skipped_features:
            print(f"跳过的特征数量: {len(skipped_features)}")
            # 统计跳过原因
            reason_counts = {}
            for skip in skipped_features:
                reason_type = skip['reason'].split(':')[0]
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
            
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count}个特征")
                
            # 如果需要查看具体跳过的特征（可选）
            if len(skipped_features) <= 5:  # 只有少量跳过时显示详细信息
                print("  跳过的特征详情:")
                for skip in skipped_features:
                    print(f"    {skip['feature']}: {skip['reason']}")
        
        print(f"原始p值显著: Risk {risk_count}, Protective {protective_count}")
        print(f"FDR校正显著: Risk {risk_count_fdr}, Protective {protective_count_fdr}")
        
        return cox_results, protective_count_fdr, risk_count_fdr, protective_count, risk_count
    else:
        # 没有成功分析的特征
        print(f"Cox分析完成：分析了Top 100特征中的0个特征")
        if skipped_features:
            print(f"跳过的特征数量: {len(skipped_features)}")
            # 统计跳过原因
            reason_counts = {}
            for skip in skipped_features:
                reason_type = skip['reason'].split(':')[0]
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
            
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count}个特征")
    
    return [], 0, 0, 0, 0

def compute_deeplift_and_prognostic_factors(repeat, fold, args, num_features):
    """计算单个实验的DeepLIFT值并识别预后因子
    
    分析流程：
    1. DeepLIFT分析（Top 100特征）：
       - Driver因子: DeepLIFT均值显著 > 0 (对风险的正向贡献)
       - Protector因子: DeepLIFT均值显著 < 0 (对风险的负向贡献)
       - 统计检验: t检验测试DeepLIFT值是否显著偏离0
    2. Cox回归分析（Top 100特征）：
       - Risk因子: HR > 1 且显著
       - Protective因子: HR < 1 且显著
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== 分析 Repeat {repeat} Fold {fold} ===")
    
    # 检查模型文件
    model_path = os.path.join(args.cancer_results_dir, f"repeat{repeat}_s_{fold}_final_test_model.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 检查结果文件以获取cindex
    results_file = os.path.join(args.cancer_results_dir, f"repeat{repeat}_fold{fold}_results.pkl")
    cindex = load_cindex_from_results(results_file)
    if cindex is None:
        print(f"无法获取cindex，跳过 repeat{repeat}_fold{fold}")
        return None
    
    # 加载模型
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
        return None
    
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
    
    # 💡 修复：合并训练集和验证集用于Cox分析
    print("🔧 合并训练集和验证集用于Cox分析，提高样本量和统计功效")
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_dataset = RNAseqSurvivalDataset(combined_df, label_col='survival_months', seed=args.seed)
    X_combined = combined_dataset.features
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"合并后样本数: {len(combined_df)}")
    
    if isinstance(X_combined, pd.DataFrame):
        X_combined_np = X_combined.values
    else:
        X_combined_np = X_combined.numpy()
    
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
    else:
        X_train_np = X_train.numpy()
    
    # 转换为tensor
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    
    # 创建包装模型
    wrapped_model = RiskWrapper(model).to(device)
    wrapped_model.eval()
    
    # 计算baseline (训练集均值)
    baseline = torch.mean(X_train_t, dim=0, keepdim=True)
    
    # 使用DeepLIFT
    print(f"计算DeepLIFT值...")
    deeplift = DeepLift(wrapped_model)
    
    # 计算attributions
    attributions = deeplift.attribute(X_val_t, baselines=baseline)
    deeplift_values = attributions.detach().cpu().numpy()
    
    # 计算特征重要性（用于筛选）
    importance = np.abs(deeplift_values).mean(axis=0)
    
    # 保存完整的2000个特征重要性排序文件
    print(f"保存完整特征重要性排序...")
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
    importance_dir = os.path.join(args.deeplift_dir, args.cancer, 'deeplift_feature_importance')
    os.makedirs(importance_dir, exist_ok=True)
    
    importance_file = os.path.join(importance_dir, f"repeat{repeat}_fold{fold}_deeplift_feature_importance_ranking.csv")
    feature_importance_df.to_csv(importance_file, index=False)
    
    print(f"✅ 特征重要性排序已保存: {importance_file}")
    print(f"   总特征数: {len(feature_importance_df)}")
    
    # 识别预后因子 (固定Top 100特征)
    top_100_indices = np.argsort(importance)[-100:][::-1]
    
    # === DeepLIFT分析部分 ===
    prognostic_factors = []
    p_values = []
    
    for i in top_100_indices:
        deeplift_feature = deeplift_values[:, i]  # 该特征在所有样本中的DeepLIFT值
        
        # 核心逻辑：测试DeepLIFT值是否显著偏离0
        try:
            stat, pval = ttest_1samp(deeplift_feature, 0)
        except:
            pval = 1.0
            stat = 0
        
        mean_deeplift = np.mean(deeplift_feature)
        std_deeplift = np.std(deeplift_feature)
        
        # 计算一致性指标
        consistency_ratio = np.abs(np.sign(deeplift_feature).sum()) / len(deeplift_feature)
        positive_ratio = (deeplift_feature > 0).mean()
        
        prognostic_factors.append({
            'gene': feature_names[i],
            'importance': importance[i],
            'mean_deeplift': mean_deeplift,
            'abs_mean_deeplift': abs(mean_deeplift),  # 绝对平均值
            'std_deeplift': std_deeplift,
            'ttest_stat': stat,
            'p_value': pval,
            'consistency_ratio': consistency_ratio,
            'positive_ratio': positive_ratio
        })
        p_values.append(pval)
    
    # === Cox分析部分 ===
    print(f"开始Cox分析...")
    print(f"✅ 使用合并数据集进行Cox分析（训练集+验证集），提高统计功效")
    cox_results, cox_protective_count_fdr, cox_risk_count_fdr, cox_protective_count_raw, cox_risk_count_raw = perform_cox_analysis(
        X_combined, combined_df, top_100_indices, feature_names
    )
    cox_prognostic_count_fdr = cox_protective_count_fdr + cox_risk_count_fdr
    cox_prognostic_count_raw = cox_protective_count_raw + cox_risk_count_raw
    
    # FDR校正 - DeepLIFT分析
    if p_values:
        p_values = np.array(p_values)
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        
        # 统计显著的预后因子数量
        significant_count = 0
        driver_count = 0
        protector_count = 0
        
        for i, (factor, adj_p, sig) in enumerate(zip(prognostic_factors, p_adj, reject)):
            factor['p_adj'] = adj_p
            
            if sig:  # FDR显著偏离0
                significant_count += 1
                # 正确且简单的逻辑：直接基于DeepLIFT均值的符号
                if factor['mean_deeplift'] > 0:
                    factor['type'] = 'driver'      # DeepLIFT > 0: 增加风险
                    driver_count += 1
                elif factor['mean_deeplift'] < 0:
                    factor['type'] = 'protector'   # DeepLIFT < 0: 降低风险
                    protector_count += 1
                else:  # 理论上不应该发生，因为已经显著偏离0
                    factor['type'] = 'neutral'
            else:
                factor['type'] = 'not_significant'
    else:
        significant_count = 0
        driver_count = 0
        protector_count = 0
    
    # 预后因子总数 = 驱动因子 + 保护因子
    prognostic_count_fdr = driver_count + protector_count
    
    # 计算校正前的DeepLIFT预后因子数量
    prognostic_count_raw = 0
    driver_count_raw = 0
    protector_count_raw = 0
    
    for factor in prognostic_factors:
        if factor['p_value'] < 0.05:  # 原始p值显著
            if factor['mean_deeplift'] > 0:
                driver_count_raw += 1
            elif factor['mean_deeplift'] < 0:
                protector_count_raw += 1
    prognostic_count_raw = driver_count_raw + protector_count_raw
    
    result = {
        'repeat': repeat,
        'fold': fold,
        'cindex': cindex,
        'total_factors_tested': len(prognostic_factors),
        'significant_factors': significant_count,
        
        # DeepLIFT分析结果 - FDR校正后
        'deeplift_prognostic_factors_fdr': prognostic_count_fdr,
        'deeplift_driver_factors_fdr': driver_count,
        'deeplift_protector_factors_fdr': protector_count,
        
        # DeepLIFT分析结果 - 原始p值
        'deeplift_prognostic_factors_raw': prognostic_count_raw,
        'deeplift_driver_factors_raw': driver_count_raw,
        'deeplift_protector_factors_raw': protector_count_raw,
        
        'prognostic_factors': prognostic_factors,  # 每个基因的详细DeepLIFT统计信息在这里
        
        # Cox分析结果 - FDR校正后
        'cox_prognostic_factors_fdr': cox_prognostic_count_fdr,
        'cox_protective_factors_fdr': cox_protective_count_fdr,
        'cox_risk_factors_fdr': cox_risk_count_fdr,
        
        # Cox分析结果 - 原始p值
        'cox_prognostic_factors_raw': cox_prognostic_count_raw,
        'cox_protective_factors_raw': cox_protective_count_raw,
        'cox_risk_factors_raw': cox_risk_count_raw,
        
        'cox_results': cox_results,
        'feature_names': feature_names,
        'importance_scores': importance,
        'analysis_method': 'DeepLIFT_plus_Cox_Analysis'  # 标记使用DeepLIFT+Cox分析
    }
    
    # 计算一致性统计
    if prognostic_factors:
        avg_consistency = np.mean([f['consistency_ratio'] for f in prognostic_factors])
        avg_positive_ratio = np.mean([f['positive_ratio'] for f in prognostic_factors])
        high_consistency_count = sum(1 for f in prognostic_factors if f['consistency_ratio'] > 0.8)
    else:
        avg_consistency = 0
        avg_positive_ratio = 0
        high_consistency_count = 0
    
    print(f"C-index: {cindex:.4f}")
    print(f"=== DeepLIFT分析结果 ===")
    print(f"统计显著因子数量: {significant_count} (DeepLIFT值显著偏离0)")
    print(f"原始p值显著 - 预后因子总数: {prognostic_count_raw} (Driver: {driver_count_raw} + Protector: {protector_count_raw})")
    print(f"FDR校正显著 - 预后因子总数: {prognostic_count_fdr} (Driver: {driver_count} + Protector: {protector_count})")
    print(f"特征一致性 - 平均一致性比例: {avg_consistency:.3f}")
    print(f"特征一致性 - 高一致性特征(>0.8): {high_consistency_count}/{len(prognostic_factors)}")
    print(f"=== Cox分析结果 ===")
    print(f"原始p值显著 - Cox预后因子总数: {cox_prognostic_count_raw} (Risk: {cox_risk_count_raw} + Protective: {cox_protective_count_raw})")
    print(f"FDR校正显著 - Cox预后因子总数: {cox_prognostic_count_fdr} (Risk: {cox_risk_count_fdr} + Protective: {cox_protective_count_fdr})")
    
    return result

def analyze_cancer_type(args, num_features, num_repeats=10, num_folds=5):
    """分析单个癌症类型的所有实验"""
    all_results = []
    
    for repeat in range(num_repeats):
        for fold in range(num_folds):
            result = compute_deeplift_and_prognostic_factors(repeat, fold, args, num_features)
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        print(f"癌症类型 {args.cancer} 没有成功分析的实验")
        return None
    
    print(f"\n{args.cancer} 总结:")
    print(f"成功分析的实验数量: {len(all_results)}")
    
    # 提取统计数据
    cindices = [r['cindex'] for r in all_results]
    
    # DeepLIFT分析统计数据 - FDR校正后
    deeplift_prognostic_counts_fdr = [r['deeplift_prognostic_factors_fdr'] for r in all_results]
    deeplift_driver_counts_fdr = [r['deeplift_driver_factors_fdr'] for r in all_results]
    deeplift_protector_counts_fdr = [r['deeplift_protector_factors_fdr'] for r in all_results]
    
    # DeepLIFT分析统计数据 - 原始p值
    deeplift_prognostic_counts_raw = [r['deeplift_prognostic_factors_raw'] for r in all_results]
    deeplift_driver_counts_raw = [r['deeplift_driver_factors_raw'] for r in all_results]
    deeplift_protector_counts_raw = [r['deeplift_protector_factors_raw'] for r in all_results]
    
    # Cox分析统计数据 - FDR校正后
    cox_prognostic_counts_fdr = [r['cox_prognostic_factors_fdr'] for r in all_results]
    cox_risk_counts_fdr = [r['cox_risk_factors_fdr'] for r in all_results]
    cox_protective_counts_fdr = [r['cox_protective_factors_fdr'] for r in all_results]
    
    # Cox分析统计数据 - 原始p值
    cox_prognostic_counts_raw = [r['cox_prognostic_factors_raw'] for r in all_results]
    cox_risk_counts_raw = [r['cox_risk_factors_raw'] for r in all_results]
    cox_protective_counts_raw = [r['cox_protective_factors_raw'] for r in all_results]
    
    # 基本统计
    stats_summary = {
        'cancer': args.cancer,
        'n_experiments': len(all_results),
        'cindex_mean': np.mean(cindices),
        'cindex_std': np.std(cindices),
        'cindex_min': np.min(cindices),
        'cindex_max': np.max(cindices),
        
        # DeepLIFT分析统计 - FDR校正后
        'deeplift_prognostic_factors_fdr_mean': np.mean(deeplift_prognostic_counts_fdr),
        'deeplift_prognostic_factors_fdr_std': np.std(deeplift_prognostic_counts_fdr),
        'deeplift_driver_factors_fdr_mean': np.mean(deeplift_driver_counts_fdr),
        'deeplift_driver_factors_fdr_std': np.std(deeplift_driver_counts_fdr),
        'deeplift_protector_factors_fdr_mean': np.mean(deeplift_protector_counts_fdr),
        'deeplift_protector_factors_fdr_std': np.std(deeplift_protector_counts_fdr),
        
        # DeepLIFT分析统计 - 原始p值
        'deeplift_prognostic_factors_raw_mean': np.mean(deeplift_prognostic_counts_raw),
        'deeplift_prognostic_factors_raw_std': np.std(deeplift_prognostic_counts_raw),
        'deeplift_driver_factors_raw_mean': np.mean(deeplift_driver_counts_raw),
        'deeplift_driver_factors_raw_std': np.std(deeplift_driver_counts_raw),
        'deeplift_protector_factors_raw_mean': np.mean(deeplift_protector_counts_raw),
        'deeplift_protector_factors_raw_std': np.std(deeplift_protector_counts_raw),
        
        # Cox分析统计 - FDR校正后
        'cox_prognostic_factors_fdr_mean': np.mean(cox_prognostic_counts_fdr),
        'cox_prognostic_factors_fdr_std': np.std(cox_prognostic_counts_fdr),
        'cox_risk_factors_fdr_mean': np.mean(cox_risk_counts_fdr),
        'cox_risk_factors_fdr_std': np.std(cox_risk_counts_fdr),
        'cox_protective_factors_fdr_mean': np.mean(cox_protective_counts_fdr),
        'cox_protective_factors_fdr_std': np.std(cox_protective_counts_fdr),
        
        # Cox分析统计 - 原始p值
        'cox_prognostic_factors_raw_mean': np.mean(cox_prognostic_counts_raw),
        'cox_prognostic_factors_raw_std': np.std(cox_prognostic_counts_raw),
        'cox_risk_factors_raw_mean': np.mean(cox_risk_counts_raw),
        'cox_risk_factors_raw_std': np.std(cox_risk_counts_raw),
        'cox_protective_factors_raw_mean': np.mean(cox_protective_counts_raw),
        'cox_protective_factors_raw_std': np.std(cox_protective_counts_raw)
    }
    
    # 相关性分析 - 🔧 修改：主要使用原始p值（校正前）的预后因子数量
    if len(cindices) > 2:  # 至少需要3个数据点
        # C-index与DeepLIFT预后因子总数的相关性 (原始p值 - 主要分析)
        corr_deeplift_prog_raw, p_deeplift_prog_raw = pearsonr(cindices, deeplift_prognostic_counts_raw)
        spearman_deeplift_prog_raw, sp_deeplift_prog_raw = spearmanr(cindices, deeplift_prognostic_counts_raw)
        
        # C-index与driver因子数量的相关性 (原始p值)
        corr_driver_raw, p_driver_raw = pearsonr(cindices, deeplift_driver_counts_raw)
        spearman_driver_raw, sp_driver_raw = spearmanr(cindices, deeplift_driver_counts_raw)
        
        # C-index与protector因子数量的相关性 (原始p值)
        corr_protector_raw, p_protector_raw = pearsonr(cindices, deeplift_protector_counts_raw)
        spearman_protector_raw, sp_protector_raw = spearmanr(cindices, deeplift_protector_counts_raw)
        
        # C-index与Cox预后因子总数的相关性 (原始p值 - 主要分析)
        corr_cox_prog_raw, p_cox_prog_raw = pearsonr(cindices, cox_prognostic_counts_raw)
        spearman_cox_prog_raw, sp_cox_prog_raw = spearmanr(cindices, cox_prognostic_counts_raw)
        
        # 附加：FDR校正后的相关性（用于对比）
        corr_deeplift_prog_fdr, p_deeplift_prog_fdr = pearsonr(cindices, deeplift_prognostic_counts_fdr)
        corr_cox_prog_fdr, p_cox_prog_fdr = pearsonr(cindices, cox_prognostic_counts_fdr)
        
        stats_summary.update({
            # DeepLIFT相关性 - 原始p值（主要分析）
            'corr_cindex_deeplift_prognostic_raw_pearson': corr_deeplift_prog_raw,
            'corr_cindex_deeplift_prognostic_raw_pearson_p': p_deeplift_prog_raw,
            'corr_cindex_deeplift_prognostic_raw_spearman': spearman_deeplift_prog_raw,
            'corr_cindex_deeplift_prognostic_raw_spearman_p': sp_deeplift_prog_raw,
            
            'corr_cindex_driver_raw_pearson': corr_driver_raw,
            'corr_cindex_driver_raw_pearson_p': p_driver_raw,
            'corr_cindex_driver_raw_spearman': spearman_driver_raw,
            'corr_cindex_driver_raw_spearman_p': sp_driver_raw,
            
            'corr_cindex_protector_raw_pearson': corr_protector_raw,
            'corr_cindex_protector_raw_pearson_p': p_protector_raw,
            'corr_cindex_protector_raw_spearman': spearman_protector_raw,
            'corr_cindex_protector_raw_spearman_p': sp_protector_raw,
            
            # Cox相关性 - 原始p值（主要分析）
            'corr_cindex_cox_prognostic_raw_pearson': corr_cox_prog_raw,
            'corr_cindex_cox_prognostic_raw_pearson_p': p_cox_prog_raw,
            'corr_cindex_cox_prognostic_raw_spearman': spearman_cox_prog_raw,
            'corr_cindex_cox_prognostic_raw_spearman_p': sp_cox_prog_raw,
            
            # FDR校正后的相关性（辅助分析）
            'corr_cindex_deeplift_prognostic_fdr_pearson': corr_deeplift_prog_fdr,
            'corr_cindex_deeplift_prognostic_fdr_pearson_p': p_deeplift_prog_fdr,
            'corr_cindex_cox_prognostic_fdr_pearson': corr_cox_prog_fdr,
            'corr_cindex_cox_prognostic_fdr_pearson_p': p_cox_prog_fdr
        })
        
        print(f"=== DeepLIFT相关性分析（使用原始p值预后因子数量）===")
        print(f"C-index vs DeepLIFT预后因子总数 - Pearson: r={corr_deeplift_prog_raw:.3f}, p={p_deeplift_prog_raw:.3f}")
        print(f"C-index vs Driver因子数量 - Pearson: r={corr_driver_raw:.3f}, p={p_driver_raw:.3f}")
        print(f"C-index vs Protector因子数量 - Pearson: r={corr_protector_raw:.3f}, p={p_protector_raw:.3f}")
        print(f"=== Cox相关性分析（使用原始p值预后因子数量）===")
        print(f"C-index vs Cox预后因子总数 - Pearson: r={corr_cox_prog_raw:.3f}, p={p_cox_prog_raw:.3f}")
        print(f"对比FDR校正后 - DeepLIFT: r={corr_deeplift_prog_fdr:.3f}, Cox: r={corr_cox_prog_fdr:.3f}")
    
    # 保存详细结果
    results_dir = os.path.join(args.deeplift_dir, args.cancer)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存每个实验的详细结果
    detailed_results_df = pd.DataFrame([{
        'repeat': r['repeat'],
        'fold': r['fold'],
        'cindex': r['cindex'],
        'significant_factors': r['significant_factors'],
        
        # DeepLIFT分析结果 - FDR校正后
        'deeplift_prognostic_factors_fdr': r['deeplift_prognostic_factors_fdr'],
        'deeplift_driver_factors_fdr': r['deeplift_driver_factors_fdr'],
        'deeplift_protector_factors_fdr': r['deeplift_protector_factors_fdr'],
        
        # DeepLIFT分析结果 - 原始p值
        'deeplift_prognostic_factors_raw': r['deeplift_prognostic_factors_raw'],
        'deeplift_driver_factors_raw': r['deeplift_driver_factors_raw'],
        'deeplift_protector_factors_raw': r['deeplift_protector_factors_raw'],
        
        # Cox分析结果 - FDR校正后
        'cox_prognostic_factors_fdr': r['cox_prognostic_factors_fdr'],
        'cox_risk_factors_fdr': r['cox_risk_factors_fdr'],
        'cox_protective_factors_fdr': r['cox_protective_factors_fdr'],
        
        # Cox分析结果 - 原始p值
        'cox_prognostic_factors_raw': r['cox_prognostic_factors_raw'],
        'cox_risk_factors_raw': r['cox_risk_factors_raw'],
        'cox_protective_factors_raw': r['cox_protective_factors_raw']
    } for r in all_results])
    
    detailed_results_df.to_csv(os.path.join(results_dir, f"{args.cancer}_detailed_results.csv"), index=False)
    
    # 生成可视化
    if len(all_results) > 2:
        create_correlation_plots(args.cancer, detailed_results_df, results_dir)
    
    # 保存完整结果
    with open(os.path.join(results_dir, f"{args.cancer}_complete_results.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
    
    # 同时保存Cox分析的详细结果
    cox_df = detailed_results_df[['repeat', 'fold', 'cindex', 
                                  'cox_prognostic_factors_fdr', 'cox_risk_factors_fdr', 'cox_protective_factors_fdr',
                                  'cox_prognostic_factors_raw', 'cox_risk_factors_raw', 'cox_protective_factors_raw']].copy()
    cox_df.to_csv(os.path.join(results_dir, f"{args.cancer}_cox_analysis_summary.csv"), index=False)
        
    # 保存每个实验的Cox分析详细结果（包含每个基因的DeepLIFT统计信息）
    all_cox_details = []
    for r in all_results:
        if 'cox_results' in r and r['cox_results'] and 'prognostic_factors' in r:
            # 创建基因名到DeepLIFT统计信息的映射
            deeplift_stats_map = {pf['gene']: pf for pf in r['prognostic_factors']}
            
            for cox_result in r['cox_results']:
                gene_name = cox_result['gene']
                cox_detail = {
                    'repeat': r['repeat'],
                    'fold': r['fold'],
                    'cindex': r['cindex'],
                    **cox_result
                }
                
                # 添加该基因的DeepLIFT统计信息
                if gene_name in deeplift_stats_map:
                    deeplift_stats = deeplift_stats_map[gene_name]
                    cox_detail.update({
                        'mean_deeplift': deeplift_stats['mean_deeplift'],
                        'abs_mean_deeplift': deeplift_stats['abs_mean_deeplift'],
                        'deeplift_importance': deeplift_stats['importance'],
                        'deeplift_p_value': deeplift_stats['p_value'],
                        'deeplift_p_adj': deeplift_stats.get('p_adj', None),
                        'deeplift_type': deeplift_stats.get('type', 'not_significant')
                    })
                
                all_cox_details.append(cox_detail)
    
    if all_cox_details:
        cox_details_df = pd.DataFrame(all_cox_details)
        cox_details_df.to_csv(os.path.join(results_dir, f"{args.cancer}_cox_detailed_results.csv"), index=False)
    
    return stats_summary

def create_correlation_plots(cancer, df, results_dir):
    """创建相关性图（包含DeepLIFT和Cox分析）- 🔧 添加拟合直线和统计信息"""
    from scipy.stats import pearsonr
    import numpy as np
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 定义绘图函数（添加拟合直线和统计信息）
    def plot_with_fit(ax, x, y, xlabel, ylabel, title, color):
        # 散点图
        ax.scatter(x, y, alpha=0.7, color=color, s=50)
        
        # 计算相关系数和p值
        r, p = pearsonr(x, y)
        
        # 添加拟合直线
        if len(x) > 1:
            z = np.polyfit(x, y, 1)  # 一次多项式拟合
            p_fit = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2)
        
        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 添加统计信息文本框
        if p < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p:.3f}"
        
        # 判断显著性
        if p < 0.05:
            sig_text = "*"
            if p < 0.01:
                sig_text = "**"
            if p < 0.001:
                sig_text = "***"
        else:
            sig_text = "ns"
        
        stats_text = f"r = {r:.3f}\n{p_text}\n{sig_text}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10, fontweight='bold')
        
        return r, p
    
    # === 第一行：DeepLIFT分析结果（使用原始p值预后因子数量）===
    # C-index vs DeepLIFT预后因子总数
    plot_with_fit(axes[0,0], df['cindex'], df['deeplift_prognostic_factors_raw'], 
                  'C-index', 'DeepLIFT Prognostic Factors (Raw p)', 
                  f'{cancer}: C-index vs DeepLIFT Prognostic Factors', 'purple')
    
    # C-index vs Driver因子数量
    plot_with_fit(axes[0,1], df['cindex'], df['deeplift_driver_factors_raw'], 
                  'C-index', 'DeepLIFT Driver Factors (Raw p)', 
                  f'{cancer}: C-index vs DeepLIFT Driver Factors', 'red')
    
    # C-index vs Protector因子数量
    plot_with_fit(axes[0,2], df['cindex'], df['deeplift_protector_factors_raw'], 
                  'C-index', 'DeepLIFT Protector Factors (Raw p)', 
                  f'{cancer}: C-index vs DeepLIFT Protector Factors', 'blue')
    
    # === 第二行：Cox分析结果（使用原始p值预后因子数量）===
    # C-index vs Cox预后因子总数
    plot_with_fit(axes[1,0], df['cindex'], df['cox_prognostic_factors_raw'], 
                  'C-index', 'Cox Prognostic Factors (Raw p)', 
                  f'{cancer}: C-index vs Cox Prognostic Factors', 'green')
    
    # C-index vs Cox风险因子数量
    plot_with_fit(axes[1,1], df['cindex'], df['cox_risk_factors_raw'], 
                  'C-index', 'Cox Risk Factors (Raw p)', 
                  f'{cancer}: C-index vs Cox Risk Factors', 'orange')
    
    # C-index vs Cox保护因子数量
    plot_with_fit(axes[1,2], df['cindex'], df['cox_protective_factors_raw'], 
                  'C-index', 'Cox Protective Factors (Raw p)', 
                  f'{cancer}: C-index vs Cox Protective Factors', 'cyan')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{cancer}_correlation_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    """主函数 - DeepLIFT分析 + Cox回归分析"""
    print("🔥 综合分析框架: DeepLIFT + Cox回归")
    print("=== DeepLIFT分析部分 ===")
    print("   ✅ Driver因子: DeepLIFT均值显著 > 0 (增加风险)")
    print("   ✅ Protector因子: DeepLIFT均值显著 < 0 (降低风险)")
    print("   ✅ 统计检验: t检验测试DeepLIFT值是否显著偏离0")
    print("=== Cox回归分析部分 ===")
    print("   ✅ 分析Top 100重要特征的单因素Cox回归")
    print("   ✅ Risk因子: HR > 1 且 p < 0.05 (增加风险)")
    print("   ✅ Protective因子: HR < 1 且 p < 0.05 (降低风险)")
    print("   ✅ FDR多重检验校正")
    print("=== 特征重要性保存 ===")
    print("   💾 保存每个实验的2000个特征完整重要性排序")
    print("   📁 文件位置: [癌症类型]/feature_importance/repeat[X]_fold[Y]_feature_importance_ranking.csv")
    print("=== 相关性分析 ===")
    print("   ✅ C-index vs Cox预后因子总数的相关性\n")
    
    cancer_list = sorted(os.listdir(args.results_dir))
    os.makedirs(args.deeplift_dir, exist_ok=True)
    
    all_cancer_summaries = []
    
    for cancer in cancer_list:
        if not os.path.isdir(os.path.join(args.results_dir, cancer)):
            continue
            
        args.cancer = cancer
        args.cancer_lower = cancer.lower()
        args.cancer_results_dir = os.path.join(args.results_dir, cancer)
        
        # 固定特征数量和筛选数量
        num_features = 2000  # 固定2000个特征，筛选Top 100进行分析
        print(f"\n{'='*60}")
        print(f"正在处理癌症类型: {cancer}")
        print(f"{'='*60}")
        
        summary = analyze_cancer_type(args, num_features=num_features)
        if summary is not None:
            all_cancer_summaries.append(summary)
    
    # 保存所有癌症的汇总结果
    if all_cancer_summaries:
        summary_df = pd.DataFrame(all_cancer_summaries)
        summary_df.to_csv(os.path.join(args.deeplift_dir, "all_cancers_summary.csv"), index=False)
        
        print(f"\n{'='*60}")
        print("所有癌症类型汇总 (DeepLIFT + Cox分析):")
        print(f"{'='*60}")
        
        # DeepLIFT分析汇总 - 主要结果（原始p值）
        print("=== DeepLIFT分析汇总（主要结果：使用原始p值预后因子数量）===")
        deeplift_cols_raw = ['cancer', 'cindex_mean', 'deeplift_prognostic_factors_raw_mean', 
                        'corr_cindex_deeplift_prognostic_raw_pearson', 'corr_cindex_deeplift_prognostic_raw_pearson_p']
        print(summary_df[deeplift_cols_raw].to_string(index=False))
        
        # Cox分析汇总 - 主要结果（原始p值）
        print("\n=== Cox分析汇总（主要结果：使用原始p值预后因子数量）===")
        cox_cols_raw = ['cancer', 'cindex_mean', 'cox_prognostic_factors_raw_mean', 
                       'corr_cindex_cox_prognostic_raw_pearson', 'corr_cindex_cox_prognostic_raw_pearson_p']
        print(summary_df[cox_cols_raw].to_string(index=False))
        
        # 对比：FDR校正后的结果
        print("\n=== 对比：FDR校正后的相关性 ===")
        compare_cols = ['cancer', 'deeplift_prognostic_factors_fdr_mean', 'cox_prognostic_factors_fdr_mean',
                       'corr_cindex_deeplift_prognostic_fdr_pearson', 'corr_cindex_cox_prognostic_fdr_pearson']
        print(summary_df[compare_cols].to_string(index=False))
        
        print(f"\n💡 说明:")
        print(f"   🎯 主要分析：使用原始p值（校正前）的预后因子数量计算相关性")
        print(f"   📊 可视化：散点图包含拟合直线、相关系数和显著性检验")
        print(f"   💾 特征重要性：每个实验保存完整的2000个特征排序文件")
        print(f"   DeepLIFT预后因子: 基于DeepLIFT值是否显著偏离0来判定")
        print(f"   - Driver(增加风险): DeepLIFT > 0")
        print(f"   - Protector(降低风险): DeepLIFT < 0")
        print(f"   Cox预后因子: 基于单因素Cox回归分析结果")
        print(f"   - Risk因子: HR > 1 (p < 0.05)")
        print(f"   - Protective因子: HR < 1 (p < 0.05)")
        print(f"   📈 相关性显著性: * p<0.05, ** p<0.01, *** p<0.001, ns 不显著")
        print(f"   📁 特征重要性文件: {args.deeplift_dir}/[癌症类型]/feature_importance/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单独分析每个实验的DeepLIFT值和预后因子")
    parser.add_argument('--csv_path', type=str, required=True, help='CSV数据路径')
    parser.add_argument('--results_dir', type=str, required=True, help='Nested CV结果目录')
    parser.add_argument('--deeplift_dir', type=str, required=True, help='DeepLIFT分析结果保存目录')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    
    args = parser.parse_args()
    main(args)