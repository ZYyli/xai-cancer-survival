import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
from lifelines.utils import concordance_index
from model_genomic import SNN
from dataset_survival import RNAseqSurvivalDataset
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

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
            reason_counts = {}
            for skip in skipped_features:
                reason_type = skip['reason'].split(':')[0]
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
            
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count}个特征")
                
            if len(skipped_features) <= 5:
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
            reason_counts = {}
            for skip in skipped_features:
                reason_type = skip['reason'].split(':')[0]
                reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
            
            for reason, count in reason_counts.items():
                print(f"  - {reason}: {count}个特征")
    
    return [], 0, 0, 0, 0

def load_model(model_path, input_dim, device):
    """加载已训练好的SNN模型"""
    model = SNN(omic_input_dim=input_dim, model_size_omic='small', n_classes=4)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

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

def compute_pfi_and_prognostic_factors(repeat, fold, args, num_features):
    """计算单个实验的PFI值并识别预后因子"""
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
    
    # 加载数据
    val_csv = os.path.join(args.csv_path, f"{args.cancer_lower}_{fold}_val.csv")
    if not os.path.exists(val_csv):
        print(f"数据文件不存在: {val_csv}")
        return None
    
    val_df = pd.read_csv(val_csv)
    val_dataset = RNAseqSurvivalDataset(val_df, label_col='survival_months', seed=args.seed)
    X_val = val_dataset.features
    
    if isinstance(X_val, pd.DataFrame):
        feature_names = X_val.columns.tolist()
        X_val_np = X_val.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
        X_val_np = X_val.numpy()
    
    # 训练集作为background（用于合并Cox分析）
    train_csv = os.path.join(args.csv_path, f"{args.cancer_lower}_{fold}_train.csv")
    if not os.path.exists(train_csv):
        print(f"训练数据文件不存在: {train_csv}")
        return None
    
    train_df = pd.read_csv(train_csv)
    
    # 💡 修复：合并训练集和验证集用于Cox分析
    print("🔧 合并训练集和验证集用于Cox分析，提高样本量和统计功效")
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_dataset = RNAseqSurvivalDataset(combined_df, label_col='survival_months', seed=args.seed)
    X_combined = combined_dataset.features
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"合并后样本数: {len(combined_df)}")
    
    # 加载模型
    model = load_model(model_path, input_dim=X_val.shape[1], device=device)
    
    # 准备生存数据
    y_val = pd.DataFrame({
        "survival_months": val_dataset.times,
        "censorship": val_dataset.censorship
    }, index=val_dataset.case_ids)
    
    # 计算baseline C-index
    risk_scores = compute_risk_score(model, X_val_np, device)
    baseline_cindex = concordance_index(y_val["survival_months"], -risk_scores, y_val["censorship"])
    
    print(f"计算PFI值...")
    # 计算PFI importance
    importance = permutation_feature_importance(
        model, X_val_np, y_val, feature_names, baseline_cindex, device, n_repeats=5
    )
    
    # 💾 新增：保存完整的2000个特征重要性排序文件
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
    importance_dir = os.path.join(args.pfi_dir, args.cancer, 'pfi_feature_importance')
    os.makedirs(importance_dir, exist_ok=True)
    
    importance_file = os.path.join(importance_dir, f"repeat{repeat}_fold{fold}_pfi_feature_importance_ranking.csv")
    feature_importance_df.to_csv(importance_file, index=False)
    
    print(f"✅ 特征重要性排序已保存: {importance_file}")
    print(f"   总特征数: {len(feature_importance_df)}")
    
    # 按中位数分高低风险组
    median_risk = np.median(risk_scores)
    high_idx = risk_scores >= median_risk
    low_idx = risk_scores < median_risk
    
    # 识别预后因子 (以Top 100为例)
    top_100_indices = np.argsort(importance)[-100:][::-1]
    
    prognostic_factors = []
    p_values = []
    
    # 对Top 100特征进行统计检验
    # 注意：PFI是基于特征重要性，我们用重要性值本身进行高低组比较
    for i in top_100_indices:
        # 为了进行高低风险组比较，我们需要用该特征在不同风险组中的表现差异
        # 这里我们重新计算该特征在高低风险组中的PFI差异
        high_samples = X_val_np[high_idx]
        low_samples = X_val_np[low_idx]
        
        if len(high_samples) == 0 or len(low_samples) == 0:
            continue
        
        # 计算该特征在高风险组的PFI
        high_pfi = []
        for _ in range(3):  # 减少计算量
            X_high_perm = high_samples.copy()
            X_high_perm[:, i] = np.random.permutation(X_high_perm[:, i])
            
            # 构建完整数据集
            X_perm_full = X_val_np.copy()
            X_perm_full[high_idx] = X_high_perm
            
            risk_perm = compute_risk_score(model, X_perm_full, device)
            cindex_perm = concordance_index(y_val["survival_months"], -risk_perm, y_val["censorship"])
            high_pfi.append(baseline_cindex - cindex_perm)
        
        # 计算该特征在低风险组的PFI
        low_pfi = []
        for _ in range(3):
            X_low_perm = low_samples.copy()
            X_low_perm[:, i] = np.random.permutation(X_low_perm[:, i])
            
            # 构建完整数据集
            X_perm_full = X_val_np.copy()
            X_perm_full[low_idx] = X_low_perm
            
            risk_perm = compute_risk_score(model, X_perm_full, device)
            cindex_perm = concordance_index(y_val["survival_months"], -risk_perm, y_val["censorship"])
            low_pfi.append(baseline_cindex - cindex_perm)
        
        mean_high = np.mean(high_pfi)
        mean_low = np.mean(low_pfi)
        diff = mean_high - mean_low
        
        try:
            stat, pval = mannwhitneyu(high_pfi, low_pfi, alternative="two-sided")
        except:
            pval = 1.0
        
        # 计算PFI的稳定性指标
        high_pfi_std = np.std(high_pfi)
        low_pfi_std = np.std(low_pfi)
        combined_std = np.sqrt((high_pfi_std**2 + low_pfi_std**2) / 2)
        
        prognostic_factors.append({
            'gene': feature_names[i],
            'importance': importance[i],
            'mean_high': mean_high,
            'mean_low': mean_low,
            'mean_pfi': (mean_high + mean_low) / 2,  # 平均PFI值
            'abs_mean_pfi': abs((mean_high + mean_low) / 2),  # 绝对平均PFI值
            'diff': diff,
            'p_value': pval,
            'high_pfi_std': high_pfi_std,
            'low_pfi_std': low_pfi_std,
            'stability_metric': combined_std  # PFI值的稳定性
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
    
    # FDR校正 - PFI分析
    if p_values:
        p_values = np.array(p_values)
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        
        # 统计显著的预后因子数量
        significant_count = 0
        driver_count = 0
        protector_count = 0
        
        for i, (factor, adj_p, sig) in enumerate(zip(prognostic_factors, p_adj, reject)):
            factor['p_adj'] = adj_p
            if sig:  # FDR显著
                significant_count += 1
                if factor['diff'] > 0 and factor['mean_high'] > 0:
                    factor['type'] = 'driver'
                    driver_count += 1
                elif factor['diff'] < 0 and factor['mean_low'] < 0:
                    factor['type'] = 'protector'  
                    protector_count += 1
                else:
                    factor['type'] = 'other'
            else:
                factor['type'] = 'not_significant'
    else:
        significant_count = 0
        driver_count = 0
        protector_count = 0
    
    # 预后因子总数 = 驱动因子 + 保护因子 (FDR校正后)
    prognostic_count_fdr = driver_count + protector_count
    
    # 计算校正前的PFI预后因子数量
    prognostic_count_raw = 0
    driver_count_raw = 0
    protector_count_raw = 0
    
    for factor in prognostic_factors:
        if factor['p_value'] < 0.05:  # 原始p值显著
            if factor['diff'] > 0 and factor['mean_high'] > 0:
                driver_count_raw += 1
            elif factor['diff'] < 0 and factor['mean_low'] < 0:
                protector_count_raw += 1
    prognostic_count_raw = driver_count_raw + protector_count_raw
    
    result = {
        'repeat': repeat,
        'fold': fold,
        'cindex': cindex,
        'total_factors_tested': len(prognostic_factors),
        'significant_factors': significant_count,
        
        # PFI分析结果 - FDR校正后
        'pfi_prognostic_factors_fdr': prognostic_count_fdr,
        'pfi_driver_factors_fdr': driver_count,
        'pfi_protector_factors_fdr': protector_count,
        
        # PFI分析结果 - 原始p值
        'pfi_prognostic_factors_raw': prognostic_count_raw,
        'pfi_driver_factors_raw': driver_count_raw,
        'pfi_protector_factors_raw': protector_count_raw,
        
        'prognostic_factors': prognostic_factors,  # 每个基因的详细PFI统计信息在这里
        
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
        'analysis_method': 'PFI_plus_Cox_Analysis'
    }
    
    # 计算PFI稳定性统计
    if prognostic_factors:
        avg_stability = np.mean([f['stability_metric'] for f in prognostic_factors])
        high_stability_count = sum(1 for f in prognostic_factors if f['stability_metric'] < 0.1)  # 低标准差表示高稳定性
    else:
        avg_stability = 0
        high_stability_count = 0
    
    print(f"C-index: {cindex:.4f}")
    print(f"=== PFI分析结果 ===")
    print(f"统计显著因子数量: {significant_count} (PFI高低风险组差异显著)")
    print(f"原始p值显著 - 预后因子总数: {prognostic_count_raw} (Driver: {driver_count_raw} + Protector: {protector_count_raw})")
    print(f"FDR校正显著 - 预后因子总数: {prognostic_count_fdr} (Driver: {driver_count} + Protector: {protector_count})")
    print(f"PFI稳定性 - 平均稳定性指标: {avg_stability:.4f}")
    print(f"PFI稳定性 - 高稳定性特征(<0.1): {high_stability_count}/{len(prognostic_factors)}")
    print(f"=== Cox分析结果 ===")
    print(f"原始p值显著 - Cox预后因子总数: {cox_prognostic_count_raw} (Risk: {cox_risk_count_raw} + Protective: {cox_protective_count_raw})")
    print(f"FDR校正显著 - Cox预后因子总数: {cox_prognostic_count_fdr} (Risk: {cox_risk_count_fdr} + Protective: {cox_protective_count_fdr})")
    
    return result

def analyze_cancer_type(args, num_features, num_repeats=10, num_folds=5):
    """分析单个癌症类型的所有实验"""
    all_results = []
    
    for repeat in range(num_repeats):
        for fold in range(num_folds):
            try:
                result = compute_pfi_and_prognostic_factors(repeat, fold, args, num_features)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ {args.cancer} Repeat {repeat} Fold {fold} 分析失败: {str(e)}")
                continue
    
    if not all_results:
        print(f"癌症类型 {args.cancer} 没有成功分析的实验")
        return None
    
    print(f"\n{args.cancer} 总结:")
    print(f"成功分析的实验数量: {len(all_results)}")
    
    # 提取统计数据
    cindices = [r['cindex'] for r in all_results]
    
    # PFI分析统计数据 - FDR校正后
    pfi_prognostic_counts_fdr = [r['pfi_prognostic_factors_fdr'] for r in all_results]
    pfi_driver_counts_fdr = [r['pfi_driver_factors_fdr'] for r in all_results]
    pfi_protector_counts_fdr = [r['pfi_protector_factors_fdr'] for r in all_results]
    
    # PFI分析统计数据 - 原始p值
    pfi_prognostic_counts_raw = [r['pfi_prognostic_factors_raw'] for r in all_results]
    pfi_driver_counts_raw = [r['pfi_driver_factors_raw'] for r in all_results]
    pfi_protector_counts_raw = [r['pfi_protector_factors_raw'] for r in all_results]
    
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
        
        # PFI分析统计 - FDR校正后
        'pfi_prognostic_factors_fdr_mean': np.mean(pfi_prognostic_counts_fdr),
        'pfi_prognostic_factors_fdr_std': np.std(pfi_prognostic_counts_fdr),
        'pfi_driver_factors_fdr_mean': np.mean(pfi_driver_counts_fdr),
        'pfi_driver_factors_fdr_std': np.std(pfi_driver_counts_fdr),
        'pfi_protector_factors_fdr_mean': np.mean(pfi_protector_counts_fdr),
        'pfi_protector_factors_fdr_std': np.std(pfi_protector_counts_fdr),
        
        # PFI分析统计 - 原始p值
        'pfi_prognostic_factors_raw_mean': np.mean(pfi_prognostic_counts_raw),
        'pfi_prognostic_factors_raw_std': np.std(pfi_prognostic_counts_raw),
        'pfi_driver_factors_raw_mean': np.mean(pfi_driver_counts_raw),
        'pfi_driver_factors_raw_std': np.std(pfi_driver_counts_raw),
        'pfi_protector_factors_raw_mean': np.mean(pfi_protector_counts_raw),
        'pfi_protector_factors_raw_std': np.std(pfi_protector_counts_raw),
        
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
    if len(cindices) > 2:
        # C-index与PFI预后因子总数的相关性 (原始p值 - 主要分析)
        corr_pfi_prog_raw, p_pfi_prog_raw = pearsonr(cindices, pfi_prognostic_counts_raw)
        spearman_pfi_prog_raw, sp_pfi_prog_raw = spearmanr(cindices, pfi_prognostic_counts_raw)
        
        # C-index与driver因子数量的相关性 (原始p值)
        corr_driver_raw, p_driver_raw = pearsonr(cindices, pfi_driver_counts_raw)
        spearman_driver_raw, sp_driver_raw = spearmanr(cindices, pfi_driver_counts_raw)
        
        # C-index与protector因子数量的相关性 (原始p值)
        corr_protector_raw, p_protector_raw = pearsonr(cindices, pfi_protector_counts_raw)
        spearman_protector_raw, sp_protector_raw = spearmanr(cindices, pfi_protector_counts_raw)
        
        # C-index与Cox预后因子总数的相关性 (原始p值 - 主要分析)
        corr_cox_prog_raw, p_cox_prog_raw = pearsonr(cindices, cox_prognostic_counts_raw)
        spearman_cox_prog_raw, sp_cox_prog_raw = spearmanr(cindices, cox_prognostic_counts_raw)
        
        # 附加：FDR校正后的相关性（用于对比）
        corr_pfi_prog_fdr, p_pfi_prog_fdr = pearsonr(cindices, pfi_prognostic_counts_fdr)
        corr_cox_prog_fdr, p_cox_prog_fdr = pearsonr(cindices, cox_prognostic_counts_fdr)
        
        stats_summary.update({
            # PFI相关性 - 原始p值（主要分析）
            'corr_cindex_pfi_prognostic_raw_pearson': corr_pfi_prog_raw,
            'corr_cindex_pfi_prognostic_raw_pearson_p': p_pfi_prog_raw,
            'corr_cindex_pfi_prognostic_raw_spearman': spearman_pfi_prog_raw,
            'corr_cindex_pfi_prognostic_raw_spearman_p': sp_pfi_prog_raw,
            
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
            'corr_cindex_pfi_prognostic_fdr_pearson': corr_pfi_prog_fdr,
            'corr_cindex_pfi_prognostic_fdr_pearson_p': p_pfi_prog_fdr,
            'corr_cindex_cox_prognostic_fdr_pearson': corr_cox_prog_fdr,
            'corr_cindex_cox_prognostic_fdr_pearson_p': p_cox_prog_fdr
        })
        
        print(f"=== PFI相关性分析（使用原始p值预后因子数量）===")
        print(f"C-index vs PFI预后因子总数 - Pearson: r={corr_pfi_prog_raw:.3f}, p={p_pfi_prog_raw:.3f}")
        print(f"C-index vs Driver因子数量 - Pearson: r={corr_driver_raw:.3f}, p={p_driver_raw:.3f}")
        print(f"C-index vs Protector因子数量 - Pearson: r={corr_protector_raw:.3f}, p={p_protector_raw:.3f}")
        print(f"=== Cox相关性分析（使用原始p值预后因子数量）===")
        print(f"C-index vs Cox预后因子总数 - Pearson: r={corr_cox_prog_raw:.3f}, p={p_cox_prog_raw:.3f}")
        print(f"对比FDR校正后 - PFI: r={corr_pfi_prog_fdr:.3f}, Cox: r={corr_cox_prog_fdr:.3f}")
    
    # 保存详细结果
    results_dir = os.path.join(args.pfi_dir, args.cancer)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存每个实验的详细结果
    detailed_results_df = pd.DataFrame([{
        'repeat': r['repeat'],
        'fold': r['fold'],
        'cindex': r['cindex'],
        'significant_factors': r['significant_factors'],
        
        # PFI分析结果 - FDR校正后
        'pfi_prognostic_factors_fdr': r['pfi_prognostic_factors_fdr'],
        'pfi_driver_factors_fdr': r['pfi_driver_factors_fdr'],
        'pfi_protector_factors_fdr': r['pfi_protector_factors_fdr'],
        
        # PFI分析结果 - 原始p值
        'pfi_prognostic_factors_raw': r['pfi_prognostic_factors_raw'],
        'pfi_driver_factors_raw': r['pfi_driver_factors_raw'],
        'pfi_protector_factors_raw': r['pfi_protector_factors_raw'],
        
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
    
    # 保存Cox分析的详细结果 (包含原始p值和FDR校正后的计数)
    cox_df = detailed_results_df[['repeat', 'fold', 'cindex', 
                                  'cox_prognostic_factors_fdr', 'cox_risk_factors_fdr', 'cox_protective_factors_fdr',
                                  'cox_prognostic_factors_raw', 'cox_risk_factors_raw', 'cox_protective_factors_raw']].copy()
    cox_df.to_csv(os.path.join(results_dir, f"{args.cancer}_cox_analysis_summary.csv"), index=False)
    
    # 保存每个实验的Cox分析详细结果（包含每个基因的PFI统计信息）
    all_cox_details = []
    for r in all_results:
        if 'cox_results' in r and r['cox_results'] and 'prognostic_factors' in r:
            # 创建基因名到PFI统计信息的映射
            pfi_stats_map = {pf['gene']: pf for pf in r['prognostic_factors']}
            
            for cox_result in r['cox_results']:
                gene_name = cox_result['gene']
                cox_detail = {
                    'repeat': r['repeat'],
                    'fold': r['fold'],
                    'cindex': r['cindex'],
                    **cox_result
                }
                
                # 添加该基因的PFI统计信息
                if gene_name in pfi_stats_map:
                    pfi_stats = pfi_stats_map[gene_name]
                    cox_detail.update({
                        'mean_pfi': pfi_stats['mean_pfi'],
                        'abs_mean_pfi': pfi_stats['abs_mean_pfi'],
                        'pfi_importance': pfi_stats['importance'],
                        'pfi_p_value': pfi_stats['p_value'],
                        'pfi_p_adj': pfi_stats.get('p_adj', None),
                        'pfi_type': pfi_stats.get('type', 'not_significant')
                    })
                
                all_cox_details.append(cox_detail)
    
    if all_cox_details:
        cox_details_df = pd.DataFrame(all_cox_details)
        cox_details_df.to_csv(os.path.join(results_dir, f"{args.cancer}_cox_detailed_results.csv"), index=False)
    
    return stats_summary

def create_correlation_plots(cancer, df, results_dir):
    """创建相关性图（包含PFI和Cox分析）- 🔧 添加拟合直线和统计信息"""
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
    
    # === 第一行：PFI分析结果（使用原始p值预后因子数量）===
    # C-index vs PFI预后因子总数
    plot_with_fit(axes[0,0], df['cindex'], df['pfi_prognostic_factors_raw'], 
                  'C-index', 'PFI Prognostic Factors (Raw p)', 
                  f'{cancer}: C-index vs PFI Prognostic Factors', 'purple')
    
    # C-index vs Driver因子数量
    plot_with_fit(axes[0,1], df['cindex'], df['pfi_driver_factors_raw'], 
                  'C-index', 'PFI Driver Factors (Raw p)', 
                  f'{cancer}: C-index vs PFI Driver Factors', 'red')
    
    # C-index vs Protector因子数量
    plot_with_fit(axes[0,2], df['cindex'], df['pfi_protector_factors_raw'], 
                  'C-index', 'PFI Protector Factors (Raw p)', 
                  f'{cancer}: C-index vs PFI Protector Factors', 'blue')
    
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
    """主函数 - PFI分析 + Cox回归分析"""
    print("🔥 综合分析框架: PFI + Cox回归")
    print("=== PFI分析部分 ===")
    print("   ✅ 基于排列特征重要性分析预后因子")
    print("   ✅ Driver因子: 高风险组PFI更高")
    print("   ✅ Protector因子: 低风险组PFI更高")
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
    os.makedirs(args.pfi_dir, exist_ok=True)
    
    all_cancer_summaries = []
    
    for cancer in cancer_list:
        if not os.path.isdir(os.path.join(args.results_dir, cancer)):
            continue
            
        args.cancer = cancer
        args.cancer_lower = cancer.lower()
        args.cancer_results_dir = os.path.join(args.results_dir, cancer)
        
        num_features = 2000
        print(f"\n{'='*60}")
        print(f"正在处理癌症类型: {cancer} (PFI)")
        print(f"{'='*60}")
        
        try:
            summary = analyze_cancer_type(args, num_features=num_features)
            if summary is not None:
                all_cancer_summaries.append(summary)
                print(f"✅ {cancer} 分析完成")
            else:
                print(f"⚠️ {cancer} 分析失败：返回None")
        except Exception as e:
            print(f"❌ {cancer} 分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存所有癌症的汇总结果
    if all_cancer_summaries:
        summary_df = pd.DataFrame(all_cancer_summaries)
        summary_df.to_csv(os.path.join(args.pfi_dir, "all_cancers_summary.csv"), index=False)
        print(f"\n✅ 所有癌症汇总结果已保存到: {os.path.join(args.pfi_dir, 'all_cancers_summary.csv')}")
        
        print(f"\n{'='*60}")
        print("所有癌症类型汇总 (PFI + Cox分析):")
        print(f"{'='*60}")
        
        # PFI分析汇总 - 主要结果（原始p值）
        print("=== PFI分析汇总（主要结果：使用原始p值预后因子数量）===")
        pfi_cols_raw = ['cancer', 'cindex_mean', 'pfi_prognostic_factors_raw_mean', 
                       'corr_cindex_pfi_prognostic_raw_pearson', 'corr_cindex_pfi_prognostic_raw_pearson_p']
        print(summary_df[pfi_cols_raw].to_string(index=False))
        
        # Cox分析汇总 - 主要结果（原始p值）
        print("\n=== Cox分析汇总（主要结果：使用原始p值预后因子数量）===")
        cox_cols_raw = ['cancer', 'cindex_mean', 'cox_prognostic_factors_raw_mean', 
                       'corr_cindex_cox_prognostic_raw_pearson', 'corr_cindex_cox_prognostic_raw_pearson_p']
        print(summary_df[cox_cols_raw].to_string(index=False))
        
        # 对比：FDR校正后的结果
        print("\n=== 对比：FDR校正后的相关性 ===")
        compare_cols = ['cancer', 'pfi_prognostic_factors_fdr_mean', 'cox_prognostic_factors_fdr_mean',
                       'corr_cindex_pfi_prognostic_fdr_pearson', 'corr_cindex_cox_prognostic_fdr_pearson']
        print(summary_df[compare_cols].to_string(index=False))
        
        # 💡 新增：总体统计
        print(f"\n=== 总体统计 (共{len(all_cancer_summaries)}种癌症) ===")
        print(f"平均C-index: {summary_df['cindex_mean'].mean():.4f} ± {summary_df['cindex_mean'].std():.4f}")
        print(f"平均PFI预后因子(原始p): {summary_df['pfi_prognostic_factors_raw_mean'].mean():.1f} ± {summary_df['pfi_prognostic_factors_raw_mean'].std():.1f}")
        print(f"平均Cox预后因子(原始p): {summary_df['cox_prognostic_factors_raw_mean'].mean():.1f} ± {summary_df['cox_prognostic_factors_raw_mean'].std():.1f}")
        print(f"平均PFI预后因子(FDR): {summary_df['pfi_prognostic_factors_fdr_mean'].mean():.1f} ± {summary_df['pfi_prognostic_factors_fdr_mean'].std():.1f}")
        print(f"平均Cox预后因子(FDR): {summary_df['cox_prognostic_factors_fdr_mean'].mean():.1f} ± {summary_df['cox_prognostic_factors_fdr_mean'].std():.1f}")
        
        # 相关性显著性统计
        pfi_sig_count = (summary_df['corr_cindex_pfi_prognostic_raw_pearson_p'] < 0.05).sum()
        cox_sig_count = (summary_df['corr_cindex_cox_prognostic_raw_pearson_p'] < 0.05).sum()
        print(f"PFI相关性显著的癌症类型数: {pfi_sig_count}/{len(all_cancer_summaries)} ({pfi_sig_count/len(all_cancer_summaries)*100:.1f}%)")
        print(f"Cox相关性显著的癌症类型数: {cox_sig_count}/{len(all_cancer_summaries)} ({cox_sig_count/len(all_cancer_summaries)*100:.1f}%)")
        
        print(f"\n💡 说明:")
        print(f"   🎯 主要分析：使用原始p值（校正前）的预后因子数量计算相关性")
        print(f"   📊 可视化：散点图包含拟合直线、相关系数和显著性检验")
        print(f"   💾 特征重要性：每个实验保存完整的2000个特征排序文件")
        print(f"   PFI预后因子: 基于排列特征重要性在高低风险组中的差异")
        print(f"   - Driver(增加风险): 高风险组PFI更高")
        print(f"   - Protector(降低风险): 低风险组PFI更高")
        print(f"   Cox预后因子: 基于单因素Cox回归分析结果")
        print(f"   - Risk因子: HR > 1 (p < 0.05)")
        print(f"   - Protective因子: HR < 1 (p < 0.05)")
        print(f"   📈 相关性显著性: * p<0.05, ** p<0.01, *** p<0.001, ns 不显著")
        print(f"   📁 特征重要性文件: {args.pfi_dir}/[癌症类型]/feature_importance/")
    
    print(f"\nPFI Individual 分析完成！处理了{len(all_cancer_summaries)}种癌症类型")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单独分析每个实验的PFI值和预后因子")
    parser.add_argument('--csv_path', type=str, required=True, help='CSV数据路径')
    parser.add_argument('--results_dir', type=str, required=True, help='Nested CV结果目录')
    parser.add_argument('--pfi_dir', type=str, required=True, help='PFI分析结果保存目录')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    
    args = parser.parse_args()
    main(args) 