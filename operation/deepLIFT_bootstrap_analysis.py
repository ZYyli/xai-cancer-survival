import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from captum.attr import DeepLift
from model_risk import SNN_RISK
from dataset_survival import RNAseqSurvivalDataset
from scipy.stats import ttest_1samp, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

def load_cindex_from_array(results_dir):
    """从bootstrap结果中加载cindex数组"""
    try:
        cindex_file = os.path.join(results_dir, "cindex_array.npy")
        cindex_array = np.load(cindex_file)
        return cindex_array
    except Exception as e:
        print(f"读取cindex数组失败: {e}")
        return None

def perform_cox_analysis(X_data, data_df, top_100_indices, feature_names):
    """对选定特征进行批量单因素Cox分析"""
    print(f"📊 Cox分析数据统计：")
    print(f"   样本总数: {len(data_df)}")
    print(f"   特征总数: {X_data.shape[1] if hasattr(X_data, 'shape') else 'Unknown'}")
    print(f"   将分析Top 100特征")
    
    cox_results = []
    protective_count = 0
    risk_count = 0
    
    if 'survival_months' in data_df.columns and 'censorship' in data_df.columns:
        survival_time = data_df['survival_months'].values
        censorship = data_df['censorship'].values
        event = 1 - censorship  # 翻转编码
        print(f"使用列: survival_months, censorship")
    else:
        print("警告：无法找到生存时间和事件列，跳过Cox分析")
        return [], 0, 0, 0, 0
    
    print(f"Cox分析：生存时间范围 {np.min(survival_time):.2f} - {np.max(survival_time):.2f}")
    print(f"Cox分析：事件数量 {np.sum(event)} / {len(event)} ({np.mean(event)*100:.1f}%)")
    
    # 对每个特征进行单因素Cox分析
    skipped_features = []
    analyzed_count = 0
    
    for idx in top_100_indices:
        feature_name = feature_names[idx]
        
        try:
            # 准备Cox分析数据
            if isinstance(X_data, pd.DataFrame):
                feature_values = X_data.iloc[:, idx].values
            else:
                feature_values = X_data[:, idx]
            
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
                ci_lower = np.exp(cph.confidence_intervals_.loc['feature', '95% lower-bound'])
                ci_upper = np.exp(cph.confidence_intervals_.loc['feature', '95% upper-bound'])
            elif '95% CI lower' in ci_cols and '95% CI upper' in ci_cols:
                ci_lower = np.exp(cph.confidence_intervals_.loc['feature', '95% CI lower'])
                ci_upper = np.exp(cph.confidence_intervals_.loc['feature', '95% CI upper'])
            else:
                ci_lower = np.exp(cph.confidence_intervals_.loc['feature', ci_cols[0]])
                ci_upper = np.exp(cph.confidence_intervals_.loc['feature', ci_cols[1]])
            
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
        print(f"原始p值显著: Risk {risk_count}, Protective {protective_count}")
        print(f"FDR校正显著: Risk {risk_count_fdr}, Protective {protective_count_fdr}")
        
        return cox_results, protective_count_fdr, risk_count_fdr, protective_count, risk_count
    
    return [], 0, 0, 0, 0

def load_model(model_path, input_dim, device):
    """加载已训练好的SNN模型"""
    model = SNN_RISK(omic_input_dim=input_dim, model_size_omic='small', n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

class ModelWrapper(nn.Module):
    """用于Captum DeepLift的模型包装器"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)

def compute_deeplift_and_prognostic_factors(bootstrap_seed, args, num_features, feature_importance_only=False):
    """计算单个bootstrap模型的DeepLIFT值并识别预后因子
    
    Args:
        bootstrap_seed: bootstrap迭代编号
        args: 命令行参数
        num_features: 特征数量
        feature_importance_only: 如果为True，只保存特征重要性排序，不进行后续分析
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== 分析 Bootstrap Seed {bootstrap_seed} ===")
    
    # 检查模型文件
    model_path = os.path.join(args.cancer_results_dir, "models", f"bootstrap_model_seed{bootstrap_seed}.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    # 加载完整数据
    data_csv = os.path.join(args.csv_path, f"{args.cancer_upper}_preprocessed.csv")
    if not os.path.exists(data_csv):
        print(f"数据文件不存在: {data_csv}")
        return None
    
    data_df = pd.read_csv(data_csv)
    dataset = RNAseqSurvivalDataset(data_df, label_col='survival_months', seed=args.seed)
    X_full = dataset.features
    
    if isinstance(X_full, pd.DataFrame):
        feature_names = X_full.columns.tolist()
        X_full_np = X_full.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_full.shape[1])]
        X_full_np = X_full.numpy()
    
    # 加载bootstrap的训练集索引和OOB索引
    train_idx_file = os.path.join(args.cancer_results_dir, "models", f"train_idx_seed{bootstrap_seed}.npy")
    oob_idx_file = os.path.join(args.cancer_results_dir, "models", f"oob_idx_seed{bootstrap_seed}.npy")
    
    if not os.path.exists(train_idx_file) or not os.path.exists(oob_idx_file):
        print(f"索引文件不存在: {train_idx_file} or {oob_idx_file}")
        return None
    
    train_indices = np.load(train_idx_file)
    oob_indices = np.load(oob_idx_file)
    
    print(f"训练集样本数: {len(train_indices)}")
    print(f"OOB样本数: {len(oob_indices)}")
    print(f"总样本数: {len(data_df)}")
    
    # 加载模型
    model = load_model(model_path, input_dim=X_full.shape[1], device=device)
    model.eval()  # 确保模型在评估模式（关闭Dropout）
    
    # 获取训练集数据作为baseline
    X_train = X_full_np[train_indices]
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    # 获取OOB数据进行DeepLIFT分析
    X_oob = X_full_np[oob_indices]
    
    print(f"计算DeepLIFT值...")
    # 创建包装模型
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()
    
    # 计算baseline (训练集均值)
    baseline = torch.mean(X_train_t, dim=0, keepdim=True)
    
    # 限制OOB样本数量以加快计算
    oob_sample_size = min(50, len(X_oob))  # 限制OOB样本数量
    oob_indices_sampled = np.random.choice(len(X_oob), oob_sample_size, replace=False)
    X_oob_sampled = X_oob[oob_indices_sampled]
    X_oob_t = torch.tensor(X_oob_sampled, dtype=torch.float32).to(device)
    
    # 使用DeepLIFT
    deeplift = DeepLift(wrapped_model)
    
    try:
        # 计算attributions (在OOB数据上)
        attributions = deeplift.attribute(X_oob_t, baselines=baseline)
        deeplift_values = attributions.detach().cpu().numpy()
            
    except Exception as e:
        print(f"DeepLIFT计算失败: {e}")
        print(f"尝试备用方法...")
        try:
            # 备用方法：逐个样本计算
            deeplift_values_list = []
            for i in range(len(X_oob_t)):
                single_sample = X_oob_t[i:i+1]
                attr = deeplift.attribute(single_sample, baselines=baseline)
                deeplift_values_list.append(attr.detach().cpu().numpy())
            deeplift_values = np.concatenate(deeplift_values_list, axis=0)
        except Exception as e2:
            print(f"备用方法也失败: {e2}")
            return None
    
    # 计算特征重要性（用于筛选Top 100）
    importance = np.abs(deeplift_values).mean(axis=0)
    
    # === 保存所有2000个特征的重要性排序 ===
    # 创建特征重要性排序DataFrame
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_score': importance,
        'rank': np.arange(1, len(importance) + 1)
    })
    
    # 按重要性降序排序
    feature_importance_df = feature_importance_df.sort_values('importance_score', ascending=False)
    feature_importance_df['rank'] = np.arange(1, len(feature_importance_df) + 1)
    
    # 创建输出目录
    # 输出路径: args.deeplift_dir/[癌症]/deeplift_feature_importance/
    importance_dir = os.path.join(args.deeplift_dir, args.cancer, 'deeplift_feature_importance')
    if not os.path.exists(importance_dir):
        os.makedirs(importance_dir, exist_ok=True)
        print(f"📁 创建目录: {importance_dir}")
    
    # 文件名: seed{X}_deeplift_ranking.csv
    importance_file = os.path.join(importance_dir, f"seed{bootstrap_seed}_deeplift_ranking.csv")
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"✅ DeepLIFT特征重要性排序已保存: {importance_file}")
    print(f"   总特征数: {len(feature_importance_df)}")
    
    # 如果只需要特征重要性排序，直接返回简化结果
    if feature_importance_only:
        print(f"⏩ 跳过后续预后因子识别和Cox分析")
        return {
            'bootstrap_seed': bootstrap_seed,
            'feature_importance_saved': True,
            'importance_file': importance_file,
            'n_features': len(feature_importance_df)
        }
    
    # 识别预后因子 (以Top 100为例)
    top_100_indices = np.argsort(importance)[-100:][::-1]
    
    prognostic_factors = []
    p_values = []
    
    # 正确逻辑：直接基于DeepLIFT值是否显著偏离0
    print(f"使用正确逻辑：基于DeepLIFT值的直接生物学意义")
    print(f"Driver: DeepLIFT均值显著 > 0 (增加风险)")
    print(f"Protector: DeepLIFT均值显著 < 0 (降低风险)")
    
    for i in top_100_indices:
        deeplift_feature = deeplift_values[:, i]  # 该特征在所有OOB样本中的DeepLIFT值
        
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
    print(f"✅ 使用完整数据集进行Cox分析")
    cox_results, cox_protective_count_fdr, cox_risk_count_fdr, cox_protective_count_raw, cox_risk_count_raw = perform_cox_analysis(
        X_full, data_df, top_100_indices, feature_names
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
                else:
                    factor['type'] = 'neutral'     # Should not happen if sig
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
    
    # 从cindex数组中获取对应的cindex
    cindex_array = load_cindex_from_array(args.cancer_results_dir)
    if cindex_array is not None and bootstrap_seed <= len(cindex_array):
        cindex = cindex_array[bootstrap_seed - 1]  # bootstrap_seed从1开始，数组从0开始
    else:
        print(f"无法获取bootstrap {bootstrap_seed}的cindex，使用默认值")
        cindex = 0.5
    
    result = {
        'bootstrap_seed': bootstrap_seed,
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
        'analysis_method': 'Bootstrap_DeepLIFT_plus_Cox_Analysis'
    }
    
    print(f"C-index: {cindex:.4f}")
    print(f"=== DeepLIFT分析结果 ===")
    print(f"统计显著因子数量: {significant_count} (DeepLIFT值显著偏离0)")
    print(f"原始p值显著 - 预后因子总数: {prognostic_count_raw} (Driver: {driver_count_raw} + Protector: {protector_count_raw})")
    print(f"FDR校正显著 - 预后因子总数: {prognostic_count_fdr} (Driver: {driver_count} + Protector: {protector_count})")
    print(f"=== Cox分析结果 ===")
    print(f"原始p值显著 - Cox预后因子总数: {cox_prognostic_count_raw} (Risk: {cox_risk_count_raw} + Protective: {cox_protective_count_raw})")
    print(f"FDR校正显著 - Cox预后因子总数: {cox_prognostic_count_fdr} (Risk: {cox_risk_count_fdr} + Protective: {cox_protective_count_fdr})")
    
    return result

def analyze_bootstrap_cancer(args, num_features, num_bootstraps=100, feature_importance_only=False):
    """分析单个癌症类型的所有bootstrap模型
    
    Args:
        args: 命令行参数
        num_features: 特征数量
        num_bootstraps: bootstrap迭代次数
        feature_importance_only: 如果为True，只保存特征重要性排序
    """
    all_results = []
    
    # bootstrap seeds从1到100
    for bootstrap_seed in range(1, num_bootstraps + 1):
        result = compute_deeplift_and_prognostic_factors(bootstrap_seed, args, num_features, feature_importance_only)
        if result is not None:
            all_results.append(result)
        else:
            print(f"⚠️ Bootstrap seed {bootstrap_seed} 分析失败，跳过")
    
    if not all_results:
        print(f"癌症类型 {args.cancer} 没有成功分析的bootstrap模型")
        return None
    
    print(f"\n{args.cancer} Bootstrap分析总结:")
    print(f"成功分析的bootstrap模型数量: {len(all_results)}")
    
    # 如果只保存特征重要性，返回简化结果
    if feature_importance_only:
        print(f"✅ {args.cancer}: 已保存 {len(all_results)} 个bootstrap的特征重要性排序文件")
        return {
            'cancer': args.cancer,
            'n_bootstrap_models': len(all_results),
            'feature_importance_only': True
        }
    
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
        'n_bootstrap_models': len(all_results),
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
    
    # 相关性分析
    if len(cindices) > 2:
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
    
    # 打印统计摘要
    print(f"\n=== {args.cancer} Bootstrap DeepLIFT 统计摘要 ===")
    print(f"C-index: {stats_summary['cindex_mean']:.4f} ± {stats_summary['cindex_std']:.4f}")
    print(f"FDR校正 - DeepLIFT预后因子: {stats_summary['deeplift_prognostic_factors_fdr_mean']:.1f} ± {stats_summary['deeplift_prognostic_factors_fdr_std']:.1f}")
    print(f"FDR校正 - Cox预后因子: {stats_summary['cox_prognostic_factors_fdr_mean']:.1f} ± {stats_summary['cox_prognostic_factors_fdr_std']:.1f}")
    
    # 创建结果保存目录
    results_dir = os.path.join(args.deeplift_dir, args.cancer)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存详细结果（与其他方法格式一致）
    detailed_results_df = pd.DataFrame([{
        'bootstrap_seed': r['bootstrap_seed'],
        'cindex': r['cindex'],
        'significant_factors': r['significant_factors'],
        'deeplift_prognostic_factors_fdr': r['deeplift_prognostic_factors_fdr'],
        'deeplift_driver_factors_fdr': r['deeplift_driver_factors_fdr'],
        'deeplift_protector_factors_fdr': r['deeplift_protector_factors_fdr'],
        'deeplift_prognostic_factors_raw': r['deeplift_prognostic_factors_raw'],
        'deeplift_driver_factors_raw': r['deeplift_driver_factors_raw'],
        'deeplift_protector_factors_raw': r['deeplift_protector_factors_raw'],
        'cox_prognostic_factors_fdr': r['cox_prognostic_factors_fdr'],
        'cox_risk_factors_fdr': r['cox_risk_factors_fdr'],
        'cox_protective_factors_fdr': r['cox_protective_factors_fdr'],
        'cox_prognostic_factors_raw': r['cox_prognostic_factors_raw'],
        'cox_risk_factors_raw': r['cox_risk_factors_raw'],
        'cox_protective_factors_raw': r['cox_protective_factors_raw']
    } for r in all_results])
    
    detailed_results_df.to_csv(os.path.join(results_dir, f"{args.cancer}_bootstrap_detailed_results.csv"), index=False)
    
    # 生成可视化
    if len(all_results) > 2:
        create_bootstrap_correlation_plots(args.cancer, detailed_results_df, results_dir)
    
    # 保存完整结果
    with open(os.path.join(results_dir, f"{args.cancer}_bootstrap_complete_results.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
    
    # 保存Cox分析的详细结果
    cox_df = detailed_results_df[['bootstrap_seed', 'cindex', 
                                  'cox_prognostic_factors_fdr', 'cox_risk_factors_fdr', 'cox_protective_factors_fdr',
                                  'cox_prognostic_factors_raw', 'cox_risk_factors_raw', 'cox_protective_factors_raw']].copy()
    cox_df.to_csv(os.path.join(results_dir, f"{args.cancer}_bootstrap_cox_analysis_summary.csv"), index=False)
    
    # 保存每个bootstrap的Cox分析详细结果（包含每个基因的DeepLIFT统计信息）
    all_cox_details = []
    for r in all_results:
        if 'cox_results' in r and r['cox_results'] and 'prognostic_factors' in r:
            # 创建基因名到DeepLIFT统计信息的映射
            deeplift_stats_map = {pf['gene']: pf for pf in r['prognostic_factors']}
            
            for cox_result in r['cox_results']:
                gene_name = cox_result['gene']
                cox_detail = {
                    'bootstrap_seed': r['bootstrap_seed'],
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
        cox_details_df.to_csv(os.path.join(results_dir, f"{args.cancer}_bootstrap_cox_detailed_results.csv"), index=False)
    
    print(f"✅ 结果已保存到: {results_dir}")
    
    return stats_summary

def create_bootstrap_correlation_plots(cancer, df, results_dir):
    """创建bootstrap相关性图（包含DeepLIFT和Cox分析）- 🔧 添加拟合直线和统计信息"""
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
                  f'{cancer}: C-index vs DeepLIFT Prognostic Factors (Bootstrap)', 'purple')
    
    # C-index vs Driver因子数量
    plot_with_fit(axes[0,1], df['cindex'], df['deeplift_driver_factors_raw'], 
                  'C-index', 'DeepLIFT Driver Factors (Raw p)', 
                  f'{cancer}: C-index vs DeepLIFT Driver Factors (Bootstrap)', 'red')
    
    # C-index vs Protector因子数量
    plot_with_fit(axes[0,2], df['cindex'], df['deeplift_protector_factors_raw'], 
                  'C-index', 'DeepLIFT Protector Factors (Raw p)', 
                  f'{cancer}: C-index vs DeepLIFT Protector Factors (Bootstrap)', 'blue')
    
    # === 第二行：Cox分析结果（使用原始p值预后因子数量）===
    # C-index vs Cox预后因子总数
    plot_with_fit(axes[1,0], df['cindex'], df['cox_prognostic_factors_raw'], 
                  'C-index', 'Cox Prognostic Factors (Raw p)', 
                  f'{cancer}: C-index vs Cox Prognostic Factors (Bootstrap)', 'green')
    
    # C-index vs Cox风险因子数量
    plot_with_fit(axes[1,1], df['cindex'], df['cox_risk_factors_raw'], 
                  'C-index', 'Cox Risk Factors (Raw p)', 
                  f'{cancer}: C-index vs Cox Risk Factors (Bootstrap)', 'orange')
    
    # C-index vs Cox保护因子数量
    plot_with_fit(axes[1,2], df['cindex'], df['cox_protective_factors_raw'], 
                  'C-index', 'Cox Protective Factors (Raw p)', 
                  f'{cancer}: C-index vs Cox Protective Factors (Bootstrap)', 'cyan')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{cancer}_bootstrap_correlation_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    """主函数"""
    cancer_list = sorted(os.listdir(args.results_dir))
    os.makedirs(args.deeplift_dir, exist_ok=True)
    
    all_cancer_summaries = []
    
    # 确定bootstrap数量
    num_bootstraps = getattr(args, 'num_bootstrap', 300)
    
    for cancer in cancer_list:
        if not os.path.isdir(os.path.join(args.results_dir, cancer)):
            continue
            
        args.cancer = cancer
        args.cancer_upper = cancer.upper()
        args.cancer_results_dir = os.path.join(args.results_dir, cancer)
        
        num_features = 2000
        print(f"\n{'='*60}")
        print(f"正在处理癌症类型: {cancer} (Bootstrap DeepLIFT)")
        if args.feature_importance_only:
            print(f"模式: 仅保存特征重要性排序")
        print(f"{'='*60}")
        
        summary = analyze_bootstrap_cancer(
            args, 
            num_features=num_features, 
            num_bootstraps=num_bootstraps,
            feature_importance_only=args.feature_importance_only
        )
        if summary is not None:
            all_cancer_summaries.append(summary)
    
    # 保存所有癌症的汇总结果
    if all_cancer_summaries:
        all_summaries_df = pd.DataFrame(all_cancer_summaries)
        summary_file = os.path.join(args.deeplift_dir, "all_cancers_bootstrap_summary.csv")
        all_summaries_df.to_csv(summary_file, index=False)
        print(f"\n✅ 所有癌症汇总结果已保存到: {summary_file}")
        
        # 打印总体统计
        if not args.feature_importance_only:
            print(f"\n=== 总体统计 (共{len(all_cancer_summaries)}种癌症) ===")
            print(f"平均C-index: {all_summaries_df['cindex_mean'].mean():.4f}")
            print(f"平均DeepLIFT预后因子(FDR): {all_summaries_df['deeplift_prognostic_factors_fdr_mean'].mean():.1f}")
            print(f"平均Cox预后因子(FDR): {all_summaries_df['cox_prognostic_factors_fdr_mean'].mean():.1f}")
        else:
            print(f"\n=== 特征重要性保存统计 (共{len(all_cancer_summaries)}种癌症) ===")
            total_files = sum([s.get('n_bootstrap_models', 0) for s in all_cancer_summaries])
            print(f"总共保存了 {total_files} 个特征重要性排序文件")
    
    if args.feature_importance_only:
        print(f"\n✅ Bootstrap DeepLIFT特征重要性保存完成！处理了{len(all_cancer_summaries)}种癌症类型")
    else:
        print(f"\n✅ Bootstrap DeepLIFT分析完成！处理了{len(all_cancer_summaries)}种癌症类型")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对Bootstrap模型进行DeepLIFT可解释性分析和预后因子识别")
    parser.add_argument('--csv_path', type=str, required=True, help='Bootstrap数据路径 (preprocess_cancer_single)')
    parser.add_argument('--results_dir', type=str, required=True, help='Bootstrap结果目录 (results_bootstrap)')
    parser.add_argument('--deeplift_dir', type=str, required=True, help='DeepLIFT分析结果保存目录')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--num_bootstrap', type=int, default=300, help='Bootstrap迭代次数 (默认300)')
    parser.add_argument('--feature_importance_only', action='store_true', 
                        help='仅保存特征重要性排序文件，跳过预后因子识别和Cox分析')
    
    args = parser.parse_args()
    main(args)

