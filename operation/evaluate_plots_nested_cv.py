import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import cumulative_dynamic_auc
import math

# 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 可选：强制使用 Arial 字体（BBRC 推荐）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

TCGA_DIR = Path(os.environ.get('TCGA_DIR', Path(__file__).resolve().parents[1])).resolve()

# 癌症类型缩写列表
CANCER_TYPES = [
    'BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 
    'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 
    'SKCM', 'STAD', 'UCEC'
]

def load_nested_cv_results(results_dir=None, n_repeats=10, n_folds=5):
    """加载所有癌症类型的nested CV results文件"""
    if results_dir is None:
        results_dir = str(Path(os.environ.get('RESULTS_DIR', str(TCGA_DIR / 'results_2'))).resolve())
    all_data = []
    
    # 遍历每种癌症类型
    for cancer_type in CANCER_TYPES:
        cancer_dir = os.path.join(results_dir, cancer_type)
        
        # 检查癌症子目录是否存在
        if not os.path.exists(cancer_dir):
            print(f"癌症类型目录 {cancer_dir} 不存在，跳过")
            continue
        
        # 遍历每个repeat和fold
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                file_path = os.path.join(cancer_dir, f'repeat{repeat}_fold{fold}_results.pkl')
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    print(f"文件 {file_path} 不存在，跳过")
                    continue
                    
                try:
                    with open(file_path, 'rb') as f:
                        results = pickle.load(f)
                        test_results = results['test_results']
                        
                        # 提取每个患者的预测结果和真实标签
                        for patient_id, data in test_results.items():
                            risk_score = data['risk']
                            all_data.append({
                                'cancer_type': cancer_type,
                                'repeat': repeat,
                                'fold': fold,
                                'patient_id': patient_id,
                                'risk_score': risk_score,
                                'survival_months': data['survival_months'],
                                'censorship': data['censorship']
                            })
                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {e}")
    
    return pd.DataFrame(all_data)

def compute_dynamic_auc_single(data, times):
    """
    计算单个fold的动态AUC（不绘图）
    
    参数:
    - data: 包含风险评分、生存时间和删失状态的数据框
    - times: 需要计算AUC的时间点列表
    
    返回:
    - mean_auc: 平均AUC值，如果计算失败返回None
    """
    try:
        data = data.copy()
        event_observed = (1 - data['censorship']).astype(bool)
        event_time = data['survival_months'].values
        risk_score = data['risk_score'].values
        
        if len(times) == 0:
            return None
        
        y = np.array([(e, t) for e, t in zip(event_observed, event_time)], 
                    dtype=[('event', bool), ('time', float)])
        
        auc, mean_auc = cumulative_dynamic_auc(y, y, risk_score, times)
        return mean_auc
    except:
        return None

def format_pvalue(p):
    """格式化p值显示"""
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return "<0.01"
    elif p < 0.05:
        return "<0.05"
    else:
        return f"{p:.3f}"

def _pvalue_to_stars(p):
    if p is None or np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''

def load_km_logrank_significance(output_dir, alpha=0.05):
    """从汇总表中读取每癌种整合KM曲线的log-rank p值，并做FDR校正"""
    summary_file = os.path.join(output_dir, 'all_cancers_fold_analysis_summary.csv')
    if not os.path.exists(summary_file):
        return {}

    summary_df = pd.read_csv(summary_file)
    if 'cancer_type' not in summary_df.columns or 'integrated_logrank_p' not in summary_df.columns:
        return {}

    p_raw_map = {}
    for _, row in summary_df.iterrows():
        cancer = row['cancer_type']
        try:
            p_raw = float(row['integrated_logrank_p'])
        except Exception:
            p_raw = np.nan
        p_raw_map[cancer] = p_raw

    cancers = [c for c in CANCER_TYPES if c in p_raw_map and not np.isnan(p_raw_map[c])]
    if not cancers:
        return {c: {'p_value_raw': p_raw_map.get(c, np.nan), 'p_value_fdr': np.nan, 'significance': ''} for c in CANCER_TYPES}

    pvals = [p_raw_map[c] for c in cancers]
    p_fdr_map = {c: np.nan for c in CANCER_TYPES}
    try:
        from statsmodels.stats.multitest import multipletests
        _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        for c, p_fdr in zip(cancers, pvals_fdr):
            p_fdr_map[c] = float(p_fdr)
    except Exception:
        for c in cancers:
            p_fdr_map[c] = float(p_raw_map[c])

    sig = {}
    for c in CANCER_TYPES:
        p_raw = p_raw_map.get(c, np.nan)
        p_fdr = p_fdr_map.get(c, np.nan)
        sig[c] = {
            'p_value_raw': p_raw,
            'p_value_fdr': p_fdr,
            'significance': _pvalue_to_stars(p_raw),
        }
    return sig

def compute_average_auc_curves_all_folds(data, n_repeats=10, n_folds=5):
    """
    计算所有fold的平均AUC曲线
    
    返回:
    - times_years: 时间点（年）
    - auc_curves: 所有fold的AUC曲线
    """
    auc_curves = []
    all_times = []
    
    # 对每个fold计算AUC
    for repeat in range(n_repeats):
        for fold in range(n_folds):
            fold_data = data[(data['repeat'] == repeat) & (data['fold'] == fold)]
            
            if len(fold_data) < 20:
                continue
            
            # 计算动态AUC
            max_time = int(fold_data['survival_months'].max())
            times = np.arange(12, max_time, 12)
            
            if len(times) == 0:
                continue
            
            # 直接计算AUC曲线
            try:
                event_observed = (1 - fold_data['censorship']).astype(bool)
                event_time = fold_data['survival_months'].values
                risk_score = fold_data['risk_score'].values
                
                y = np.array([(e, t) for e, t in zip(event_observed, event_time)],
                           dtype=[('event', bool), ('time', float)])
                
                auc_vals, _ = cumulative_dynamic_auc(y, y, risk_score, times)
                
                auc_curves.append(auc_vals)
                all_times.append(times / 12)  # 转换为年
            except:
                continue
    
    if len(auc_curves) == 0:
        return None, None
    
    # 找到最小公共时间范围
    min_length = min([len(t) for t in all_times])
    
    # 截取到相同长度
    auc_curves_trimmed = [curve[:min_length] for curve in auc_curves]
    times_common = all_times[0][:min_length]
    
    return times_common, np.array(auc_curves_trimmed)

def plot_average_auc_curves_all_folds(data, cancer_type, save_dir):
    """
    绘制基于所有fold的平均AUC曲线及95% Bootstrap置信区间
    
    方法：对每个时间点的50个AUC值，使用Bootstrap方法计算均值的置信区间
    """
    times, auc_curves = compute_average_auc_curves_all_folds(data)
    
    if times is None or len(auc_curves) == 0:
        print(f"  警告：{cancer_type} 无足够数据绘制平均AUC曲线")
        return None
    
    # 计算平均值
    auc_mean = np.mean(auc_curves, axis=0)
    mean_auc_overall = np.mean(auc_mean)
    
    # 对每个时间点计算Bootstrap置信区间
    n_timepoints = len(times)
    auc_ci_lower = np.zeros(n_timepoints)
    auc_ci_upper = np.zeros(n_timepoints)
    
    print(f"    计算AUC曲线的Bootstrap置信区间（{n_timepoints}个时间点）...")
    for i in range(n_timepoints):
        # 该时间点的50个AUC值
        auc_at_timepoint = auc_curves[:, i]
        # Bootstrap置信区间
        ci_lower, ci_upper = bootstrap_ci(auc_at_timepoint, n_bootstrap=1000)
        auc_ci_lower[i] = ci_lower
        auc_ci_upper[i] = ci_upper
    
    # 绘图
    plt.figure(figsize=(7, 5.5))
    sns.set_style("ticks")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 绘制平均AUC曲线和Bootstrap置信区间
    plt.plot(times, auc_mean, marker='o', color='#1B6795', 
             linewidth=2, markersize=5, label='Mean AUC')
    plt.fill_between(times, auc_ci_lower, auc_ci_upper, 
                     color='#1B6795', alpha=0.2, label='95% CI (Bootstrap)')
    
    # 参考线
    plt.axhline(y=0.5, color='#15A45B', linestyle='--', linewidth=2, 
                label='Random Guess')
    plt.axhline(y=mean_auc_overall, color='#E98E1E', linestyle='--', 
                linewidth=2, label=f'Overall Mean: {mean_auc_overall:.3f}')
    
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Time-dependent AUC', fontsize=12, fontweight='bold')
    plt.title(f'{cancer_type} - Average Dynamic AUC\n(Based on {len(auc_curves)} folds)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', frameon=False)
    plt.tight_layout()
    plt.grid(False)
    
    # 保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"cancer_{cancer_type}_dynamic_auc_average_all_folds.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"  ✅ 平均AUC曲线已保存: {file_path}")
    plt.close()
    
    return mean_auc_overall

def plot_integrated_survival_curve(data, cancer_type, save_dir):
    """
    整合所有50个外层测试集，绘制全局KM曲线（推荐方法）
    
    原理：
    - 对于每个患者，先对10次重复的risk_score取平均
    - 然后对所有患者的平均risk_score取中位数作为分层阈值
    - 绘制一条最终的KM曲线
    - 进行一次log-rank检验
    
    这确保：
    1. 每个患者只有一个代表性的risk_score（10次重复的平均）
    2. 所有预测都基于外层测试集（未参与训练）
    3. 避免重复患者影响分组
    
    参数：
    - data: 包含所有repeat-fold数据的DataFrame
    - cancer_type: 癌症类型
    - save_dir: 保存目录
    
    返回：
    - logrank_p: Log-rank检验P值
    """
    # 对每个患者的10次重复预测取平均
    print(f"    原始数据：共 {len(data)} 条记录（来自50个外层测试集）")
    print(f"    唯一患者数: {data['patient_id'].nunique()}")
    
    # 按patient_id分组，对risk_score取平均，其他列取第一个值（因为同一患者的生存信息相同）
    data_averaged = data.groupby('patient_id').agg({
        'risk_score': 'mean',  # 对10次重复的risk_score取平均
        'survival_months': 'first',
        'censorship': 'first',
        'cancer_type': 'first'
    }).reset_index()
    
    print(f"    平均后数据：共 {len(data_averaged)} 个唯一患者")
    
    # 使用平均risk_score的全局中位数作为风险分层阈值
    global_median_risk = data_averaged['risk_score'].median()
    print(f"    全局风险中位数阈值（基于平均risk_score）: {global_median_risk:.4f}")
    
    # 使用平均后的数据划分高低风险组
    data_averaged['risk_group'] = data_averaged['risk_score'].apply(
        lambda x: 'high_risk' if x >= global_median_risk else 'low_risk'
    )
    
    high_risk = data_averaged[data_averaged['risk_group'] == 'high_risk']
    low_risk = data_averaged[data_averaged['risk_group'] == 'low_risk']
    
    print(f"    高风险组: {len(high_risk)} 患者")
    print(f"    低风险组: {len(low_risk)} 患者")
    
    # 转换生存时间：月 → 年
    high_risk = high_risk.copy()
    low_risk = low_risk.copy()
    high_risk['survival_years'] = high_risk['survival_months'] / 12
    low_risk['survival_years'] = low_risk['survival_months'] / 12
    
    # 执行log-rank检验
    results = logrank_test(
        durations_A=high_risk['survival_months'], 
        durations_B=low_risk['survival_months'],
        event_observed_A=(1 - high_risk['censorship']).astype(bool),
        event_observed_B=(1 - low_risk['censorship']).astype(bool)
    )
    
    logrank_p = results.p_value
    print(f"    Log-rank P值: {format_pvalue(logrank_p)}")
    
    # 绘制KM生存曲线
    plt.figure(figsize=(7, 5.5))
    sns.set_style("ticks")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    kmf_high.fit(high_risk['survival_years'], 
                event_observed=(1 - high_risk['censorship']), 
                label='High Risk')
    kmf_low.fit(low_risk['survival_years'], 
               event_observed=(1 - low_risk['censorship']), 
               label='Low Risk')
    
    kmf_high.plot(ci_show=False, color='#E94F1E', drawstyle='steps-post', 
                 linestyle='-', linewidth=2.5, alpha=1.0)
    kmf_low.plot(ci_show=False, color='#1B6795', drawstyle='steps-post', 
                linestyle='-', linewidth=2.5, alpha=1.0)
    
    # 绘制删失点
    censored_high = high_risk[high_risk['censorship'] == 1]
    if not censored_high.empty:
        plt.scatter(
            x=censored_high['survival_years'],
            y=kmf_high.survival_function_at_times(censored_high['survival_years']),
            marker='|', color='#E94F1E', s=100, linewidth=1.0, alpha=0.8
        )
    
    censored_low = low_risk[low_risk['censorship'] == 1]
    if not censored_low.empty:
        plt.scatter(
            x=censored_low['survival_years'],
            y=kmf_low.survival_function_at_times(censored_low['survival_years']),
            marker='|', color='#1B6795', s=100, linewidth=1.0, alpha=0.8
        )
    
    # 设置坐标轴
    max_time = data_averaged['survival_months'].max() / 12
    interval_years = max(1, int(math.ceil(max_time / 4)))
    time_points = [interval_years * i for i in range(1, 5)]
    
    plt.xlim(left=0)
    plt.xlabel('Survival Time (years)', fontsize=12, fontweight='bold')
    plt.xticks(time_points, fontsize=12)
    plt.ylim(bottom=0, top=1.02)
    plt.ylabel('Survival Probability', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    
    # 标题
    plt.title(f'{cancer_type} - Integrated Survival Curve\n(All 50 Outer Test Sets, N={len(data_averaged)} unique patients)', 
              fontsize=14, fontweight='bold', y=1.03)
    
    # 图例
    plt.legend(fontsize=12, loc='best', frameon=False)
    
    # 添加P值文本框
    p_str = format_pvalue(logrank_p)
    if logrank_p < 0.001:
        p_str_with_star = f'{p_str}***'
    elif logrank_p < 0.01:
        p_str_with_star = f'{p_str}**'
    elif logrank_p < 0.05:
        p_str_with_star = f'{p_str}*'
    else:
        p_str_with_star = p_str
    
    plt.text(0.5, 0.90, f'Log-rank P: {p_str_with_star}', 
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8),
            ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.grid(False)
    
    # 保存图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"cancer_{cancer_type}_survival_curve_integrated.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"  ✅ 整合KM曲线已保存: {file_path}")
    plt.close()
    
    return logrank_p


def load_cindex_data_nested_cv(root_folder=None):
    """加载nested CV的C-index数据"""
    if root_folder is None:
        root_folder = str(Path(os.environ.get('RESULTS_DIR', str(TCGA_DIR / 'results_2'))).resolve())
    cindex_data = []
    
    for cancer_folder in os.listdir(root_folder):
        cancer_path = os.path.join(root_folder, cancer_folder)
        if not os.path.isdir(cancer_path):
            continue
        
        cancer_type = cancer_folder
        file_path = os.path.join(cancer_path, "detailed_results.csv")
        
        try:
            df = pd.read_csv(file_path)
            if 'test_cindex' in df.columns:
                df['Cancer_Type'] = cancer_type
                df = df.rename(columns={'test_cindex': 'Concordance Index'})
                cindex_data.append(df[['Cancer_Type', 'Concordance Index', 'repeat', 'fold']])
        except Exception as e:
            print(f"读取 {file_path} 失败: {e}")
            
    if cindex_data:
        return pd.concat(cindex_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Cancer_Type', 'Concordance Index', 'repeat', 'fold'])

def fold_level_analysis(data, cancer_type, cindex_data=None, n_repeats=10, n_folds=5):
    """
    Fold级别分析：计算每个fold的C-index和AUC
    
    参数：
    - data: 包含所有repeat-fold数据的DataFrame
    - cancer_type: 癌症类型
    - cindex_data: 预加载的C-index数据（可选），如果提供则直接使用而不重新计算
    - n_repeats: 重复次数，默认10
    - n_folds: 折数，默认5
    
    返回：
    - fold_results: DataFrame，包含每个fold的分析结果
    """
    fold_results = []
    
    # 对每个fold单独分析
    for repeat in range(n_repeats):
        for fold in range(n_folds):
            fold_data = data[(data['repeat'] == repeat) & (data['fold'] == fold)]
            
            if len(fold_data) < 20:  # 样本量检查
                continue
            
            # 1. C-index：优先使用预加载的数据
            c_index = None
            if cindex_data is not None:
                try:
                    c_index = cindex_data[
                        (cindex_data['Cancer_Type'] == cancer_type) & 
                        (cindex_data['repeat'] == repeat) & 
                        (cindex_data['fold'] == fold)
                    ]['Concordance Index'].values[0]
                except:
                    print(f"    警告：未找到 {cancer_type} repeat{repeat} fold{fold} 的C-index，将设为None")
                    c_index = None

            # 2. 动态AUC计算
            max_time = int(fold_data['survival_months'].max())
            times = np.arange(12, max_time, 12)
            if len(times) > 0:
                auc = compute_dynamic_auc_single(fold_data, times)
            else:
                auc = None
            
            fold_results.append({
                'cancer_type': cancer_type,
                'repeat': repeat,
                'fold': fold,
                'n_samples': len(fold_data),
                'c_index': c_index,
                'mean_auc': auc
            })
    
    return pd.DataFrame(fold_results)


def bootstrap_ci(data, n_bootstrap=1000, alpha=0.05, random_seed=42):
    """
    Bootstrap方法计算均值的置信区间
    
    理论基础：
    - Bootstrap是一种非参数重采样方法
    - 通过重复从样本中有放回抽样，估计统计量的分布
    - 不需要假设数据分布，适合小样本（n=50）
    - 给出的是总体均值的置信区间（而非预测区间）
    
    参数：
    - data: 数据列表或数组（如50个C-index值）
    - n_bootstrap: Bootstrap重采样次数，默认10000
    - alpha: 显著性水平，默认0.05（对应95%置信区间）
    - random_seed: 随机种子，确保可重复性
    
    返回：
    - (ci_lower, ci_upper): 置信区间下界和上界
    """
    data = np.array(data)
    n = len(data)
    bootstrap_means = []
    
    np.random.seed(random_seed)
    for _ in range(n_bootstrap):
        # 有放回抽样
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # 计算置信区间（使用bootstrap分布的分位数）
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1-alpha/2) * 100)
    
    return ci_lower, ci_upper

def cindex_ci_nested_cv(data, alpha=0.05, n_bootstrap=10000):
    """
    使用Bootstrap方法计算C-Index均值的置信区间（⭐推荐用于论文）
    
    理论基础：
    - 每个repeat-fold产生一个独立的C-index值（10 repeats × 5 folds = 50个值）
    - Bootstrap方法通过重采样估计均值的抽样分布
    - 给出的是总体平均C-index的置信区间（统计学严谨）
    - 不需要假设数据分布，对小样本和偏态分布都适用
    
    与分位数法的区别：
    - Bootstrap CI：总体均值有95%概率在此区间（更窄，约为分位数法的1/7）
    - 分位数法：新实验的值有95%概率在此区间（更宽，实际是预测区间）
    
    参数：
    - data: DataFrame，包含'Cancer_Type', 'Concordance Index', 'repeat', 'fold'列
    - alpha: 置信水平，默认为0.05（对应95%置信区间）
    - n_bootstrap: Bootstrap重采样次数，默认10000
    
    返回：
    - 字典，格式为 {癌症类型: (置信区间下限, 置信区间上限)}
    """
    # 按癌症类型分组
    grouped_data = data.groupby('Cancer_Type')['Concordance Index']
    ci_results = {}
    
    for cancer_type, cindex_series in grouped_data:
        # 转换为列表并过滤NaN
        cindex_list = cindex_series.dropna().tolist()
        n_values = len(cindex_list)
        
        print(f"{cancer_type}: 共有 {n_values} 个C-index值")
        
        if n_values < 3:  # 至少需要3个值计算有意义的置信区间
            print(f"  警告：{cancer_type} 只有 {n_values} 个C-index值，置信区间不稳定")
            ci_results[cancer_type] = (np.nan, np.nan)
            continue
        
        # Bootstrap置信区间（推荐方法）
        lower_bound, upper_bound = bootstrap_ci(cindex_list, n_bootstrap=n_bootstrap, alpha=alpha)
        
        # 计算均值用于报告
        mean_cindex = np.mean(cindex_list)
        std_cindex = np.std(cindex_list, ddof=1)  # 样本标准差
        se_cindex = std_cindex / np.sqrt(n_values)  # 标准误差
        
        print(f"  C-index: {mean_cindex:.4f} ± {std_cindex:.4f} (SD)")
        print(f"  标准误差: {se_cindex:.4f}")
        print(f"  95% CI (Bootstrap): [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        ci_results[cancer_type] = (lower_bound, upper_bound)
    
    return ci_results

def test_cindex_vs_random(combined_df):
    """
    对每种癌症的C-index进行单样本t检验，检验是否显著大于0.5（随机水平）
    使用FDR（Benjamini-Hochberg方法）进行多重检验校正
    
    参数：
    - combined_df: 包含所有癌症C-index数据的DataFrame
    
    返回：
    - ttest_results: 字典，格式为 {癌症类型: (t_stat, p_raw, mean_c, std_c, n, p_fdr, sig_fdr)}
    """
    from scipy import stats
    
    print("\n=== C-index vs 随机水平(0.5)的统计检验 ===")
    print("检验方法: 单样本单侧t检验")
    print("H0: μ = 0.5 (模型无预测能力)")
    print("H1: μ > 0.5 (模型优于随机)")
    print("-" * 70)
    
    ttest_results = {}
    p_values_for_correction = []
    cancer_list = []
    
    for cancer_type in CANCER_TYPES:
        cancer_data = combined_df[combined_df['Cancer_Type'] == cancer_type]['Concordance Index']
        cindex_values = cancer_data.dropna().values
        
        if len(cindex_values) < 3:
            print(f"{cancer_type}: 样本量不足 (n={len(cindex_values)})")
            ttest_results[cancer_type] = (np.nan, np.nan, '')
            continue
        
        # 单样本t检验：检验均值是否显著大于0.5
        # alternative='greater' 表示单侧检验（H1: mean > 0.5）
        t_stat, p_value = stats.ttest_1samp(cindex_values, 0.5, alternative='greater')
        
        mean_cindex = np.mean(cindex_values)
        std_cindex = np.std(cindex_values, ddof=1)
        
        # 收集p值用于多重检验校正
        p_values_for_correction.append(p_value)
        cancer_list.append(cancer_type)
        
        # 暂时不标注显著性，等多重检验校正后再标注
        ttest_results[cancer_type] = (t_stat, p_value, mean_cindex, std_cindex, len(cindex_values))
        
        print(f"{cancer_type:8s}: mean={mean_cindex:.4f}±{std_cindex:.4f}, "
              f"t={t_stat:6.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # 多重检验校正（FDR - Benjamini-Hochberg方法）
    if len(p_values_for_correction) > 0:
        from statsmodels.stats.multitest import multipletests
        
        # FDR校正
        reject_fdr, pvals_corrected_fdr, _, _ = multipletests(
            p_values_for_correction, alpha=0.05, method='fdr_bh'
        )
        
        print("\n" + "=" * 70)
        print("多重检验校正结果 (FDR - Benjamini-Hochberg方法):")
        print("=" * 70)
        print(f"{'Cancer':<8s} {'Raw p':<12s} {'FDR p':<12s} {'Significance':<12s}")
        print("-" * 70)
        
        for i, cancer_type in enumerate(cancer_list):
            t_stat, p_raw, mean_c, std_c, n = ttest_results[cancer_type]
            p_fdr = pvals_corrected_fdr[i]
            
            # 根据FDR校正后的p值确定显著性标记
            if p_fdr < 0.001:
                sig_fdr = '***'
            elif p_fdr < 0.01:
                sig_fdr = '**'
            elif p_fdr < 0.05:
                sig_fdr = '*'
            else:
                sig_fdr = ''
            
            # 更新结果，添加FDR校正后的显著性
            ttest_results[cancer_type] = (t_stat, p_raw, mean_c, std_c, n, p_fdr, sig_fdr)
            
            print(f"{cancer_type:<8s} {p_raw:<12.6f} {p_fdr:<12.6f} {sig_fdr:<12s}")
        
        print("-" * 70)
        print(f"FDR校正后显著的癌症: {sum(reject_fdr)}/{len(reject_fdr)} ({sum(reject_fdr)/len(reject_fdr)*100:.1f}%)")
    
    return ttest_results

def plot_overall_average_auc_all_cancers(all_data, save_dir):
    """
    绘制所有癌症的整体平均AUC曲线（Pan-cancer level）
    
    方法：
    - 对每个癌症计算其所有fold的AUC曲线
    - 对每个时间点，合并所有癌症的AUC值
    - 使用Bootstrap方法计算该时间点的均值置信区间
    
    优点：
    - 展示模型的泛癌种预测能力
    - 适合论文中的整体性能总结
    - 每个时间点标注参与计算的癌症/fold数量
    
    参数：
    - all_data: 包含所有癌症数据的DataFrame
    - save_dir: 保存目录
    
    返回：
    - overall_mean_auc: 整体平均AUC值
    """
    print("\n=== 计算泛癌种整体平均AUC曲线 ===")
    
    # 收集所有癌症的AUC曲线
    all_auc_data = {}  # {time_point: [auc_values]}
    cancer_count = {}  # {time_point: set of cancers}
    
    for cancer in CANCER_TYPES:
        cancer_data = all_data[all_data['cancer_type'] == cancer]
        if cancer_data.empty:
            continue
        
        print(f"  处理 {cancer}...")
        times, auc_curves = compute_average_auc_curves_all_folds(cancer_data)
        
        if times is None or len(auc_curves) == 0:
            print(f"    跳过 {cancer}（数据不足）")
            continue
        
        print(f"    {cancer}: {len(auc_curves)} folds, 时间范围 {times[0]:.1f}-{times[-1]:.1f} 年")
        
        # 将该癌症的AUC值添加到对应时间点
        for i, t in enumerate(times):
            t_rounded = round(t, 2)  # 四舍五入到0.01年以匹配时间点
            
            if t_rounded not in all_auc_data:
                all_auc_data[t_rounded] = []
                cancer_count[t_rounded] = set()
            
            # 添加该时间点所有fold的AUC值
            all_auc_data[t_rounded].extend(auc_curves[:, i].tolist())
            cancer_count[t_rounded].add(cancer)
    
    if len(all_auc_data) == 0:
        print("  警告：无足够数据绘制整体平均AUC曲线")
        return None
    
    # 按时间点排序
    time_points = sorted(all_auc_data.keys())
    
    # 计算每个时间点的平均AUC和Bootstrap置信区间
    auc_means = []
    auc_ci_lower = []
    auc_ci_upper = []
    n_cancers_per_time = []
    n_folds_per_time = []
    
    print(f"\n  计算每个时间点的Bootstrap置信区间...")
    for t in time_points:
        auc_values = np.array(all_auc_data[t])
        auc_means.append(np.mean(auc_values))
        
        # Bootstrap置信区间
        ci_lower, ci_upper = bootstrap_ci(auc_values, n_bootstrap=1000)
        auc_ci_lower.append(ci_lower)
        auc_ci_upper.append(ci_upper)
        
        n_cancers_per_time.append(len(cancer_count[t]))
        n_folds_per_time.append(len(auc_values))
    
    # 转换为数组
    time_points = np.array(time_points)
    auc_means = np.array(auc_means)
    auc_ci_lower = np.array(auc_ci_lower)
    auc_ci_upper = np.array(auc_ci_upper)
    overall_mean_auc = np.mean(auc_means)
    
    print(f"\n  整体平均AUC: {overall_mean_auc:.4f}")
    print(f"  时间范围: {time_points[0]:.1f}-{time_points[-1]:.1f} 年")
    print(f"  时间点数: {len(time_points)}")
    print(f"  平均每个时间点：{np.mean(n_cancers_per_time):.1f} 种癌症, {np.mean(n_folds_per_time):.1f} 个folds")
    
    # 绘图（只绘制AUC曲线）
    plt.figure(figsize=(7, 5.5))
    sns.set_style("ticks")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 绘制平均AUC曲线和置信区间
    plt.plot(time_points, auc_means, marker='o', color='#1B6795', 
             linewidth=2.5, markersize=6, label='Mean AUC')
    plt.fill_between(time_points, auc_ci_lower, auc_ci_upper, 
                     color='#1B6795', alpha=0.2, label='95% CI (Bootstrap)')
    
    # 参考线
    plt.axhline(y=0.5, color='#15A45B', linestyle='--', linewidth=2, 
                label='Random Guess')
    plt.axhline(y=overall_mean_auc, color='#E98E1E', linestyle='--', 
                linewidth=2, label=f'Overall Mean: {overall_mean_auc:.3f}')
    
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Time-dependent AUC', fontsize=12, fontweight='bold')
    plt.title('Pan-Cancer Overall Average Dynamic AUC\n(All 15 Cancer Types Combined)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', frameon=False)
    plt.grid(False)
    plt.tight_layout()
    
    # 保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "overall_average_auc_all_cancers.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"  ✅ 整体平均AUC曲线已保存: {file_path}")
        
    plt.close()
    
    # 保存详细数据到CSV
    detail_df = pd.DataFrame({
        'time_years': time_points,
        'mean_auc': auc_means,
        'ci_lower': auc_ci_lower,
        'ci_upper': auc_ci_upper,
        'n_cancers': n_cancers_per_time,
        'n_folds': n_folds_per_time
    })
    detail_file = os.path.join(save_dir, "overall_average_auc_details.csv")
    detail_df.to_csv(detail_file, index=False)
    print(f"  ✅ 详细数据已保存: {detail_file}")
    
    return overall_mean_auc

def draw_cindex_boxplot_with_stars_nested_cv(all_fold_results, output_dir):
    """
    绘制C-index箱线图并标注显著性星号
    - 黑色星号（箱线图上方）: 每癌种整合KM曲线（log-rank）原始p值显著性
    
    参数：
    - all_fold_results: 所有癌症的fold级别分析结果字典 {cancer_type: fold_results_df}
    - output_dir: 输出目录
    """
    print("\n=== C-index置信区间计算（Bootstrap方法）===")
    
    # 读取c-index数据
    combined_df = load_cindex_data_nested_cv()
    print(f"C-index数据结构: {combined_df.shape} (应为 各癌症×50行)")
    print(f"涵盖癌症类型: {len(combined_df['Cancer_Type'].unique())} 种")
    print(f"每种癌症的C-index值数量:")
    print(combined_df.groupby('Cancer_Type').size())

    # 使用Bootstrap方法计算置信区间
    print(f"\n使用Bootstrap方法：通过1000次重采样估计均值的置信区间")
    cindex_ci = cindex_ci_nested_cv(combined_df, n_bootstrap=1000)

    print(f"\n=== C-index 95% Bootstrap置信区间汇总 ===")
    for cancer, (lower, upper) in cindex_ci.items():
        if not np.isnan(lower) and not np.isnan(upper):
            print(f"{cancer}: [{lower:.4f}, {upper:.4f}]")
        else:
            print(f"{cancer}: 数据不足，无法计算置信区间")
    
    # KM整合曲线的log-rank显著性（从汇总表读取）
    km_sig = load_km_logrank_significance(output_dir)

    # 按平均C-index由低到高排序类别
    mean_cindex_per_cancer = (
        combined_df
        .groupby('Cancer_Type')['Concordance Index']
        .mean()
        .sort_values(ascending=True)
    )
    sorted_categories = mean_cindex_per_cancer.index.tolist()
    
    # 绘制箱线图
    plt.figure(figsize=(5.6, 4))
    ax = sns.boxplot(
        x='Cancer_Type', 
        y='Concordance Index', 
        data=combined_df,
        order=sorted_categories,
        color='#135AA9',
        linewidth=1.5,
        flierprops={"marker": "x", "markersize": 5}
    )
    
    # 绘制随机水平参考线（y=0.5）
    ax.axhline(
        y=0.5, 
        color='#D32F2F',  # 红色
        linestyle=':', 
        linewidth=2.0,
        label='Random (C-index=0.5)',
        zorder=2
    )
    
    # 计算并绘制整体平均线
    overall_mean = combined_df['Concordance Index'].mean()  
    ax.axhline(
        y=overall_mean, 
        color='#E98E1E', 
        linestyle='--', 
        linewidth=1.5,
        label=f'Overall: {overall_mean:.3f}',
        zorder=2
    )

    # 获取当前y轴范围
    y_min, y_max = ax.get_ylim()
    # 扩大y轴范围以容纳上方标记
    ax.set_ylim(y_min, y_max + 0.05)
    
    # 遍历每个癌症类型，标注整合KM曲线（log-rank）的显著性
    for i, cancer_type in enumerate(sorted_categories):
        x_pos = i

        sig_stars = km_sig.get(cancer_type, {}).get('significance', '')
        if sig_stars:
            boxprops = ax.artists[i].get_bbox() if i < len(ax.artists) else None
            if boxprops:
                y_max_box = boxprops.y1
            else:
                group_data = combined_df[combined_df['Cancer_Type'] == cancer_type]['Concordance Index']
                y_max_box = group_data.max() if not group_data.empty else 0.8

            star_y_upper = y_max_box + 0.015
            ax.text(x_pos, star_y_upper, sig_stars,
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='black')
    
    # 标题和坐标轴
    plt.title('Concordance Index Distribution for Different Cancers (Nested CV)', 
              fontsize=14, fontweight='bold', y=1.03)
    plt.xlabel('Cancer Type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Concordance Index', fontsize=12, fontweight='bold')
    
    # 图例
    plt.legend(fontsize=11, loc='upper left', frameon=False, facecolor='white', framealpha=0.8)
    plt.tight_layout()
    plt.grid(False)
    
    # 保存箱线图
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'nested_cv_cindex_boxplot.png'), dpi=360, bbox_inches="tight", transparent=True)
    plt.savefig(os.path.join(output_dir, 'nested_cv_cindex_boxplot.tiff'), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, 'nested_cv_cindex_boxplot.pdf'), dpi=600, bbox_inches="tight")
    plt.close()
    
    print("\n=== 保存KM log-rank显著性结果 ===")
    km_summary = []
    for cancer_type in CANCER_TYPES:
        if cancer_type in km_sig:
            km_summary.append({
                'cancer_type': cancer_type,
                'integrated_logrank_p_raw': km_sig[cancer_type].get('p_value_raw', np.nan),
                'integrated_logrank_p_fdr': km_sig[cancer_type].get('p_value_fdr', np.nan),
                'significance': km_sig[cancer_type].get('significance', ''),
            })

    if km_summary:
        km_df = pd.DataFrame(km_summary)
        km_file = os.path.join(output_dir, 'km_logrank_significance_results.csv')
        km_df.to_csv(km_file, index=False)
        print(f"✅ KM log-rank结果已保存至: {km_file}")

def main():
    output_dir = str(Path(os.environ.get('OUTPUT_DIR', str(TCGA_DIR / 'results_nested_cv_plots_1'))).resolve())

    print("=" * 80)
    print("Nested CV (10 repeats × 5 folds) 生存分析评估")
    print("=" * 80)
    print("输出图表（4种核心图表）:")
    print("• ⭐ C-index箱线图：50个fold的分布 + Bootstrap置信区间 + 统计检验")
    print("• ⭐ 整合全局KM曲线：基于所有50个外层测试集 + log-rank检验")
    print("• ⭐ 平均AUC曲线（各癌症）：50个fold的平均 + Bootstrap置信区间")
    print("• ⭐⭐ 泛癌种整体平均AUC曲线：所有15种癌症合并分析（新增）")
    print("=" * 80)
    print("统计方法：")
    print("• 所有置信区间：Bootstrap方法（1000次重采样）⭐ 统计学严谨，适合论文")
    print("• AUC曲线CI：对每个时间点的AUC值独立计算Bootstrap置信区间")
    print("• 泛癌种整体AUC：每个时间点合并所有癌症的AUC值，展示模型泛化能力")
    print("=" * 80)
    
    # 加载所有nested CV结果
    print("\n=== 加载nested CV结果 ===")
    all_data = load_nested_cv_results()
    print(f"总共加载了 {len(all_data)} 个患者记录")
    print(f"涵盖 {len(all_data['cancer_type'].unique())} 种癌症类型")
    print(f"数据结构: {all_data.shape}")
    
    # 验证数据完整性
    repeat_fold_count = all_data[['repeat', 'fold']].drop_duplicates().shape[0]
    expected_combinations = 10 * 5  # 10 repeats × 5 folds
    print(f"Repeat-fold组合数: {repeat_fold_count}/{expected_combinations}")
    if repeat_fold_count < expected_combinations:
        print("警告：某些repeat-fold组合缺失，这可能影响置信区间的稳定性")
    
    # 预加载C-index数据
    print("\n=== 预加载C-index数据 ===")
    cindex_data = load_cindex_data_nested_cv()
    print(f"✅ 成功加载C-index数据: {cindex_data.shape}")
    print(f"   涵盖 {len(cindex_data['Cancer_Type'].unique())} 种癌症类型")
    
    # 对每种癌症类型进行fold级别分析
    print("\n" + "=" * 80)
    print("各癌症类型的Fold级别分析")
    print("=" * 80)
    
    all_fold_results = {}
    all_summary = []
    
    for cancer in CANCER_TYPES:
        print(f"\n{'='*70}")
        print(f"癌症类型: {cancer}")
        print(f"{'='*70}")
        
        # 筛选当前癌症类型的数据
        cancer_data = all_data[all_data['cancer_type'] == cancer]
        if cancer_data.empty:
            print(f"{cancer} 无有效数据，跳过")
            continue
            
        # Fold级别分析（使用预加载的C-index数据）
        fold_results = fold_level_analysis(cancer_data, cancer_type=cancer, cindex_data=cindex_data)
        all_fold_results[cancer] = fold_results
        
        # 保存fold级别详细结果
        fold_results_file = os.path.join(output_dir, f"{cancer}_fold_level_results.csv")
        fold_results.to_csv(fold_results_file, index=False)
        print(f"✅ Fold级别详细结果已保存: {fold_results_file}")
        
        # 统计汇总
        valid_c = fold_results['c_index'].dropna()
        valid_auc = fold_results['mean_auc'].dropna()
        
        print(f"\n📊 分析结果汇总 (基于 {len(fold_results)} 个fold):")
        print(f"  样本量: {fold_results['n_samples'].mean():.1f} ± {fold_results['n_samples'].std():.1f}")
        
        # C-index结果
        if len(valid_c) > 0:
            ci_lower_c, ci_upper_c = bootstrap_ci(valid_c.values, n_bootstrap=1000)
            print(f"\n  【C-index】 (n={len(valid_c)} folds)")
            print(f"    • 均值: {valid_c.mean():.4f} ± {valid_c.std():.4f}")
            print(f"    • 95% CI (Bootstrap): [{ci_lower_c:.4f}, {ci_upper_c:.4f}]")
            print(f"    • 范围: [{valid_c.min():.4f}, {valid_c.max():.4f}]")
        else:
            ci_lower_c, ci_upper_c = np.nan, np.nan
        
        # AUC结果
        if len(valid_auc) > 0:
            ci_lower_auc, ci_upper_auc = bootstrap_ci(valid_auc.values, n_bootstrap=1000)
            print(f"\n  【动态AUC】 (n={len(valid_auc)} folds)")
            print(f"    • 均值: {valid_auc.mean():.4f} ± {valid_auc.std():.4f}")
            print(f"    • 95% CI (Bootstrap): [{ci_lower_auc:.4f}, {ci_upper_auc:.4f}]")
            print(f"    • 范围: [{valid_auc.min():.4f}, {valid_auc.max():.4f}]")
        else:
            ci_lower_auc, ci_upper_auc = np.nan, np.nan
        
        # 整合所有50个外层测试集绘制全局KM曲线
        print(f"\n  【整合全局KM曲线】")
        integrated_logrank_p = plot_integrated_survival_curve(
            cancer_data, cancer, output_dir
        )
        
        # 绘制平均AUC曲线
        print(f"\n  【平均AUC曲线】")
        if len(valid_auc) > 0:
            mean_auc_all_folds = plot_average_auc_curves_all_folds(
                cancer_data, cancer, output_dir
            )
            if mean_auc_all_folds is not None:
                print(f"    平均AUC: {mean_auc_all_folds:.4f}")
        
        # 汇总到总结表
        summary_row = {
            'cancer_type': cancer,
            'n_folds': len(fold_results),
            'integrated_logrank_p': integrated_logrank_p if integrated_logrank_p is not None else np.nan,
            'mean_cindex': valid_c.mean() if len(valid_c) > 0 else np.nan,
            'std_cindex': valid_c.std() if len(valid_c) > 0 else np.nan,
            'ci_lower_cindex': ci_lower_c,  # Bootstrap置信区间
            'ci_upper_cindex': ci_upper_c,  # Bootstrap置信区间
            'mean_auc': valid_auc.mean() if len(valid_auc) > 0 else np.nan,
            'std_auc': valid_auc.std() if len(valid_auc) > 0 else np.nan,
            'ci_lower_auc': ci_lower_auc,  # Bootstrap置信区间
            'ci_upper_auc': ci_upper_auc,  # Bootstrap置信区间
        }
        all_summary.append(summary_row)
    
    # 保存总结表
    summary_df = pd.DataFrame(all_summary)
    summary_file = os.path.join(output_dir, 'all_cancers_fold_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ 所有癌症的汇总结果已保存: {summary_file}")

    # 绘制带星号的 C-index 箱线图
    print("\n" + "=" * 80)
    print("绘制C-index箱线图（星号为整合KM曲线log-rank显著性）")
    print("=" * 80)
    draw_cindex_boxplot_with_stars_nested_cv(all_fold_results, output_dir)

    # 绘制所有癌症的整体平均AUC曲线
    print("\n" + "=" * 80)
    print("绘制泛癌种整体平均AUC曲线（Pan-Cancer Level）")
    print("=" * 80)
    overall_mean_auc = plot_overall_average_auc_all_cancers(all_data, output_dir)
    if overall_mean_auc is not None:
        print(f"\n✅ 泛癌种整体平均AUC: {overall_mean_auc:.4f}")

    print("\n" + "=" * 80)
    print(f"Nested CV Fold级别分析完成！")
    print("=" * 80)
    print(f"✅ 分析了 {len(CANCER_TYPES)} 种癌症类型")
    print(f"✅ 每种癌症分析了最多50个独立fold")
    print(f"✅ 统计学严格：避免模型异质性问题")
    print(f"✅ 所有结果已保存至: {output_dir}")
    print("=" * 80)
    print("\n📋 输出文件清单:")
    print(f"  【汇总结果】")
    print(f"  • all_cancers_fold_analysis_summary.csv - 总体汇总")
    print(f"  • km_logrank_significance_results.csv - KM log-rank统计检验结果")
    print(f"  • nested_cv_cindex_boxplot.png/pdf/tiff - ⭐ C-index箱线图（带显著性标记）")
    print(f"  • overall_average_auc_all_cancers.png - ⭐⭐ 泛癌种整体平均AUC曲线（新增）")
    print(f"  • overall_average_auc_details.csv - 整体AUC曲线详细数据")
    print(f"  【各癌症详细结果（每种癌症3个文件）】")
    print(f"  • {{cancer}}_fold_level_results.csv - fold级别详细结果")
    print(f"  • cancer_{{cancer}}_survival_curve_integrated.png - ⭐ 整合全局KM曲线（基于所有50个外层测试集）")
    print(f"  • cancer_{{cancer}}_dynamic_auc_average_all_folds.png - ⭐ 平均AUC曲线（50 folds + Bootstrap置信区间）")
    print(f"\n  【说明】")
    print(f"  • 所有置信区间均使用Bootstrap方法（统计学严谨，适合论文发表）")
    print(f"  • AUC曲线的置信区间：在每个时间点独立计算，反映该时间点AUC均值的不确定性")
    print(f"  • 泛癌种整体AUC曲线：展示模型在所有15种癌症上的平均预测能力（适合论文摘要/结论）")
    print("=" * 80)
    print(f"📊 总计: 7个汇总文件 + 3×{len(CANCER_TYPES)}个癌症文件 = {7 + 3*len(CANCER_TYPES)}个文件")
    print("=" * 80)

if __name__ == "__main__":
    main() 