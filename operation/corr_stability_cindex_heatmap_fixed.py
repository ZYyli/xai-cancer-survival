"""
相关性可视化：C-index vs XAI 稳定性 (Nested CV)
- 图2：整体相关性散点图（90个点：15癌症 × 6 XAI）
- 图3：按XAI方法分组的箱线图（展示相关系数分布）
- 扇形热图：补充材料
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ============================== 数据路径配置 ==============================
CANCER_LIST = ['LGG', 'GBMLGG', 'KIRP', 'UCEC', 'KIRC', 'SKCM', 'LIHC', 'PAAD', 
               'LUAD', 'BRCA', 'BLCA', 'STAD', 'HNSC', 'LUSC', 'COADREAD']

# XAI方法配置: {stability_dir_name: (display_name, results_dir, feature_subdir, file_prefix)}
XAI_METHODS = {
    'LRP': ('LRP', '/home/zuoyiyi/SNN/TCGA/LRP_results_2', 'lrp_feature_importance', 'lrp'),
    'PFI': ('PFI', '/home/zuoyiyi/SNN/TCGA/PFI_results_2', 'pfi_feature_importance', 'pfi'),
    'shap': ('G-SHAP', '/home/zuoyiyi/SNN/TCGA/shap_results_2', 'shap_feature_importance', 'shap'),
    'IG': ('IG', '/home/zuoyiyi/SNN/TCGA/IG_results_2', 'ig_feature_importance', 'ig'),
    'deepshap': ('D-SHAP', '/home/zuoyiyi/SNN/TCGA/deepshap_results_2', 'deepshap_feature_importance', 'deepshap'),
    'DeepLIFT': ('DeepLIFT', '/home/zuoyiyi/SNN/TCGA/DeepLIFT_results_2', 'deeplift_feature_importance', 'deeplift')
}

STABILITY_DIR = "/home/zuoyiyi/SNN/TCGA/stability_analysis_3"
OUTPUT_DIR = "/home/zuoyiyi/SNN/TCGA/stability_comparison_plots_nestedcv"

TOP_K = 100

# ============================== 颜色配置 ==============================
HEATMAP_COLORS = ["#43A8A8", "white", "#D75F5F"]

# XAI方法颜色
XAI_COLORS = {
    'LRP': '#A6DAEF',
    'PFI': '#E26472',
    'G-SHAP': '#D3C0A3',
    'IG': '#F49E39',
    'D-SHAP': '#9271B1',
    'DeepLIFT': '#66c1a4'
}


def calculate_kuncheva_index(set1, set2, k, m):
    """计算Kuncheva指数（机会校正的Jaccard）"""
    intersection = len(set1.intersection(set2))
    expected_overlap = k * k / m
    denominator = k - expected_overlap
    
    if abs(denominator) < 1e-10:
        return 1.0 if intersection == k else 0.0
    
    return (intersection - expected_overlap) / denominator


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正"""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q
    p_valid = p[mask]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = float(len(ranked))
    q_ranked = ranked * (m / (np.arange(1, len(ranked) + 1)))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    q[mask] = q_ranked[inv]
    return q


def load_jaccard_correlation_data():
    """
    从 performance_correlations.csv 读取 Jaccard vs C-index 的 Spearman 相关系数
    """
    xai_names = [v[0] for v in XAI_METHODS.values()]  # display names
    xai_dirs = list(XAI_METHODS.keys())  # stability dir names
    
    corr_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    p_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    
    print("=" * 60)
    print(f"读取 Jaccard Stability vs C-index 相关性 (Top-{TOP_K})")
    print("=" * 60)
    
    for j, (xai_dir, xai_info) in enumerate(XAI_METHODS.items()):
        xai_name = xai_info[0]
        for i, cancer in enumerate(CANCER_LIST):
            perf_file = os.path.join(STABILITY_DIR, cancer, xai_dir, "performance_correlations.csv")
            
            if not os.path.exists(perf_file):
                continue
            
            try:
                df = pd.read_csv(perf_file)
                row = df[df['top_k'] == TOP_K]
                
                if len(row) > 0:
                    corr_matrix[i, j] = row['spearman_r'].values[0]
                    p_matrix[i, j] = row['spearman_p'].values[0]
            except Exception as e:
                print(f"⚠️ 读取 {cancer}/{xai_dir} 失败: {e}")
        
        n_valid = (~np.isnan(corr_matrix[:, j])).sum()
        print(f"✅ {xai_name}: 读取 {n_valid}/{len(CANCER_LIST)} 个癌症")
    
    # 使用 FDR 校正计算显著性 (q < 0.05)
    p_flat = p_matrix.flatten()
    q_flat = _bh_fdr(p_flat)
    q_matrix = q_flat.reshape(p_matrix.shape)
    sig_matrix = (q_matrix < 0.05) & np.isfinite(q_matrix)
    
    df_corr = pd.DataFrame(corr_matrix, index=CANCER_LIST, columns=xai_names)
    df_p = pd.DataFrame(p_matrix, index=CANCER_LIST, columns=xai_names)
    df_sig = pd.DataFrame(sig_matrix, index=CANCER_LIST, columns=xai_names)
    
    return df_corr, df_p, df_sig

def compute_kuncheva_correlation_data():
    """
    计算 Kuncheva Index vs C-index 的 Spearman 相关系数
    从 XAI 原始结果目录读取 ranking 文件
    """
    xai_names = [v[0] for v in XAI_METHODS.values()]  # display names
    
    corr_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    p_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    
    print("\n" + "=" * 60)
    print(f"计算 Kuncheva Index vs C-index 相关性 (Top-{TOP_K})")
    print("=" * 60)
    
    for j, (xai_dir, xai_info) in enumerate(XAI_METHODS.items()):
        xai_name, results_dir, feature_subdir, file_prefix = xai_info
        
        for i, cancer in enumerate(CANCER_LIST):
            # 构建 ranking 文件目录路径
            # 例如: /home/zuoyiyi/SNN/TCGA/DeepLIFT_results_2/BLCA/deeplift_feature_importance/
            ranking_dir = os.path.join(results_dir, cancer, feature_subdir)
            
            if not os.path.exists(ranking_dir):
                continue
            
            try:
                # 读取所有 bootstrap 的特征排名
                rankings = {}
                cindex_values = {}
                m = None  # 总特征数
                
                # 查找所有 ranking 文件: repeat{X}_fold{Y}_{prefix}_feature_importance_ranking.csv
                for f in os.listdir(ranking_dir):
                    if f.startswith('repeat') and f.endswith(f'_{file_prefix}_feature_importance_ranking.csv'):
                        # 提取 repeat 和 fold 号: repeat0_fold1_lrp_feature_importance_ranking.csv -> (0, 1)
                        parts = f.split('_')
                        try:
                            repeat_num = int(parts[0].replace('repeat', ''))
                            fold_num = int(parts[1].replace('fold', ''))
                            # 计算索引: repeat * 5 + fold (假设每个repeat有5个fold)
                            idx = repeat_num * 5 + fold_num
                        except (ValueError, IndexError):
                            continue
                        
                        df_rank = pd.read_csv(os.path.join(ranking_dir, f))
                        
                        # 获取总特征数
                        if m is None:
                            m = len(df_rank)
                        
                        # 提取 Top-K 特征集合
                        rankings[idx] = set(df_rank.head(TOP_K)['feature_name'].tolist())
                
                # 读取 C-index 数据
                cindex_file = f"/home/zuoyiyi/SNN/TCGA/results_bootstrap/{cancer}/cindex_array.npy"
                if os.path.exists(cindex_file):
                    cindex_arr = np.load(cindex_file)
                    for idx in rankings.keys():
                        if idx < len(cindex_arr):
                            cindex_values[idx] = cindex_arr[idx]
                
                if len(rankings) < 3 or len(cindex_values) < 3 or m is None:
                    continue
                
                # 计算每个 bootstrap 的平均 Kuncheva 指数
                experiment_data = []
                for idx, top_k_set in rankings.items():
                    if idx not in cindex_values:
                        continue
                    
                    other_indices = [i for i in rankings.keys() if i != idx]
                    kuncheva_scores = []
                    
                    for other_idx in other_indices:
                        ki = calculate_kuncheva_index(top_k_set, rankings[other_idx], TOP_K, m)
                        kuncheva_scores.append(ki)
                    
                    if kuncheva_scores:
                        avg_kuncheva = np.mean(kuncheva_scores)
                        experiment_data.append({
                            'cindex': cindex_values[idx],
                            'avg_kuncheva': avg_kuncheva
                        })
                
                # 计算 Spearman 相关性
                if len(experiment_data) >= 3:
                    df_exp = pd.DataFrame(experiment_data)
                    rho, p_val = stats.spearmanr(df_exp['cindex'], df_exp['avg_kuncheva'])
                    corr_matrix[i, j] = rho
                    p_matrix[i, j] = p_val
                    
            except Exception as e:
                print(f"⚠️ 计算 {cancer}/{xai_name} 失败: {e}")
        
        n_valid = (~np.isnan(corr_matrix[:, j])).sum()
        print(f"✅ {xai_name}: 计算 {n_valid}/{len(CANCER_LIST)} 个癌症")
    
    # 使用 FDR 校正计算显著性 (q < 0.05)
    p_flat = p_matrix.flatten()
    q_flat = _bh_fdr(p_flat)
    q_matrix = q_flat.reshape(p_matrix.shape)
    sig_matrix = (q_matrix < 0.05) & np.isfinite(q_matrix)
    
    df_corr = pd.DataFrame(corr_matrix, index=CANCER_LIST, columns=xai_names)
    df_p = pd.DataFrame(p_matrix, index=CANCER_LIST, columns=xai_names)
    df_sig = pd.DataFrame(sig_matrix, index=CANCER_LIST, columns=xai_names)
    
    return df_corr, df_p, df_sig


def prepare_stability_cindex_data(metric_name='Jaccard'):
    """
    准备稳定性-Cindex数据
    对每个癌症-XAI组合，计算稳定性与C-index的关系
    返回: DataFrame with columns ['cancer', 'xai_method', 'stability', 'cindex']
    """
    data_list = []
    total_combinations = len(XAI_METHODS) * len(CANCER_LIST)
    processed = 0
    
    print(f"\n处理 {total_combinations} 个癌症-XAI组合...")
    
    for j, (xai_dir, xai_info) in enumerate(XAI_METHODS.items()):
        xai_name, results_dir, feature_subdir, file_prefix = xai_info
        
        for cancer in CANCER_LIST:
            processed += 1
            
            ranking_dir = os.path.join(results_dir, cancer, feature_subdir)
            cindex_file = f"/home/zuoyiyi/SNN/TCGA/results_bootstrap/{cancer}/cindex_array.npy"
            
            # 检查文件是否存在
            if not os.path.exists(ranking_dir) or not os.path.exists(cindex_file):
                print(f"  [{processed}/{total_combinations}] ⚠️  {cancer}/{xai_name}: 文件不存在")
                continue
            
            try:
                rankings = {}
                m = None
                
                for f in os.listdir(ranking_dir):
                    if f.startswith('repeat') and f.endswith(f'_{file_prefix}_feature_importance_ranking.csv'):
                        parts = f.split('_')
                        try:
                            repeat_num = int(parts[0].replace('repeat', ''))
                            fold_num = int(parts[1].replace('fold', ''))
                            idx = repeat_num * 5 + fold_num
                        except (ValueError, IndexError):
                            continue
                        
                        df_rank = pd.read_csv(os.path.join(ranking_dir, f))
                        if m is None:
                            m = len(df_rank)
                        rankings[idx] = set(df_rank.head(TOP_K)['feature_name'].tolist())
                
                # 读取C-index（一次性读取）
                cindex_arr = np.load(cindex_file)
                
                if len(rankings) < 3 or m is None:
                    print(f"  [{processed}/{total_combinations}] ⚠️  {cancer}/{xai_name}: 数据不足 ({len(rankings)} experiments)")
                    continue
                
                n_valid = 0
                for idx, top_k_set in rankings.items():
                    if idx >= len(cindex_arr):
                        continue
                    
                    other_indices = [i for i in rankings.keys() if i != idx]
                    stability_scores = []
                    
                    for other_idx in other_indices:
                        if metric_name == 'Kuncheva':
                            score = calculate_kuncheva_index(top_k_set, rankings[other_idx], TOP_K, m)
                        else:
                            intersection = len(top_k_set.intersection(rankings[other_idx]))
                            union = len(top_k_set.union(rankings[other_idx]))
                            score = intersection / union if union > 0 else 0.0
                        stability_scores.append(score)
                    
                    if stability_scores:
                        avg_stability = np.mean(stability_scores)
                        data_list.append({
                            'cancer': cancer,
                            'xai_method': xai_name,
                            'stability': avg_stability,
                            'cindex': cindex_arr[idx]
                        })
                        n_valid += 1
                
                print(f"  [{processed}/{total_combinations}] ✅ {cancer}/{xai_name}: {n_valid} 个数据点")
                        
            except Exception as e:
                print(f"  [{processed}/{total_combinations}] ❌ {cancer}/{xai_name}: {e}")
    
    # 统计每个组合的数据点数
    df_result = pd.DataFrame(data_list)
    if len(df_result) > 0:
        print("\n" + "=" * 60)
        print("数据点统计（按癌症-XAI组合）")
        print("=" * 60)
        summary = df_result.groupby(['cancer', 'xai_method']).size().reset_index(name='count')
        summary_pivot = summary.pivot(index='cancer', columns='xai_method', values='count')
        print(summary_pivot.to_string())
        
        # 找出缺失的组合
        missing = summary[summary['count'] < 100]
        if len(missing) > 0:
            print("\n⚠️  数据点少于100的组合：")
            for _, row in missing.iterrows():
                print(f"  {row['cancer']}/{row['xai_method']}: {row['count']} 个数据点 (缺失 {100 - row['count']})")
        
        total_expected = len(CANCER_LIST) * len(XAI_METHODS) * 100
        total_actual = len(df_result)
        print(f"\n总计: {total_actual}/{total_expected} 个数据点 (缺失 {total_expected - total_actual})")
    
    return df_result


def create_overall_scatter_plot(df_data, output_path, metric_name):
    """
    图2：整体相关性散点图
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    xai_methods = df_data['xai_method'].unique()
    
    for xai in xai_methods:
        xai_data = df_data[df_data['xai_method'] == xai]
        ax.scatter(xai_data['cindex'], xai_data['stability'],
                   label=xai, color=XAI_COLORS.get(xai, 'gray'),
                   alpha=0.6, s=80, edgecolors='white', linewidths=1.2)
    
    if len(df_data) > 0:
        from scipy.stats import spearmanr, kendalltau
        
        rho_spearman, p_spearman = spearmanr(df_data['cindex'], df_data['stability'])
        tau_kendall, p_kendall = kendalltau(df_data['cindex'], df_data['stability'])
        
        z = np.polyfit(df_data['cindex'], df_data['stability'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_data['cindex'].min(), df_data['cindex'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.6, linewidth=2.5, zorder=1)
        
        textstr = f"Spearman ρ = {rho_spearman:.3f} (p = {p_spearman:.3e})\n"
        textstr += f"Kendall τ = {tau_kendall:.3f} (p = {p_kendall:.3e})\n"
        textstr += f"n = {len(df_data)} points"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Model Performance (C-index)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{metric_name} Stability', fontsize=13, fontweight='bold')
    ax.set_title(f'Overall Relationship: C-index vs {metric_name} Stability\n(15 Cancers × 6 XAI Methods)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fontsize=9, 
              title='XAI Method', title_fontsize=10, ncol=2)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ 整体相关性散点图已保存: {output_path}")
    plt.close()


def create_xai_grouped_boxplot(df_corr, output_path, metric_name):
    """
    图3：按XAI方法分组的箱线图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_list = []
    for cancer in df_corr.index:
        for xai in df_corr.columns:
            val = df_corr.loc[cancer, xai]
            if not np.isnan(val):
                data_list.append({'XAI Method': xai, 'Correlation': val, 'Cancer': cancer})
    
    df_plot = pd.DataFrame(data_list)
    
    xai_order = ['LRP', 'PFI', 'G-SHAP', 'IG', 'D-SHAP', 'DeepLIFT']
    colors = [XAI_COLORS[xai] for xai in xai_order]
    
    bp = ax.boxplot([df_plot[df_plot['XAI Method'] == xai]['Correlation'].values 
                      for xai in xai_order],
                     positions=range(len(xai_order)), widths=0.6, patch_artist=True,
                     showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for i, xai in enumerate(xai_order):
        xai_data = df_plot[df_plot['XAI Method'] == xai]['Correlation'].values
        x = np.random.normal(i, 0.04, size=len(xai_data))
        ax.scatter(x, xai_data, alpha=0.4, s=30, color='black', zorder=3)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(xai_order)))
    ax.set_xticklabels(xai_order, fontsize=11, fontweight='bold')
    ax.set_xlabel('XAI Method', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{metric_name} Stability - C-index Correlation (Spearman ρ)', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Correlation Coefficients across 15 Cancer Types\n({metric_name} Stability vs C-index)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    summary_text = "Mean ρ per XAI:\n"
    for xai in xai_order:
        xai_data = df_plot[df_plot['XAI Method'] == xai]['Correlation'].values
        mean_val = np.mean(xai_data)
        summary_text += f"{xai}: {mean_val:.3f}\n"
    
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.98, 0.02, summary_text.strip(), transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ XAI分组箱线图已保存: {output_path}")
    plt.close()


def create_fan_heatmap(df_corr, df_sig, output_path, title, metric_name):
    """
    创建扇形热图
    - 横向（扇形方向）：癌症类型
    - 纵向（径向方向）：XAI方法
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_position([0.1, 0.15, 0.9, 0.8])
    
    cancers = df_corr.index.tolist()
    xai_methods = df_corr.columns.tolist()
    
    N_cancers = len(cancers)
    N_xai = len(xai_methods)
    
    # 扇形角度设置（180度扇形）
    total_fan_angle_deg = 180
    total_fan_angle_rad = np.deg2rad(total_fan_angle_deg)
    start_angle = 0
    end_angle = np.pi
    
    # 每个癌症的角度位置
    theta = np.linspace(start_angle, end_angle, N_cancers + 1)
    theta_centers = (theta[:-1] + theta[1:]) / 2
    bar_width = (total_fan_angle_rad / N_cancers) * 0.95
    
    # 每个XAI方法的径向位置
    r_inner = 5.8
    r_step = 1.6
    radii = [r_inner + i * r_step for i in range(N_xai)]
    
    # 创建颜色映射
    cmap = LinearSegmentedColormap.from_list("corr_cmap", HEATMAP_COLORS)
    norm = plt.Normalize(vmin=-1, vmax=1)
    
    # 绘制扇形热图格子
    for j, xai_name in enumerate(xai_methods):
        r = radii[j]
        for i, cancer in enumerate(cancers):
            val = df_corr.loc[cancer, xai_name]
            is_sig = df_sig.loc[cancer, xai_name]
            
            if np.isnan(val):
                color = 'lightgray'
            else:
                color = cmap(norm(val))
            
            ax.bar(theta_centers[i], r_step * 0.92, width=bar_width, bottom=r, 
                   color=color, edgecolor='white', linewidth=0.5)
            
            if not np.isnan(val):
                sig_marker = '*' if is_sig else ''
                text_val = f'{val:.2f}{sig_marker}'
                text_r = r + r_step * 0.5
                ax.text(theta_centers[i], text_r, text_val, ha='center', va='center',
                        fontsize=7, fontweight='bold', rotation=0)
    
    # 添加癌症类型标签
    label_radius = radii[-1] + r_step + 0.2
    for i, cancer in enumerate(cancers):
        angle = theta_centers[i]
        angle_deg = np.rad2deg(angle)
        rotation = angle_deg - 90
        ax.text(angle, label_radius, cancer, ha='center', va='center',
                fontsize=9, fontweight='bold', rotation=rotation)
    
    # 添加XAI方法标签
    for j, xai_name in enumerate(xai_methods):
        r = radii[j] + r_step * 0.5
        label_angle = np.pi + 0.12
        ax.text(label_angle, r, xai_name, ha='center', va='center',
                fontsize=9, fontweight='bold', rotation=45)
    
    # 隐藏极坐标元素
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_ylim(0, radii[-1] + r_step + 0.8)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.24, 0.6, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Spearman ρ', fontsize=11, fontweight='bold')
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    
    # 显著性图例
    fig.text(0.5, 0.35, '* p < 0.05', fontsize=11, va='center')
    
    # 标题
    fig.text(0.55, 0.76, title, fontsize=13, fontweight='bold', ha='center', va='bottom')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ {metric_name} 热图已保存: {output_path}")
    plt.close()


def print_statistics(df_corr, df_sig, metric_name):
    """打印统计摘要"""
    n_sig_positive = ((df_corr > 0) & df_sig).sum().sum()
    n_sig_negative = ((df_corr < 0) & df_sig).sum().sum()
    n_total = (~df_corr.isna()).sum().sum()
    
    print(f"\n=== {metric_name} 统计摘要 ===")
    print(f"总计算组合数: {n_total}")
    print(f"显著正相关数: {n_sig_positive}")
    print(f"显著负相关数: {n_sig_negative}")
    if n_total > 0:
        print(f"显著比例: {(n_sig_positive + n_sig_negative) / n_total * 100:.1f}%")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==================== Jaccard Stability ====================
    df_jaccard_corr, df_jaccard_sig = load_jaccard_correlation_data()
    
    print("\n=== Jaccard Stability vs C-index Spearman 相关系数矩阵 ===")
    print(df_jaccard_corr.round(3).to_string())
    
    # 保存 Jaccard 结果
    df_jaccard_corr.to_csv(os.path.join(OUTPUT_DIR, "jaccard_stability_cindex_correlation.csv"))
    df_jaccard_sig.to_csv(os.path.join(OUTPUT_DIR, "jaccard_stability_cindex_significance.csv"))
    
    # 准备Jaccard稳定性-Cindex数据
    print("\n" + "=" * 60)
    print("准备Jaccard稳定性数据...")
    print("=" * 60)
    df_jaccard_data = prepare_stability_cindex_data(metric_name='Jaccard')
    print(f"\n✅ 总计准备了 {len(df_jaccard_data)} 个数据点")
    
    # 图2：整体相关性散点图
    create_overall_scatter_plot(
        df_jaccard_data,
        os.path.join(OUTPUT_DIR, f"fig2_overall_scatter_jaccard_cindex_top{TOP_K}.png"),
        "Jaccard"
    )
    
    # 图3：按XAI方法分组的箱线图
    create_xai_grouped_boxplot(
        df_jaccard_corr,
        os.path.join(OUTPUT_DIR, f"fig3_xai_grouped_boxplot_jaccard_top{TOP_K}.png"),
        "Jaccard"
    )
    
    # 绘制 Jaccard 热图（保留原有功能）
    create_fan_heatmap(
        df_jaccard_corr, df_jaccard_sig,
        os.path.join(OUTPUT_DIR, f"fan_heatmap_jaccard_stability_cindex_top{TOP_K}.png"),
        f"Correlation between C-index and Jaccard Stability (Top-{TOP_K})\nacross Cancer Types and XAI Methods",
        "Jaccard"
    )
    print_statistics(df_jaccard_corr, df_jaccard_sig, "Jaccard Stability")
    
    # ==================== Kuncheva Index ====================
    df_kuncheva_corr, df_kuncheva_sig = compute_kuncheva_correlation_data()
    
    print("\n=== Kuncheva Index vs C-index Spearman 相关系数矩阵 ===")
    print(df_kuncheva_corr.round(3).to_string())
    
    # 保存 Kuncheva 结果
    df_kuncheva_corr.to_csv(os.path.join(OUTPUT_DIR, "kuncheva_stability_cindex_correlation.csv"))
    df_kuncheva_sig.to_csv(os.path.join(OUTPUT_DIR, "kuncheva_stability_cindex_significance.csv"))
    
    # 准备Kuncheva稳定性-Cindex数据
    print("\n" + "=" * 60)
    print("准备Kuncheva稳定性数据...")
    print("=" * 60)
    df_kuncheva_data = prepare_stability_cindex_data(metric_name='Kuncheva')
    print(f"\n✅ 总计准备了 {len(df_kuncheva_data)} 个数据点")
    
    # 图2：整体相关性散点图
    create_overall_scatter_plot(
        df_kuncheva_data,
        os.path.join(OUTPUT_DIR, f"fig2_overall_scatter_kuncheva_cindex_top{TOP_K}.png"),
        "Kuncheva"
    )
    
    # 图3：按XAI方法分组的箱线图
    create_xai_grouped_boxplot(
        df_kuncheva_corr,
        os.path.join(OUTPUT_DIR, f"fig3_xai_grouped_boxplot_kuncheva_top{TOP_K}.png"),
        "Kuncheva"
    )
    
    # 绘制 Kuncheva 热图（保留原有功能）
    create_fan_heatmap(
        df_kuncheva_corr, df_kuncheva_sig,
        os.path.join(OUTPUT_DIR, f"fan_heatmap_kuncheva_stability_cindex_top{TOP_K}.png"),
        f"Correlation between C-index and Kuncheva Index (Top-{TOP_K})\nacross Cancer Types and XAI Methods",
        "Kuncheva"
    )
    print_statistics(df_kuncheva_corr, df_kuncheva_sig, "Kuncheva Index")
    
    print("\n" + "=" * 60)
    print("所有图表绘制完成！")
    print("- 图2: 整体相关性散点图 (Fig. 5A候选)")
    print("- 图3: XAI方法分组箱线图")
    print("- 扇形热图 (补充材料)")
    print("=" * 60)


if __name__ == "__main__":
    main()
