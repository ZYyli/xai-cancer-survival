"""
相关性可视化：C-index vs XAI 稳定性

仅生成：Kuncheva Index vs C-index 的扇形热图 + 对应CSV（相关矩阵/显著性矩阵等）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from typing import Optional, Dict, Set
from pathlib import Path

# 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8

# ============================== 数据路径配置 ==============================
CANCER_LIST = ['COADREAD', 'LUSC', 'HNSC', 'STAD', 'BLCA', 'BRCA', 'LUAD', 'PAAD',
    'LIHC', 'SKCM', 'KIRC', 'UCEC', 'KIRP', 'GBMLGG', 'LGG']

TCGA_DIR = Path(os.environ.get('TCGA_DIR', Path(__file__).resolve().parents[1])).resolve()

# XAI方法配置: {stability_dir_name: (display_name, results_dir, feature_subdir, file_prefix)}
XAI_METHODS = {
    'shap': ('G-SHAP', str(TCGA_DIR / 'shap_bootstrap_results'), 'shap_feature_importance', 'shap'),
    'DeepLIFT': ('DeepLIFT', str(TCGA_DIR / 'DeepLIFT_bootstrap_results'), 'deeplift_feature_importance', 'deeplift'),
    'deepshap': ('D-SHAP', str(TCGA_DIR / 'deepshap_bootstrap_results'), 'deepshap_feature_importance', 'deepshap'),
    'IG': ('IG', str(TCGA_DIR / 'IG_bootstrap_results'), 'ig_feature_importance', 'ig'),
    'LRP': ('LRP', str(TCGA_DIR / 'LRP_bootstrap_results'), 'lrp_feature_importance', 'lrp'),
    'PFI': ('PFI', str(TCGA_DIR / 'PFI_bootstrap_results'), 'pfi_feature_importance', 'pfi')
}

STABILITY_DIR = str(Path(os.environ.get('STABILITY_DIR', str(TCGA_DIR / 'stability_analysis_bootstrap'))).resolve())
OUTPUT_DIR = str(Path(os.environ.get('OUTPUT_DIR', str(TCGA_DIR / 'stability_comparison_plots'))).resolve())

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


def load_jaccard_correlation_data():
    """
    从 performance_correlations.csv 读取 Jaccard vs C-index 的 Spearman 相关系数
    """
    xai_names = [v[0] for v in XAI_METHODS.values()]  # display names
    xai_dirs = list(XAI_METHODS.keys())  # stability dir names
    
    corr_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    p_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    sig_matrix = np.full((len(CANCER_LIST), len(xai_names)), False)
    
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


def _read_bootstrap_cox_summary(results_dir: str, cancer: str) -> Optional[pd.DataFrame]:
    summary_path = os.path.join(results_dir, cancer, f"{cancer}_bootstrap_cox_analysis_summary.csv")
    if not os.path.exists(summary_path):
        return None
    df = pd.read_csv(summary_path)
    required = {'bootstrap_seed', 'cindex', 'cox_prognostic_factors_raw'}
    if not required.issubset(df.columns):
        return None
    out = df[['bootstrap_seed', 'cindex', 'cox_prognostic_factors_raw']].copy()
    out['seed_idx'] = out['bootstrap_seed'].astype(int) - 1
    out = out[out['seed_idx'] >= 0]
    return out


def _compute_avg_kuncheva_per_seed(rankings: Dict[int, Set[str]], k: int, m: int) -> Dict[int, float]:
    avg = {}
    seeds = list(rankings.keys())
    if len(seeds) < 3:
        return avg
    for seed in seeds:
        top_k_set = rankings[seed]
        other_seeds = [s for s in seeds if s != seed]
        kuncheva_scores = []
        for other_seed in other_seeds:
            ki = calculate_kuncheva_index(top_k_set, rankings[other_seed], k, m)
            kuncheva_scores.append(ki)
        if kuncheva_scores:
            avg[seed] = float(np.mean(kuncheva_scores))
    return avg


def _linear_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 3 or len(x) < 3:
        return np.full_like(y, np.nan, dtype=float)
    if np.nanstd(x) < 1e-12:
        return np.full_like(y, np.nan, dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return y - (intercept + slope * x)


def compute_prognostic_vs_kuncheva_correlation_data():
    """计算 prognostic factor count vs Kuncheva stability（raw + residual controlling cindex）"""
    xai_names = [v[0] for v in XAI_METHODS.values()]  # display names

    corr_raw = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    p_raw = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    corr_resid = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    p_resid = np.full((len(CANCER_LIST), len(xai_names)), np.nan)

    points_rows = []

    print("\n" + "=" * 60)
    print(f"计算 Prognostic factor count (raw) vs Kuncheva stability 相关性 (Top-{TOP_K})")
    print("=" * 60)

    for j, (xai_dir, xai_info) in enumerate(XAI_METHODS.items()):
        xai_name, results_dir, feature_subdir, file_prefix = xai_info
        for i, cancer in enumerate(CANCER_LIST):
            ranking_dir = os.path.join(results_dir, cancer, feature_subdir)
            if not os.path.exists(ranking_dir):
                continue

            df_summary = _read_bootstrap_cox_summary(results_dir, cancer)
            if df_summary is None or df_summary.empty:
                continue

            try:
                rankings = {}
                m = None

                for f in os.listdir(ranking_dir):
                    if f.startswith('seed') and f.endswith(f'_{file_prefix}_ranking.csv'):
                        seed_str = f.replace(f'_{file_prefix}_ranking.csv', '').replace('seed', '')
                        try:
                            seed = int(seed_str) - 1
                        except ValueError:
                            continue

                        if seed < 0:
                            continue

                        df_rank = pd.read_csv(os.path.join(ranking_dir, f))
                        if m is None:
                            m = len(df_rank)
                        rankings[seed] = set(df_rank.head(TOP_K)['feature_name'].tolist())

                if m is None or len(rankings) < 3:
                    continue

                avg_kuncheva_map = _compute_avg_kuncheva_per_seed(rankings, TOP_K, m)
                if len(avg_kuncheva_map) < 3:
                    continue

                df_join = df_summary[df_summary['seed_idx'].isin(avg_kuncheva_map.keys())].copy()
                if df_join.empty:
                    continue

                df_join['avg_kuncheva'] = df_join['seed_idx'].map(avg_kuncheva_map)
                df_join = df_join.dropna(subset=['cindex', 'cox_prognostic_factors_raw', 'avg_kuncheva'])
                if len(df_join) < 3:
                    continue

                for _, row in df_join.iterrows():
                    points_rows.append({
                        'cancer_type': cancer,
                        'xai_method': xai_name,
                        'bootstrap_seed': int(row['bootstrap_seed']),
                        'seed_idx': int(row['seed_idx']),
                        'cindex': float(row['cindex']),
                        'cox_prognostic_factors_raw': float(row['cox_prognostic_factors_raw']),
                        'avg_kuncheva': float(row['avg_kuncheva'])
                    })

                rho, pv = stats.spearmanr(df_join['cox_prognostic_factors_raw'], df_join['avg_kuncheva'])
                corr_raw[i, j] = rho
                p_raw[i, j] = pv

                x = df_join['cindex'].to_numpy(dtype=float)
                y_count = df_join['cox_prognostic_factors_raw'].to_numpy(dtype=float)
                y_stab = df_join['avg_kuncheva'].to_numpy(dtype=float)

                count_resid = _linear_residuals(y_count, x)
                stab_resid = _linear_residuals(y_stab, x)
                mask = np.isfinite(count_resid) & np.isfinite(stab_resid)
                if mask.sum() >= 3:
                    rho_r, pv_r = stats.spearmanr(count_resid[mask], stab_resid[mask])
                    corr_resid[i, j] = rho_r
                    p_resid[i, j] = pv_r
            except Exception as e:
                print(f"⚠️ 计算 {cancer}/{xai_name} prognostic vs stability 失败: {e}")

        n_valid = (~np.isnan(corr_raw[:, j])).sum()
        print(f"✅ {xai_name}: 计算 {n_valid}/{len(CANCER_LIST)} 个癌症")

    df_corr_raw = pd.DataFrame(corr_raw, index=CANCER_LIST, columns=xai_names)
    df_p_raw = pd.DataFrame(p_raw, index=CANCER_LIST, columns=xai_names)
    df_corr_resid = pd.DataFrame(corr_resid, index=CANCER_LIST, columns=xai_names)
    df_p_resid = pd.DataFrame(p_resid, index=CANCER_LIST, columns=xai_names)
    df_points = pd.DataFrame(points_rows)
    return df_corr_raw, df_p_raw, df_corr_resid, df_p_resid, df_points

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
            # 例如: /DeepLIFT_bootstrap_results/BLCA/deeplift_feature_importance/
            ranking_dir = os.path.join(results_dir, cancer, feature_subdir)
            
            if not os.path.exists(ranking_dir):
                continue
            
            try:
                # 读取所有 bootstrap 的特征排名
                rankings = {}
                cindex_values = {}
                m = None  # 总特征数
                
                # 查找所有 ranking 文件: seed{X}_{prefix}_ranking.csv
                for f in os.listdir(ranking_dir):
                    if f.startswith('seed') and f.endswith(f'_{file_prefix}_ranking.csv'):
                        # 提取 seed 号: seed100_deeplift_ranking.csv -> 100
                        seed_str = f.replace(f'_{file_prefix}_ranking.csv', '').replace('seed', '')
                        try:
                            seed = int(seed_str) - 1  # 将seed1-100映射到索引0-99
                        except ValueError:
                            continue

                        if seed < 0:
                            continue
                        
                        df_rank = pd.read_csv(os.path.join(ranking_dir, f))
                        
                        # 获取总特征数
                        if m is None:
                            m = len(df_rank)
                        
                        # 提取 Top-K 特征集合
                        rankings[seed] = set(df_rank.head(TOP_K)['feature_name'].tolist())
                
                # 读取 C-index 数据
                cindex_file = str(TCGA_DIR / 'results_bootstrap' / cancer / 'cindex_array.npy')
                if os.path.exists(cindex_file):
                    cindex_arr = np.load(cindex_file)
                    for seed in rankings.keys():
                        if seed < len(cindex_arr):
                            cindex_values[seed] = cindex_arr[seed]
                
                if len(rankings) < 3 or len(cindex_values) < 3 or m is None:
                    continue
                
                # 计算每个 bootstrap 的平均 Kuncheva 指数
                experiment_data = []
                for seed, top_k_set in rankings.items():
                    if seed not in cindex_values:
                        continue
                    
                    other_seeds = [s for s in rankings.keys() if s != seed]
                    kuncheva_scores = []
                    
                    for other_seed in other_seeds:
                        ki = calculate_kuncheva_index(top_k_set, rankings[other_seed], TOP_K, m)
                        kuncheva_scores.append(ki)
                    
                    if kuncheva_scores:
                        avg_kuncheva = np.mean(kuncheva_scores)
                        experiment_data.append({
                            'cindex': cindex_values[seed],
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


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
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


def create_fan_heatmap(df_corr, df_sig, output_path, title, metric_name):
    """
    创建矩形热图（与 stability_visualization.py 图3风格一致）
    - X轴：癌症类型
    - Y轴：XAI方法
    """
    cancers = df_corr.index.tolist()
    xai_methods = df_corr.columns.tolist()

    pivot = df_corr.T
    if len(xai_methods) > 0:
        pivot = pivot.reindex(index=xai_methods)
    if len(cancers) > 0:
        pivot = pivot.reindex(columns=cancers)

    annot = pivot.copy().astype(object)
    for m in annot.index:
        for c in annot.columns:
            val = pivot.loc[m, c]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                annot.loc[m, c] = ''
                continue
            is_sig = False
            try:
                is_sig = bool(df_sig.loc[c, m])
            except Exception:
                is_sig = False
            label = '*' if is_sig else ''
            annot.loc[m, c] = f"{float(val):.2f}\n{label}" if label != '' else f"{float(val):.2f}"

    fig, ax = plt.subplots(figsize=(6.89, 3))
    custom_cmap = LinearSegmentedColormap.from_list(
        'corr_colormap',
        ["#1a2a6c", "white", "#b21f1f"],
    )

    sns.heatmap(
        pivot,
        annot=annot.values,
        fmt='',
        cmap=custom_cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        ax=ax,
        cbar_kws={
            'label': 'Spearman ρ',
            'shrink': 0.8,
            'aspect': 20,
            'pad': 0.02,
        },
        linewidths=0.8,
        linecolor='white',
        annot_kws={'size': 8},
    )

    ax.set_ylabel('XAI Method', labelpad=10)
    ax.set_xlabel('Cancer Type', labelpad=10)
    ax.set_title(title, fontsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    fig.text(0.5, 0.02, '* FDR q < 0.05', fontsize=8, ha='center', va='bottom')

    plt.tight_layout()

    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ {metric_name} 热图已保存: {output_path}")
    plt.close()


def create_xai_rho_distribution_plot(df_corr_raw: pd.DataFrame, df_corr_resid: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(2.35, 3.8), sharex=False)

    panels = [
        (axes[0], df_corr_raw, 'raw Spearman ρ (count vs stability)'),
        (axes[1], df_corr_resid, 'residual Spearman ρ (control C-index)')
    ]

    rng = np.random.default_rng(0)
    for panel_idx, (ax, df_corr, title) in enumerate(panels):
        med = df_corr.median(axis=0, skipna=True)
        methods = med.sort_values(ascending=True).index.tolist()
        positions = np.arange(1, len(methods) + 1)
        data = [df_corr[m].dropna().to_numpy(dtype=float) for m in methods]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 0.9},
            whiskerprops={'color': 'black', 'linewidth': 0.9},
            capprops={'color': 'black', 'linewidth': 0.9},
            boxprops={'edgecolor': 'black', 'linewidth': 0.9}
        )
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(XAI_COLORS.get(m, '#cccccc'))
            patch.set_alpha(0.7)

        for idx, ys in enumerate(data):
            if len(ys) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(ys))
            xs = positions[idx] + jitter
            ax.scatter(xs, ys, s=6, color=XAI_COLORS.get(methods[idx], '#444444'), alpha=1, linewidths=0)

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel('Spearman ρ')
        ax.set_title(title, loc='left', fontsize=8)
        ax.set_ylim(-1.05, 1.05)

        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=30)

        # Panel A/B 各自有不同的 XAI 顺序，因此各自显示自己的 X 轴标签
        # Panel A (raw): 标签显示在下方（位于两面板之间）
        # Panel B (resid): 标签显示在最下方
        ax.tick_params(axis='x', labelbottom=True, bottom=True, labeltop=False, top=False)

    fig.tight_layout(h_pad=1.6)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
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

    # ==================== Kuncheva Index (only) ====================
    df_kuncheva_corr, df_kuncheva_p, df_kuncheva_sig = compute_kuncheva_correlation_data()
    
    print("\n=== Kuncheva Index vs C-index Spearman 相关系数矩阵 ===")
    print(df_kuncheva_corr.round(3).to_string())

    q = _bh_fdr(df_kuncheva_p.to_numpy())
    df_kuncheva_q = pd.DataFrame(q, index=df_kuncheva_p.index, columns=df_kuncheva_p.columns)

    # 将热图相关矩阵整合为单个长表CSV（每行一个 cancer × xai_method）
    rows = []
    for cancer in df_kuncheva_corr.index:
        for xai in df_kuncheva_corr.columns:
            r = df_kuncheva_corr.loc[cancer, xai]
            p = df_kuncheva_p.loc[cancer, xai]
            qv = df_kuncheva_q.loc[cancer, xai]
            sig = df_kuncheva_sig.loc[cancer, xai]
            if pd.isna(r) and pd.isna(p) and pd.isna(qv):
                continue
            rows.append({
                'cancer_type': cancer,
                'xai_method': xai,
                'spearman_r': r,
                'p_value': p,
                'q_value_bh': qv,
                'sig_fdr05': bool(sig) if pd.notna(sig) else False
            })

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, f"kuncheva_stability_cindex_heatmap_data_top{TOP_K}.csv")
    df_out.to_csv(out_csv, index=False)

    # 绘制 Kuncheva 热图
    create_fan_heatmap(
        df_kuncheva_corr, df_kuncheva_sig,
        os.path.join(OUTPUT_DIR, f"fan_heatmap_kuncheva_stability_cindex_top{TOP_K}.png"),
        f"Correlation between C-index and Kuncheva Index (Top-{TOP_K})\nacross Cancer Types and XAI Methods",
        "Kuncheva"
    )
    print_statistics(df_kuncheva_corr, df_kuncheva_sig, "Kuncheva Index")

    # ==================== Prognostic factor count (raw) vs Kuncheva stability ====================
    (
        df_pc_corr_raw,
        df_pc_p_raw,
        df_pc_corr_resid,
        df_pc_p_resid,
        df_pc_points
    ) = compute_prognostic_vs_kuncheva_correlation_data()

    # per-seed 点数据（用于复现与可选散点图）
    points_csv = os.path.join(OUTPUT_DIR, f"prognostic_count_vs_kuncheva_points_top{TOP_K}.csv")
    if not df_pc_points.empty:
        df_pc_points.to_csv(points_csv, index=False)

    # FDR q-values and significance
    q_raw = _bh_fdr(df_pc_p_raw.to_numpy())
    df_pc_q_raw = pd.DataFrame(q_raw, index=df_pc_p_raw.index, columns=df_pc_p_raw.columns)
    df_pc_sig_raw = (df_pc_q_raw < 0.05) & df_pc_q_raw.notna()

    q_resid = _bh_fdr(df_pc_p_resid.to_numpy())
    df_pc_q_resid = pd.DataFrame(q_resid, index=df_pc_p_resid.index, columns=df_pc_p_resid.columns)
    df_pc_sig_resid = (df_pc_q_resid < 0.05) & df_pc_q_resid.notna()

    # consolidated long-form CSV: raw + resid
    n_seeds_map = {}
    if not df_pc_points.empty:
        n_seeds_map = (
            df_pc_points.groupby(['cancer_type', 'xai_method'])['seed_idx']
            .nunique()
            .to_dict()
        )

    rows = []
    for cancer in df_pc_corr_raw.index:
        for xai in df_pc_corr_raw.columns:
            n_seeds = int(n_seeds_map.get((cancer, xai), 0))

            r = df_pc_corr_raw.loc[cancer, xai]
            p = df_pc_p_raw.loc[cancer, xai]
            qv = df_pc_q_raw.loc[cancer, xai]
            sig = df_pc_sig_raw.loc[cancer, xai]
            if not (pd.isna(r) and pd.isna(p) and pd.isna(qv)):
                rows.append({
                    'cancer_type': cancer,
                    'xai_method': xai,
                    'corr_type': 'raw',
                    'spearman_r': r,
                    'p_value': p,
                    'q_value_bh': qv,
                    'sig_fdr05': bool(sig) if pd.notna(sig) else False,
                    'n_seeds': n_seeds
                })

            r = df_pc_corr_resid.loc[cancer, xai]
            p = df_pc_p_resid.loc[cancer, xai]
            qv = df_pc_q_resid.loc[cancer, xai]
            sig = df_pc_sig_resid.loc[cancer, xai]
            if not (pd.isna(r) and pd.isna(p) and pd.isna(qv)):
                rows.append({
                    'cancer_type': cancer,
                    'xai_method': xai,
                    'corr_type': 'resid',
                    'spearman_r': r,
                    'p_value': p,
                    'q_value_bh': qv,
                    'sig_fdr05': bool(sig) if pd.notna(sig) else False,
                    'n_seeds': n_seeds
                })

    df_pc_out = pd.DataFrame(rows)
    out_pc_csv = os.path.join(OUTPUT_DIR, f"prognostic_count_vs_kuncheva_correlation_top{TOP_K}.csv")
    df_pc_out.to_csv(out_pc_csv, index=False)

    # supplementary: heatmaps (raw / resid)
    create_fan_heatmap(
        df_pc_corr_raw,
        df_pc_sig_raw,
        os.path.join(OUTPUT_DIR, f"fan_heatmap_kuncheva_prognostic_raw_top{TOP_K}.png"),
        f"Correlation between Prognostic factor count and Kuncheva Index (raw, Top-{TOP_K})\nacross Cancer Types and XAI Methods",
        "Kuncheva"
    )
    create_fan_heatmap(
        df_pc_corr_resid,
        df_pc_sig_resid,
        os.path.join(OUTPUT_DIR, f"fan_heatmap_kuncheva_prognostic_resid_top{TOP_K}.png"),
        f"Correlation between Prognostic factor count and Kuncheva Index \n(residual, control C-index, Top-{TOP_K}) across Cancer Types and XAI Methods",
        "Kuncheva"
    )

    # main-text style figure: across-cancers distribution per XAI
    create_xai_rho_distribution_plot(
        df_pc_corr_raw,
        df_pc_corr_resid,
        os.path.join(OUTPUT_DIR, f"xai_rho_distribution_prognostic_vs_kuncheva_top{TOP_K}.png")
    )

    print("\n" + "=" * 60)
    print("Prognostic factor count vs Kuncheva 稳定性分析完成！")
    print(f"- 主图(分布图): xai_rho_distribution_prognostic_vs_kuncheva_top{TOP_K}.png")
    print(f"- 补充热图(raw): fan_heatmap_kuncheva_prognostic_raw_top{TOP_K}.png")
    print(f"- 补充热图(resid): fan_heatmap_kuncheva_prognostic_resid_top{TOP_K}.png")
    print(f"- per-seed 点数据CSV: {os.path.basename(points_csv)}")
    print(f"- 相关性长表CSV: {os.path.basename(out_pc_csv)}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Kuncheva 热图绘制完成！")
    print(f"- 扇形热图: fan_heatmap_kuncheva_stability_cindex_top{TOP_K}.png")
    print(f"- CSV: {os.path.basename(out_csv)}")
    print("=" * 60)
if __name__ == "__main__":
    main()