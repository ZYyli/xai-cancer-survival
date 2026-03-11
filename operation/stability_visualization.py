""" 
特征稳定性可视化脚本

功能：基于feature_stability_analysis.py生成的数据文件，创建5个主要可视化图表
作者：AI助手
日期：2025-12-11

生成图表：
1. 6种XAI方法稳定性箱线图（跨癌种）
2. 3类XAI方法稳定性箱线图（跨癌种）
3. 15癌症×6XAI稳定性热力图
4. 代表癌症（LGG）6种XAI方法稳定性箱线图（50个抽样模型）
5. Kuncheva随Top-k变化折线图
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
# 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 可选：强制使用 Arial 字体（BBRC 推荐）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 8


# XAI方法分类映射
XAI_CATEGORIES = {
    'IG': 'Gradient',
    'shap': 'Gradient',
    'deepshap': 'Propagation',
    'LRP': 'Propagation',
    'DeepLIFT': 'Propagation',
    'PFI': 'Perturbation'
}

# XAI方法显示名称（统一大小写）
XAI_DISPLAY_NAMES = {
    'shap': 'G-SHAP',
    'deepshap': 'D-SHAP',
    'DeepLIFT': 'DeepLIFT',
    'LRP': 'LRP',
    'IG': 'IG',
    'PFI': 'PFI'
}

# 类别颜色
CATEGORY_COLORS = {
    'Gradient': '#e6cf88',      # 黄色
    'Perturbation': '#e6889f',  # 粉红
    'Propagation': '#88cee6'    # 蓝色
}

# XAI方法独立颜色（可自定义每个方法的颜色）
XAI_COLORS = {
    'IG': '#F49E39',        # 黄色 (Gradient)
    'shap': '#D3C0A3',      # 米棕色 (Gradient)
    'deepshap': '#9271B1',  # 紫色 (Gradient)
    'LRP': '#A6DAEF',       # 天蓝色 (Propagation)
    'DeepLIFT': '#66c1a4',  # 绿色 (Propagation)
    'PFI': '#E26472'        # 粉红色 (Perturbation)
}

# XAI方法颜色（基于类别或独立颜色）
def get_xai_color(xai_method, use_individual=True):
    """
    获取XAI方法的颜色
    
    参数:
    - xai_method: XAI方法名称
    - use_individual: True=使用独立颜色, False=使用类别颜色
    """
    if use_individual and xai_method in XAI_COLORS:
        return XAI_COLORS[xai_method]
    else:
        category = XAI_CATEGORIES.get(xai_method, 'Unknown')
        return CATEGORY_COLORS.get(category, '#95a5a6')


# ============================================================================
# 统计检验函数（与boxplot_prognotic.py一致）
# ============================================================================

def friedman_test_paired(df, block_col, group_col, value_col, group_order=None):
    wide = df.pivot_table(index=block_col, columns=group_col, values=value_col, aggfunc='median')
    if group_order is not None:
        wide = wide.reindex(columns=list(group_order))
    wide = wide.dropna(axis=0, how='any')
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        return np.nan, 1.0, np.nan, int(wide.shape[0])

    arrays = [wide[col].values for col in wide.columns]
    stat, p_value = friedmanchisquare(*arrays)

    n = int(wide.shape[0])
    k = int(wide.shape[1])
    kendalls_w = float(stat) / (n * (k - 1)) if (n > 0 and k > 1) else np.nan
    return float(stat), float(p_value), float(kendalls_w), n


def _wilcoxon_safe(x, y):
    try:
        return wilcoxon(x, y, alternative='two-sided', zero_method='zsplit')
    except TypeError:
        try:
            return wilcoxon(x, y, alternative='two-sided')
        except ValueError:
            return np.nan, 1.0
    except ValueError:
        return np.nan, 1.0


def _rank_biserial_paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[~np.isnan(d)]
    d_nz = d[d != 0]
    n_nonzero = int(d_nz.size)
    if n_nonzero == 0:
        return np.nan, 0

    ranks = rankdata(np.abs(d_nz))
    w_plus = float(np.sum(ranks[d_nz > 0]))
    w_minus = float(np.sum(ranks[d_nz < 0]))
    denom = w_plus + w_minus
    if denom == 0:
        return np.nan, n_nonzero

    r_rb = (w_plus - w_minus) / denom
    return float(r_rb), n_nonzero


def pairwise_wilcoxon_paired(df, block_col, group_col, value_col, group_order=None):
    wide = df.pivot_table(index=block_col, columns=group_col, values=value_col, aggfunc='median')
    if group_order is not None:
        wide = wide.reindex(columns=list(group_order))
    wide = wide.dropna(axis=0, how='any')

    groups = list(wide.columns)
    results = []
    for g1, g2 in combinations(groups, 2):
        x = wide[g1].values
        y = wide[g2].values
        stat, p_value = _wilcoxon_safe(x, y)
        results.append({
            'Group_1': g1,
            'Group_2': g2,
            'Statistic': float(stat) if stat is not None else np.nan,
            'p_value': float(p_value),
            'n_blocks': int(len(x)),
            'Median_1': float(np.median(x)),
            'Median_2': float(np.median(y)),
            'Median_Diff': float(np.median(x) - np.median(y))
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df['sig_fdr'] = results_df['p_fdr'] < 0.05

    return results_df


def add_significance_annotations_from_results(ax, pairwise_results, group_order, max_annotations=None):
    if pairwise_results is None or len(pairwise_results) == 0:
        return None

    if 'sig_fdr' not in pairwise_results.columns or 'p_fdr' not in pairwise_results.columns:
        return pairwise_results

    sig_pairs = pairwise_results[pairwise_results['sig_fdr']].copy()
    if max_annotations is not None and len(sig_pairs) > max_annotations:
        sig_pairs = sig_pairs.nsmallest(max_annotations, 'p_fdr')
    if len(sig_pairs) == 0:
        return pairwise_results

    group_to_idx = {g: i for i, g in enumerate(list(group_order))
                    }
    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low
    if y_range <= 0:
        y_range = 1.0

    for level, (_, row) in enumerate(sig_pairs.iterrows()):
        g1_idx = group_to_idx.get(row['Group_1'])
        g2_idx = group_to_idx.get(row['Group_2'])
        if g1_idx is None or g2_idx is None:
            continue

        y = y_high + y_range * (0.05 + 0.08 * level)
        ax.plot([g1_idx, g2_idx], [y, y], 'k-', lw=1.5)

        sig_symbol = _p_to_sig_symbol(row['p_fdr'])
        if sig_symbol == '':
            continue

        ax.text((g1_idx + g2_idx) / 2, y, sig_symbol,
                ha='center', va='bottom', fontsize=8)

    last_level = max(0, len(sig_pairs) - 1)
    y_top_needed = y_high + y_range * (0.05 + 0.08 * last_level + 0.06)
    if y_top_needed > y_high:
        ax.set_ylim(y_low, y_top_needed)

    return pairwise_results


def _p_to_sig_symbol(p_val):
    p_val = float(p_val)
    if p_val < 0.001:
        return '***'
    if p_val < 0.01:
        return '**'
    if p_val < 0.05:
        return '*'
    return ''


def _get_pairwise_p(pairwise_results, g1, g2, p_col='p_fdr', g1_col='Group_1', g2_col='Group_2'):
    if pairwise_results is None or len(pairwise_results) == 0:
        return np.nan
    if g1_col not in pairwise_results.columns or g2_col not in pairwise_results.columns:
        return np.nan
    mask = ((pairwise_results[g1_col] == g1) & (pairwise_results[g2_col] == g2)) | (
        (pairwise_results[g1_col] == g2) & (pairwise_results[g2_col] == g1)
    )
    row = pairwise_results.loc[mask]
    if len(row) == 0 or p_col not in row.columns:
        return np.nan
    return float(row.iloc[0][p_col])


def add_champion_significance_annotations(ax, pairwise_results, group_order, champion, y_pad_fraction=0.00006, fontsize=12):
    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low
    if y_range <= 0:
        y_range = 1.0
    y = y_high + y_range * y_pad_fraction

    for i, g in enumerate(list(group_order)):
        if g == champion:
            continue
        p_val = _get_pairwise_p(pairwise_results, champion, g, p_col='p_fdr', g1_col='Group_1', g2_col='Group_2')
        if np.isnan(p_val):
            label = ''
        else:
            label = _p_to_sig_symbol(p_val)
        if label == '':
            continue
        ax.text(i, y, label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

    ax.set_ylim(y_low, y + y_range * max(0.05, y_pad_fraction * 0.6))


def _describe_by_group(df, group_cols, value_col):
    summary = (
        df.groupby(list(group_cols))[value_col]
        .agg(
            n_models='count',
            mean='mean',
            std='std',
            min='min',
            q25=lambda x: x.quantile(0.25),
            median='median',
            q75=lambda x: x.quantile(0.75),
            max='max',
        )
        .reset_index()
    )
    summary['IQR'] = summary['q75'] - summary['q25']
    return summary


def _pairwise_to_per_model_metric(df_pairs, metric_col):
    required_cols = ['repeat1', 'fold1', 'repeat2', 'fold2', metric_col]
    if any(c not in df_pairs.columns for c in required_cols):
        return pd.DataFrame()

    a = df_pairs[['repeat1', 'fold1', metric_col]].copy()
    a.rename(columns={'repeat1': 'repeat', 'fold1': 'fold', metric_col: 'value'}, inplace=True)
    b = df_pairs[['repeat2', 'fold2', metric_col]].copy()
    b.rename(columns={'repeat2': 'repeat', 'fold2': 'fold', metric_col: 'value'}, inplace=True)

    long_df = pd.concat([a, b], axis=0, ignore_index=True)
    long_df['repeat'] = long_df['repeat'].astype(int)
    long_df['fold'] = long_df['fold'].astype(int)
    long_df['replicate_id'] = long_df['repeat'].astype(str) + '_' + long_df['fold'].astype(str)

    out = (
        long_df.groupby(['replicate_id', 'repeat', 'fold'], as_index=False)['value']
        .mean()
    )
    return out


def within_category_method_tests(df, output_dir, metric, top_k):
    rows_overall = []
    rows_pairwise = []

    df0 = df.copy()
    for category, sub in df0.groupby('xai_category'):
        methods = sorted(sub['xai_display'].dropna().unique().tolist())
        if len(methods) < 2:
            continue

        wide = sub.pivot_table(index='cancer_type', columns='xai_display', values=metric, aggfunc='median')
        wide = wide.reindex(columns=methods)
        wide = wide.dropna(axis=0, how='any')
        if wide.shape[0] < 2:
            continue

        if len(methods) == 2:
            m1, m2 = methods
            stat, p_value = _wilcoxon_safe(wide[m1].values, wide[m2].values)
            r_rb, n_nonzero = _rank_biserial_paired(wide[m1].values, wide[m2].values)
            rows_overall.append({
                'Category': category,
                'Test': 'Wilcoxon signed-rank (paired, per-cancer median)',
                'n_cancers': int(wide.shape[0]),
                'k_methods': int(len(methods)),
                'n_nonzero': int(n_nonzero),
                'Methods': f"{m1} vs {m2}",
                'Statistic': float(stat),
                'p_value': float(p_value),
                "Effect_Size_Kendalls_W": np.nan,
                'Effect_Size_RankBiserial': float(r_rb) if r_rb is not None else np.nan
            })
        else:
            arrays = [wide[m].values for m in methods]
            stat, p_value = friedmanchisquare(*arrays)
            n = int(wide.shape[0])
            k = int(len(methods))
            kendalls_w = float(stat) / (n * (k - 1)) if (n > 0 and k > 1) else np.nan
            rows_overall.append({
                'Category': category,
                'Test': 'Friedman test (paired, per-cancer median)',
                'n_cancers': int(wide.shape[0]),
                'k_methods': int(len(methods)),
                'n_nonzero': np.nan,
                'Methods': ' | '.join(methods),
                'Statistic': float(stat),
                'p_value': float(p_value),
                "Effect_Size_Kendalls_W": float(kendalls_w) if kendalls_w is not None else np.nan,
                'Effect_Size_RankBiserial': np.nan
            })

        for m1, m2 in combinations(methods, 2):
            stat_pw, p_pw = _wilcoxon_safe(wide[m1].values, wide[m2].values)
            r_rb, n_nonzero = _rank_biserial_paired(wide[m1].values, wide[m2].values)
            rows_pairwise.append({
                'Category': category,
                'Method_1': m1,
                'Method_2': m2,
                'n_cancers': int(wide.shape[0]),
                'n_nonzero': int(n_nonzero),
                'Statistic': float(stat_pw),
                'p_value': float(p_pw),
                'Median_1': float(np.median(wide[m1].values)),
                'Median_2': float(np.median(wide[m2].values)),
                'Median_Diff': float(np.median(wide[m1].values) - np.median(wide[m2].values)),
                'Effect_Size_RankBiserial': float(r_rb) if r_rb is not None else np.nan
            })

    overall_df = pd.DataFrame(rows_overall)
    pairwise_df = pd.DataFrame(rows_pairwise)
    if len(pairwise_df) > 0:
        _, p_fdr, _, _ = multipletests(pairwise_df['p_value'], method='fdr_bh')
        pairwise_df['p_fdr'] = p_fdr
        pairwise_df['sig_fdr'] = pairwise_df['p_fdr'] < 0.05

    out_overall = os.path.join(output_dir, f'within_category_method_tests_overall_top{top_k}_{metric}.csv')
    out_pairwise = os.path.join(output_dir, f'within_category_method_tests_pairwise_top{top_k}_{metric}.csv')
    overall_df.to_csv(out_overall, index=False)
    pairwise_df.to_csv(out_pairwise, index=False)
    return out_overall, out_pairwise


def _stability_metric_to_raw_column(metric):
    if 'kuncheva' in metric:
        return 'kuncheva'
    if 'jaccard' in metric:
        return 'jaccard'
    return metric


def within_cancer_method_tests_from_pairwise_raw(df_long, output_dir, metric, top_k, method_order):
    rows_overall = []
    rows_pairwise = []

    for cancer, sub in df_long.groupby('cancer_type'):
        pair_index_col = 'pair_key' if 'pair_key' in sub.columns else 'pair_id'
        wide = sub.pivot_table(index=pair_index_col, columns='xai_display', values=metric, aggfunc='median')
        wide = wide.reindex(columns=list(method_order))
        wide = wide.dropna(axis=0, how='any')
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue

        arrays = [wide[m].values for m in wide.columns]
        stat, p_value = friedmanchisquare(*arrays)
        n = int(wide.shape[0])
        k = int(wide.shape[1])
        kendalls_w = float(stat) / (n * (k - 1)) if (n > 0 and k > 1) else np.nan
        rows_overall.append({
            'Cancer': cancer,
            'Test': 'Friedman (paired by pair_id)',
            'n_blocks': n,
            'k_methods': k,
            'Statistic': float(stat),
            'p_value': float(p_value),
            "Effect_Size_Kendalls_W": float(kendalls_w) if kendalls_w is not None else np.nan
        })

        for m1, m2 in combinations(list(wide.columns), 2):
            x = wide[m1].values
            y = wide[m2].values
            stat_pw, p_pw = _wilcoxon_safe(x, y)
            r_rb, n_nonzero = _rank_biserial_paired(x, y)
            rows_pairwise.append({
                'Cancer': cancer,
                'Method_1': m1,
                'Method_2': m2,
                'n_blocks': int(len(x)),
                'n_nonzero': int(n_nonzero),
                'Statistic': float(stat_pw) if stat_pw is not None else np.nan,
                'p_value': float(p_pw),
                'Median_1': float(np.median(x)),
                'Median_2': float(np.median(y)),
                'Median_Diff': float(np.median(x) - np.median(y)),
                'Effect_Size_RankBiserial': float(r_rb) if r_rb is not None else np.nan
            })

    overall_df = pd.DataFrame(rows_overall)
    pairwise_df = pd.DataFrame(rows_pairwise)
    if len(pairwise_df) > 0:
        pairwise_df['p_fdr'] = np.nan
        pairwise_df['sig_fdr'] = False
        for cancer, idx in pairwise_df.groupby('Cancer').groups.items():
            pvals = pairwise_df.loc[idx, 'p_value'].values
            _, p_fdr, _, _ = multipletests(pvals, method='fdr_bh')
            pairwise_df.loc[idx, 'p_fdr'] = p_fdr
            pairwise_df.loc[idx, 'sig_fdr'] = pairwise_df.loc[idx, 'p_fdr'] < 0.05

    out_overall = os.path.join(output_dir, f'within_cancer_method_tests_overall_top{top_k}_{metric}.csv')
    out_pairwise = os.path.join(output_dir, f'within_cancer_method_tests_pairwise_top{top_k}_{metric}.csv')
    overall_df.to_csv(out_overall, index=False)
    pairwise_df.to_csv(out_pairwise, index=False)
    return overall_df, pairwise_df


def _resolve_metric_column(df, metric):
    if metric in df.columns:
        return metric
    if metric.endswith('_median'):
        fallback = metric[:-7] + '_mean'
        if fallback in df.columns:
            print(f"   ⚠️ 列 {metric} 不存在，回退使用 {fallback}")
            return fallback
    raise KeyError(f"Metric column not found: {metric}")


def _save_figure(fig, save_path_png, dpi=600):
    if fig is None:
        return
    plt.savefig(save_path_png, dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path_png.replace('.png', '.pdf'), bbox_inches='tight')


def _style_axes(ax, spine_width=0.9, tick_width=1.2, grid_alpha=0.3):
    ax.grid(axis='y', linestyle='-', alpha=grid_alpha)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(spine_width)
    ax.tick_params(axis='both', colors='black', width=tick_width)


class StabilityVisualizer:
    """稳定性可视化器"""
    
    def __init__(self, data_dir, output_dir=None):
        """
        初始化可视化器
        
        参数:
        - data_dir: 稳定性分析数据目录（包含cross_cancer_stability_summary.csv等）
        - output_dir: 图表输出目录，默认为data_dir/figures
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, 'figures')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载主数据文件
        self.summary_df = self._load_summary_data()
        
        print(f"📊 初始化稳定性可视化器")
        print(f"   数据目录: {data_dir}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   数据行数: {len(self.summary_df)}")
        
    def _load_summary_data(self):
        """加载跨癌症汇总数据"""
        summary_path = os.path.join(self.data_dir, 'cross_cancer_stability_summary.csv')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"找不到汇总数据文件: {summary_path}")
        
        df = pd.read_csv(summary_path)
        
        # 添加XAI类别列
        df['xai_category'] = df['xai_method'].map(XAI_CATEGORIES)
        
        # 添加显示名称列
        df['xai_display'] = df['xai_method'].map(XAI_DISPLAY_NAMES)
        
        return df

    def _get_display_to_original(self):
        return {v: k for k, v in XAI_DISPLAY_NAMES.items() if pd.notna(v)}
    
    def plot_xai_method_boxplot(self, top_k=100, metric='kuncheva_median', save=True):
        """
        图1: 6种XAI方法稳定性箱线图
        
        参数:
        - top_k: 使用的Top-k值
        - metric: 使用的指标（例如 kuncheva_median）
        - save: 是否保存图片
        """
        print(f"\n📈 绘制图1: 6种XAI方法稳定性箱线图 (Top-{top_k}, {metric})")
        
        # 筛选数据
        df = self.summary_df[self.summary_df['top_k'] == top_k].copy()
        metric = _resolve_metric_column(df, metric)
        
        if df.empty:
            print(f"   ⚠️ 没有Top-{top_k}的数据")
            return None

        if save:
            print(
                "   🧪 附加统计: 将在每个XAI类别(xai_category)内部做方法间差异检验(不是图1的整体6方法比较)，结果输出为 within_category_method_tests_*.csv"
            )
            within_category_method_tests(df, self.output_dir, metric, top_k)
        
        # 按XAI方法的中位数排序（从低到高，与boxplot_prognotic.py一致）
        order = df.groupby('xai_display')[metric].median().sort_values().index.tolist()
        
        display_to_original = self._get_display_to_original()

        # 获取排序后的颜色
        colors_ordered = [get_xai_color(display_to_original.get(m, m)) for m in order]
        
        # 统计检验（与boxplot_prognotic.py图2一致：Friedman + 配对Wilcoxon + FDR）
        print(f"\n   Friedman检验 (paired by Cancer):")
        stat, p_value, kendalls_w, n_blocks = friedman_test_paired(
            df, block_col='cancer_type', group_col='xai_display', value_col=metric, group_order=order
        )
        print(f"     参与癌种数 = {n_blocks}")
        print(f"     统计量 = {stat:.4f}")
        print(f"     p值 = {p_value:.6e}")
        print(f"     效应量 (Kendall's W) = {kendalls_w:.4f}")
        print(f"     结论: {'存在显著差异 ✓' if p_value < 0.05 else '无显著差异'}")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(2.35, 2.6))
        
        # 绘制箱线图
        sns.boxplot(data=df, x='xai_display', y=metric, order=order, 
                    palette=colors_ordered, showfliers=False, ax=ax, linecolor="black", linewidth=1)
        
        # 设置箱体透明度
        for patch in ax.patches:
            patch.set_alpha(0.7)
        
        # 添加散点
        for i, g in enumerate(order):
            y = df[df['xai_display'] == g][metric].values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.9, s=5, color=colors_ordered[i], zorder=3)

        pairwise_results = pairwise_wilcoxon_paired(
            df, block_col='cancer_type', group_col='xai_display', value_col=metric, group_order=order
        )
        group_scores = df.groupby('xai_display')[metric].median().to_dict()
        champion = max(list(order), key=lambda g: (group_scores.get(g, -np.inf), str(g)))
        add_champion_significance_annotations(ax, pairwise_results, order, champion)
        
        _style_axes(ax)
        
        # 设置标签
        metric_label = {
            'kuncheva_mean': 'Kuncheva Index',
            'kuncheva_median': 'Kuncheva Index',
        }.get(metric, metric)
        
        #ax.set_title(f'Feature Stability by XAI Method\nAggregated Across 15 Cancer Types (Top-{top_k})', fontweight='bold')
        ax.set_ylabel(metric_label)
        ax.set_xlabel('XAI Method', labelpad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'fig1_xai_method_boxplot_top{top_k}.png')
            _save_figure(fig, save_path, dpi=600)
            print(f"   ✅ 保存到: {save_path}")

            fig1_plot_data = df[['cancer_type', 'xai_display', metric]].copy()
            fig1_plot_data['cancer_type'] = pd.Categorical(fig1_plot_data['cancer_type'], ordered=False)
            fig1_plot_data['xai_display'] = pd.Categorical(fig1_plot_data['xai_display'], categories=order, ordered=True)
            fig1_plot_data = fig1_plot_data.sort_values(['cancer_type', 'xai_display'])
            fig1_plot_data_file = os.path.join(self.output_dir, f'supplementary_fig1_plot_data_method_per_cancer_top{top_k}_{metric}.csv')
            fig1_plot_data.to_csv(fig1_plot_data_file, index=False)

            fig1_summary = _describe_by_group(fig1_plot_data, ['xai_display'], metric)
            fig1_summary.rename(columns={'n_models': 'n_cancers', 'xai_display': 'XAI_Method'}, inplace=True)
            fig1_summary['XAI_Method'] = pd.Categorical(fig1_summary['XAI_Method'], categories=order, ordered=True)
            fig1_summary = fig1_summary.sort_values(['XAI_Method'])
            fig1_summary_file = os.path.join(self.output_dir, f'supplementary_fig1_summary_across_cancers_by_method_top{top_k}_{metric}.csv')
            fig1_summary.to_csv(fig1_summary_file, index=False)
            
            # 保存统计检验结果
            if pairwise_results is not None:
                stats_file = os.path.join(self.output_dir, f'statistical_tests_xai_methods_top{top_k}.csv')
                overall_stats = pd.DataFrame([{
                    'Test': 'Friedman (paired by Cancer)',
                    'Statistic': stat,
                    'p_value': p_value,
                    "Effect_Size_Kendalls_W": kendalls_w,
                    'n_cancers': n_blocks,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                }])

                with open(stats_file, 'w') as f:
                    f.write("# Overall Test: Friedman Test (paired by Cancer)\n")
                    overall_stats.to_csv(f, index=False)
                    f.write("\n# Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction\n")
                    pairwise_results.to_csv(f, index=False)

                print(f"   ✅ 统计检验结果保存到: {stats_file}")
        
        plt.close()
        return fig

    def plot_representative_cancer_boxplot(self, cancer_type, top_k=100, save=True):
        print(f"\n📈 绘制图4: 代表癌症 {cancer_type} 稳定性箱线图 (Kuncheva, Top-{top_k})")

        display_to_original = {v: k for k, v in XAI_DISPLAY_NAMES.items()}
        rows = []
        for xai_display in sorted(self.summary_df['xai_display'].dropna().unique().tolist()):
            original = display_to_original.get(xai_display)
            if original is None:
                continue
            raw_path = os.path.join(self.data_dir, cancer_type, original, f'pairwise_stability_raw_top{top_k}.csv')
            if not os.path.exists(raw_path):
                continue
            dfr = pd.read_csv(raw_path)
            if 'kuncheva' not in dfr.columns:
                continue
            per_model = _pairwise_to_per_model_metric(dfr, 'kuncheva')
            if per_model.empty:
                continue
            per_model['xai_display'] = xai_display
            per_model.rename(columns={'value': 'kuncheva'}, inplace=True)
            rows.append(per_model)

        if len(rows) == 0:
            print(f"   ⚠️ 未找到 {cancer_type} 的原始pairwise稳定性文件，无法绘制图4")
            return None

        df_lgg = pd.concat(rows, axis=0, ignore_index=True)

        method_order = df_lgg.groupby('xai_display')['kuncheva'].median().sort_values().index.tolist()
        colors_ordered = [
            get_xai_color(display_to_original.get(m, m))
            for m in method_order
        ]

        stat, p_value, kendalls_w, n_blocks = friedman_test_paired(
            df_lgg, block_col='replicate_id', group_col='xai_display', value_col='kuncheva', group_order=method_order
        )

        fig, ax = plt.subplots(figsize=(2.5, 2.6))
        sns.violinplot(
            data=df_lgg,
            x='xai_display',
            y='kuncheva',
            order=method_order,
            palette=colors_ordered,
            ax=ax,
            inner=None,
            cut=0,
            linewidth=0.8
        )
        for coll in ax.collections:
            try:
                coll.set_alpha(0.5)
            except Exception:
                pass
        sns.boxplot(
            data=df_lgg,
            x='xai_display',
            y='kuncheva',
            order=method_order,
            palette=colors_ordered,
            showfliers=False,
            ax=ax,
            linecolor="black", linewidth=1
        )
        for patch in ax.patches:
            patch.set_alpha(0.7)
        for i, m in enumerate(method_order):
            y = df_lgg[df_lgg['xai_display'] == m]['kuncheva'].values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.9, s=3, color=colors_ordered[i], zorder=3)

        pairwise_results = pairwise_wilcoxon_paired(
            df_lgg, block_col='replicate_id', group_col='xai_display', value_col='kuncheva', group_order=method_order
        )
        method_scores = df_lgg.groupby('xai_display')['kuncheva'].median().to_dict()
        champion = max(list(method_order), key=lambda g: (method_scores.get(g, -np.inf), str(g)))
        add_champion_significance_annotations(ax, pairwise_results, method_order, champion)

        _style_axes(ax)

        #ax.set_title(f'Feature Stability by XAI Method\n{cancer_type} Cancer Type (Top-{top_k})', fontweight='bold')
        ax.set_ylabel('Kuncheva Index')
        ax.set_xlabel('XAI Method', labelpad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_dir, f'fig4_representative_{cancer_type}_boxplot_kuncheva_top{top_k}.png')
            _save_figure(fig, save_path, dpi=600)

            fig4_plot_data = df_lgg[['replicate_id', 'repeat', 'fold', 'xai_display', 'kuncheva']].copy()
            fig4_plot_data['xai_display'] = pd.Categorical(fig4_plot_data['xai_display'], categories=method_order, ordered=True)
            fig4_plot_data = fig4_plot_data.sort_values(['replicate_id', 'xai_display'])
            fig4_plot_data_file = os.path.join(self.output_dir, f'supplementary_fig4_plot_data_{cancer_type}_models_kuncheva_top{top_k}.csv')
            fig4_plot_data.to_csv(fig4_plot_data_file, index=False)

            fig4_summary = _describe_by_group(fig4_plot_data, ['xai_display'], 'kuncheva')
            fig4_summary.rename(columns={'xai_display': 'XAI_Method'}, inplace=True)
            fig4_summary['XAI_Method'] = pd.Categorical(fig4_summary['XAI_Method'], categories=method_order, ordered=True)
            fig4_summary = fig4_summary.sort_values(['XAI_Method'])
            fig4_summary_file = os.path.join(self.output_dir, f'supplementary_fig4_summary_across_models_by_method_{cancer_type}_top{top_k}.csv')
            fig4_summary.to_csv(fig4_summary_file, index=False)

            fig4_wide = (
                df_lgg
                .pivot_table(index='replicate_id', columns='xai_display', values='kuncheva', aggfunc='median')
                .reindex(columns=method_order)
                .dropna(axis=0, how='any')
            )
            fig4_wide_file = os.path.join(self.output_dir, f'supplementary_fig4_paired_wide_data_used_for_tests_{cancer_type}_top{top_k}.csv')
            fig4_wide.reset_index().to_csv(fig4_wide_file, index=False)

            stats_file = os.path.join(self.output_dir, f'statistical_tests_{cancer_type}_xai_methods_kuncheva_top{top_k}.csv')
            overall_stats = pd.DataFrame([{
                'Test': 'Friedman (paired by replicate_id)',
                'Statistic': stat,
                'p_value': p_value,
                'Effect_Size_Kendalls_W': kendalls_w,
                'n_models': n_blocks,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            }])
            with open(stats_file, 'w') as f:
                f.write("# Overall Test: Friedman Test (paired by replicate_id)\n")
                overall_stats.to_csv(f, index=False)
                f.write("\n# Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction\n")
                pairwise_results.to_csv(f, index=False)

        plt.close()
        return fig
    
    def plot_xai_category_boxplot(self, top_k=100, metric='kuncheva_median', save=True):
        """
        图2: 3类XAI方法稳定性箱线图（与boxplot_prognotic.py风格一致）
        
        参数:
        - top_k: 使用的Top-k值
        - metric: 使用的指标
        - save: 是否保存图片
        """
        print(f"\n📈 绘制图2: 3类XAI方法稳定性箱线图 (Top-{top_k}, {metric})")
        
        # 筛选数据
        df = self.summary_df[self.summary_df['top_k'] == top_k].copy()
        metric = _resolve_metric_column(df, metric)
        
        if df.empty:
            print(f"   ⚠️ 没有Top-{top_k}的数据")
            return None
        
        # 按癌种聚合到(Cancer, Category)一个值，避免伪重复（与boxplot_prognotic.py图3一致）
        df = (
            df.groupby(['cancer_type', 'xai_category'], as_index=False)[metric]
            .median()
        )

        # 按类别的中位数排序（从低到高，与boxplot_prognotic.py一致）
        order = df.groupby('xai_category')[metric].median().sort_values().index.tolist()
        
        # 根据order重新排列颜色（使用全局CATEGORY_COLORS）
        colors_ordered = [CATEGORY_COLORS.get(cat, '#95a5a6') for cat in order]
        
        # 统计检验（与boxplot_prognotic.py图3一致：Friedman + 配对Wilcoxon + FDR）
        print(f"\n   Friedman检验 (paired by Cancer):")
        stat, p_value, kendalls_w, n_blocks = friedman_test_paired(
            df, block_col='cancer_type', group_col='xai_category', value_col=metric, group_order=order
        )
        print(f"     参与癌种数 = {n_blocks}")
        print(f"     统计量 = {stat:.4f}")
        print(f"     p值 = {p_value:.6e}")
        print(f"     效应量 (Kendall's W) = {kendalls_w:.4f}")
        print(f"     结论: {'存在显著差异 ✓' if p_value < 0.05 else '无显著差异'}")
        
        # 创建图表（与boxplot_prognotic.py一致的尺寸）
        fig, ax = plt.subplots(figsize=(1.8, 2.8))
        
        # 绘制箱线图
        sns.boxplot(data=df, x='xai_category', y=metric, order=order,
                    palette=colors_ordered, showfliers=False, ax=ax, linecolor="black", linewidth=1)
        
        # 设置箱体透明度
        for patch in ax.patches:
            patch.set_alpha(0.7)
        
        # 添加散点（与boxplot_prognotic.py一致的 jitter/scatter 参数）
        for i, g in enumerate(order):
            y = df[df['xai_category'] == g][metric].values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.9, s=6, color=colors_ordered[i], zorder=3)

        pairwise_results = pairwise_wilcoxon_paired(
            df, block_col='cancer_type', group_col='xai_category', value_col=metric, group_order=order
        )
        pairwise_results = add_significance_annotations_from_results(
            ax, pairwise_results, order, max_annotations=None
        )
        
        _style_axes(ax)

        # 设置网格线：只显示横线
        ax.grid(axis='y', linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)

        # 设置标签
        metric_label = {
            'kuncheva_mean': 'Kuncheva Index',
            'kuncheva_median': 'Kuncheva Index',
        }.get(metric, metric)
        
        #ax.set_title(f'Feature Stability by XAI Category\nAggregated Across 15 Cancer Types (Top-{top_k})', fontweight='bold')
        ax.set_ylabel(metric_label)
        ax.set_xlabel('XAI Category', labelpad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'fig2_xai_category_boxplot_top{top_k}.png')
            _save_figure(fig, save_path, dpi=600)
            print(f"   ✅ 保存到: {save_path}")

            fig2_plot_data = df[['cancer_type', 'xai_category', metric]].copy()
            fig2_plot_data['cancer_type'] = pd.Categorical(fig2_plot_data['cancer_type'], ordered=False)
            fig2_plot_data['xai_category'] = pd.Categorical(fig2_plot_data['xai_category'], categories=order, ordered=True)
            fig2_plot_data = fig2_plot_data.sort_values(['cancer_type', 'xai_category'])
            fig2_plot_data_file = os.path.join(self.output_dir, f'supplementary_fig2_plot_data_category_per_cancer_top{top_k}_{metric}.csv')
            fig2_plot_data.to_csv(fig2_plot_data_file, index=False)

            fig2_summary = _describe_by_group(fig2_plot_data, ['xai_category'], metric)
            fig2_summary.rename(columns={'n_models': 'n_cancers', 'xai_category': 'XAI_Category'}, inplace=True)
            fig2_summary['XAI_Category'] = pd.Categorical(fig2_summary['XAI_Category'], categories=order, ordered=True)
            fig2_summary = fig2_summary.sort_values(['XAI_Category'])
            fig2_summary_file = os.path.join(self.output_dir, f'supplementary_fig2_summary_across_cancers_by_category_top{top_k}_{metric}.csv')
            fig2_summary.to_csv(fig2_summary_file, index=False)
            
            # 保存统计检验结果
            if pairwise_results is not None:
                stats_file = os.path.join(self.output_dir, f'statistical_tests_xai_categories_top{top_k}.csv')
                overall_stats = pd.DataFrame([{
                    'Test': 'Friedman (paired by Cancer)',
                    'Statistic': stat,
                    'p_value': p_value,
                    "Effect_Size_Kendalls_W": kendalls_w,
                    'n_cancers': n_blocks,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                }])

                with open(stats_file, 'w') as f:
                    f.write("# Overall Test: Friedman Test (paired by Cancer)\n")
                    overall_stats.to_csv(f, index=False)
                    f.write("\n# Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction\n")
                    pairwise_results.to_csv(f, index=False)

                print(f"   ✅ 统计检验结果保存到: {stats_file}")
        
        plt.close()
        return fig
    
    def plot_stability_heatmap(self, top_k=100, metric='kuncheva_median', save=True, 
                               xai_order=None, cancer_order=None):
        """
        图3: 15癌症×6XAI稳定性热力图
        
        参数:
        - top_k: 使用的Top-k值
        - metric: 使用的指标
        - save: 是否保存图片
        - xai_order: XAI方法的显示顺序（列表）
        - cancer_order: 癌症类型的显示顺序（列表）
        """
        print(f"\n📈 绘制图3: 癌症×XAI稳定性热力图 (Top-{top_k}, {metric})")
        
        # 筛选数据
        df = self.summary_df[self.summary_df['top_k'] == top_k].copy()
        metric = _resolve_metric_column(df, metric)
        
        if df.empty:
            print(f"   ⚠️ 没有Top-{top_k}的数据")
            return None
        
        fig3_plot_data = df[['cancer_type', 'xai_display', metric]].copy()
        fig3_plot_data_file = os.path.join(self.output_dir, f'supplementary_fig3_plot_data_method_per_cancer_top{top_k}_{metric}.csv')
        if save:
            fig3_plot_data.to_csv(fig3_plot_data_file, index=False)

        # 创建透视表：行=XAI方法，列=癌症类型
        pivot = df.pivot(index='xai_display', columns='cancer_type', values=metric)
        
        # fig3 每个癌症内部：6种XAI做配对检验（使用 pairwise_stability_raw_top{top_k}.csv 的 pair_id 作为配对块）
        raw_metric_col = _stability_metric_to_raw_column(metric)
        display_to_original = {v: k for k, v in XAI_DISPLAY_NAMES.items()}
        raw_rows = []

        for cancer in df['cancer_type'].dropna().unique().tolist():
            for xai_display in df['xai_display'].dropna().unique().tolist():
                original = display_to_original.get(xai_display)
                if original is None:
                    continue
                raw_path = os.path.join(self.data_dir, cancer, original, f'pairwise_stability_raw_top{top_k}.csv')
                if not os.path.exists(raw_path):
                    continue
                dfr = pd.read_csv(raw_path)
                if 'pair_id' not in dfr.columns or raw_metric_col not in dfr.columns:
                    continue
                base_cols = ['pair_id', raw_metric_col]
                extra_cols = ['repeat1', 'fold1', 'repeat2', 'fold2']
                use_cols = base_cols + [c for c in extra_cols if c in dfr.columns]
                subr = dfr[use_cols].copy()
                subr['cancer_type'] = cancer
                if all(c in subr.columns for c in extra_cols):
                    subr['pair_key'] = (
                        subr['repeat1'].astype(str) + '_' + subr['fold1'].astype(str)
                        + '_' + subr['repeat2'].astype(str) + '_' + subr['fold2'].astype(str)
                    )
                subr['xai_display'] = xai_display
                subr.rename(columns={raw_metric_col: metric}, inplace=True)
                raw_rows.append(subr)

        pairwise_by_cancer = {}
        champion_by_cancer = {}
        if len(raw_rows) > 0:
            df_raw_long = pd.concat(raw_rows, axis=0, ignore_index=True)

            method_order_for_tests = list(pivot.index)
            overall_df, pairwise_df = within_cancer_method_tests_from_pairwise_raw(
                df_raw_long, self.output_dir, metric, top_k, method_order_for_tests
            )

            if len(pairwise_df) > 0:
                for cancer, sub_pw in pairwise_df.groupby('Cancer'):
                    sub_raw = df_raw_long[df_raw_long['cancer_type'] == cancer]
                    pair_index_col = 'pair_key' if 'pair_key' in sub_raw.columns else 'pair_id'
                    wide = (
                        sub_raw
                        .pivot_table(index=pair_index_col, columns='xai_display', values=metric, aggfunc='median')
                        .reindex(columns=method_order_for_tests)
                        .dropna(axis=0, how='any')
                    )
                    if wide.shape[0] == 0:
                        continue
                    scores = wide.median(axis=0).to_dict()
                    champion = max(list(method_order_for_tests), key=lambda g: (scores.get(g, -np.inf), str(g)))
                    pairwise_by_cancer[cancer] = sub_pw
                    champion_by_cancer[cancer] = champion

        # 按指定顺序排列XAI方法（行）
        if xai_order is not None:
            xai_order_filtered = [x for x in xai_order if x in pivot.index]
            pivot = pivot.loc[xai_order_filtered]
        
        # 按指定顺序排列癌症类型（列）
        if cancer_order is not None:
            cancer_order_filtered = [c for c in cancer_order if c in pivot.columns]
            pivot = pivot[cancer_order_filtered]

        if save:
            fig3_matrix_file = os.path.join(self.output_dir, f'supplementary_fig3_matrix_method_by_cancer_top{top_k}_{metric}.csv')
            pivot.to_csv(fig3_matrix_file)
        
        # 创建图表（优化尺寸：加宽减高，更适合15列数据）
        fig, ax = plt.subplots(figsize=(6.89, 3))

        # 创建颜色映射
        custom_cmap = LinearSegmentedColormap.from_list(
            'kuncheva_colormap',
            ["#1a2a6c", "white", "#b21f1f"]
        )
        
        annot = pivot.copy().astype(object)
        for m in annot.index:
            for c in annot.columns:
                val = annot.loc[m, c]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    annot.loc[m, c] = ''
                    continue
                label = ''
                if c in pairwise_by_cancer and c in champion_by_cancer:
                    champion = champion_by_cancer[c]
                    if m != champion:
                        p_val = _get_pairwise_p(
                            pairwise_by_cancer[c],
                            champion,
                            m,
                            p_col='p_fdr',
                            g1_col='Method_1',
                            g2_col='Method_2',
                        )
                        if not np.isnan(p_val):
                            label = _p_to_sig_symbol(p_val)
                annot.loc[m, c] = f"{float(val):.2f}\n{label}" if label != '' else f"{float(val):.2f}"

        # 绘制热力图（优化配色和格式）
        im = sns.heatmap(pivot, 
                         annot=annot.values,   # 显示数值+冠军基准显著性星号
                         fmt='',               # annot为字符串
                         cmap=custom_cmap,     
                         center=0 if 'kuncheva' in metric else None,  # Kuncheva以0为中心
                         square=False,         # False=自适应矩形，True=强制方形
                         ax=ax, 
                         cbar_kws={
                             'label': metric.replace('_', ' ').title(),
                             'shrink': 0.8,    # 颜色条高度比例（0-1）
                             'aspect': 20,     # 颜色条宽高比
                             'pad': 0.02       # 颜色条与热力图的距离
                         },
                         linewidths=0.8,       # 增加网格线宽度
                         linecolor='white',    # 白色网格线
                         annot_kws={'size': 8})  # 数值字体大小
        
        # 设置标题
        #ax.set_title(f'Feature Stability Across Cancer Types and XAI Methods (Top-{top_k})', 
        #            fontsize=15, fontweight='bold', pad=20)
        # 设置Y轴标签（XAI方法）
        ax.set_ylabel('XAI Method', labelpad=10)
        
        # 设置X轴标签（癌症类型）
        ax.set_xlabel('Cancer Type', labelpad=10)
        
        # 旋转x轴标签，增大字体
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        
        # 调整颜色条位置和样式
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
        #cbar.set_label(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'fig3_stability_heatmap_top{top_k}.png')
            _save_figure(fig, save_path, dpi=600)
            print(f"   ✅ 保存到: {save_path}")
        
        plt.close()
        return fig
    
    def plot_kuncheva_vs_topk(self, metric='kuncheva_median', save=True):
        """
        图5: Kuncheva随Top-k变化折线图
        
        6条折线（代表6种XAI），每条线是15种癌症的平均值
        """
        print(f"\n📈 绘制图5: Kuncheva随Top-k变化折线图")

        metric = _resolve_metric_column(self.summary_df, metric)

        if metric.endswith('_median'):
            agg_df = (
                self.summary_df
                .groupby(['xai_display', 'top_k'])[metric]
                .agg(
                    kuncheva_center='median',
                    kuncheva_q25=lambda x: x.quantile(0.25),
                    kuncheva_q75=lambda x: x.quantile(0.75),
                    n='count'
                )
                .reset_index()
            )
            y_col = 'kuncheva_center'
            low_col = 'kuncheva_q25'
            high_col = 'kuncheva_q75'
            y_label = 'Kuncheva Index'
        else:
            agg_df = (
                self.summary_df
                .groupby(['xai_display', 'top_k'])[metric]
                .agg(
                    kuncheva_center='mean',
                    kuncheva_std='std',
                    n='count'
                )
                .reset_index()
            )
            agg_df['kuncheva_se'] = agg_df['kuncheva_std'] / np.sqrt(agg_df['n'])
            y_col = 'kuncheva_center'
            low_col = None
            high_col = None
            y_label = 'Kuncheva Index'
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(4.8, 3))
        
        # 获取所有XAI方法
        xai_methods = agg_df['xai_display'].unique()

        display_to_original = self._get_display_to_original()
        
        for xai in xai_methods:
            xai_data = agg_df[agg_df['xai_display'] == xai].sort_values('top_k')
            
            # 获取原始方法名以确定颜色
            original_method = display_to_original.get(xai)
            color = get_xai_color(original_method) if original_method is not None else '#95a5a6'
            
            # 绘制折线
            ax.plot(xai_data['top_k'], xai_data[y_col], 
                   marker='o', label=xai, color=color, linewidth=1.2, markersize=4)

            if low_col is not None and high_col is not None:
                ax.fill_between(
                    xai_data['top_k'],
                    xai_data[low_col],
                    xai_data[high_col],
                    alpha=0.2,
                    color=color
                )
            else:
                ax.fill_between(
                    xai_data['top_k'],
                    xai_data[y_col] - xai_data['kuncheva_se'],
                    xai_data[y_col] + xai_data['kuncheva_se'],
                    alpha=0.2,
                    color=color
                )
        
        # 添加随机基线
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random baseline')
        
        ax.set_xlabel('Top-k')
        ax.set_ylabel(y_label)
        ax.set_title('Feature Stability vs Top-k by XAI Method')
        
        # 设置x轴刻度
        ax.set_xticks(agg_df['top_k'].unique())
        
        # 添加图例
        ax.legend(
            loc='center right',
            fontsize=8,
            title_fontsize=8,
            markerscale=0.9,
            handlelength=1.2,
            labelspacing=0.2,
            borderpad=0.2,
        )
        
        _style_axes(ax)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'fig5_kuncheva_vs_topk.png')
            _save_figure(fig, save_path, dpi=600)
            print(f"   ✅ 保存到: {save_path}")

            fig5_plot_data_file = os.path.join(self.output_dir, f'supplementary_fig5_plot_data_kuncheva_vs_topk_{metric}.csv')
            agg_df.to_csv(fig5_plot_data_file, index=False)
        
        plt.close()
        return fig
    
    def generate_all_figures(self, top_k=100, xai_order=None, cancer_order=None):
        """
        生成所有5个主要图表
        
        参数:
        - top_k: 用于箱线图和热力图的Top-k值
        - xai_order: 热力图中XAI方法的显示顺序（列表）
        - cancer_order: 热力图中癌症类型的显示顺序（列表）
        """
        print(f"\n{'='*60}")
        print(f"🎨 开始生成所有可视化图表 (Top-{top_k})")
        print(f"{'='*60}")
        
        # 图1: 6种XAI方法稳定性箱线图
        self.plot_xai_method_boxplot(top_k=top_k, metric='kuncheva_median')
        
        # 图2: 3类XAI方法稳定性箱线图
        self.plot_xai_category_boxplot(top_k=top_k, metric='kuncheva_median')
        
        # 图3: 癌症×XAI稳定性热力图
        self.plot_stability_heatmap(top_k=top_k, metric='kuncheva_median', xai_order=xai_order, cancer_order=cancer_order)

        # 图4: 代表癌症（LGG）方法比较箱线图（Kuncheva，50个抽样模型）
        self.plot_representative_cancer_boxplot(cancer_type='LIHC', top_k=top_k, save=True)
        
        # 图5: Kuncheva随Top-k变化折线图
        self.plot_kuncheva_vs_topk()
        
        print(f"\n{'='*60}")
        print(f"🎉 所有图表生成完成!")
        print(f"   输出目录: {self.output_dir}")
        print(f"{'='*60}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="特征稳定性可视化")
    parser.add_argument('--data_dir', type=str, 
                       default='/home/zuoyiyi/SNN/TCGA/stability_analysis_4',
                       help='稳定性分析数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/zuoyiyi/SNN/TCGA/stability_comparison_plots_nestedcv/visualization',
                       help='图表输出目录（默认为data_dir/figures）')
    parser.add_argument('--top_k', type=int, default=100,
                       help='用于箱线图和热力图的Top-k值')
    args = parser.parse_args()
    
    print(f"🎯 可视化配置:")
    print(f"   数据目录: {args.data_dir}")
    print(f"   Top-k: {args.top_k}")
    
    # 创建可视化器
    visualizer = StabilityVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # 定义XAI方法和癌症类型的显示顺序
    xai_order = ['G-SHAP', 'DeepLIFT', 'D-SHAP', 'IG', 'LRP', 'PFI'] 
    cancer_order = ['COADREAD', 'LUSC', 'HNSC', 'STAD', 'BLCA', 'BRCA', 'LUAD', 'PAAD',
    'LIHC', 'SKCM', 'KIRC', 'UCEC', 'KIRP', 'GBMLGG', 'LGG']
    
    # 生成所有图表
    visualizer.generate_all_figures(top_k=args.top_k, xai_order=xai_order, cancer_order=cancer_order)


if __name__ == '__main__':
    main()
