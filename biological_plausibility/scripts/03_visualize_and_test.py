#!/usr/bin/env python3
"""
生物学合理性可视化

展示不同XAI方法的整体生物学合理性，包括：
1. 箱线图（6种XAI方法比较）
2. 箱线图（3种XAI类别比较）
3. 热力图（数据库×XAI）
4. 热力图（癌种×XAI）

使用方法:
    python 03_visualize_and_test.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore', category=FutureWarning)

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, XAI_METHODS, CANCER_TYPES

# 设置绘图风格
# 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 可选：强制使用 Arial 字体（BBRC 推荐）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['font.size'] = 8
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

# ============================================================================
# 统计检验函数
# ============================================================================

def friedman_test_paired(df, block_col='cancer_type', group_col='xai_method', value_col='total_hit_same_any', group_order=None):
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


def pairwise_wilcoxon_paired(df, block_col='cancer_type', group_col='xai_method', value_col='total_hit_same_any', group_order=None):
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
        r_rb, n_nonzero = _rank_biserial_paired(x, y)
        results.append({
            'Group_1': g1,
            'Group_2': g2,
            'Statistic': float(stat) if stat is not None else np.nan,
            'p_value': float(p_value),
            'n_blocks': int(len(x)),
            'n_nonzero': int(n_nonzero),
            'Effect_Size_RankBiserial': float(r_rb) if r_rb is not None else np.nan,
            'Median_1': float(np.median(x)),
            'Median_2': float(np.median(y)),
            'Median_Diff': float(np.median(x) - np.median(y)),
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

    group_to_idx = {g: i for i, g in enumerate(list(group_order))}
    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low
    if y_range <= 0:
        y_range = 1.0

    for level, (_, row) in enumerate(sig_pairs.iterrows()):
        i = group_to_idx.get(row['Group_1'])
        j = group_to_idx.get(row['Group_2'])
        if i is None or j is None:
            continue
        if i > j:
            i, j = j, i

        y = y_high + y_range * (0.05 + 0.08 * level)
        ax.plot([i, j], [y, y], 'k-', lw=1.5)

        sig_symbol = _p_to_sig_symbol(row['p_fdr'])

        ax.text((i + j) / 2, y, sig_symbol, ha='center', va='bottom', fontsize=10, fontweight='bold')

    last_level = max(0, len(sig_pairs) - 1)
    y_top_needed = y_high + y_range * (0.05 + 0.08 * last_level + 0.08)
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


def _get_pairwise_p(pairwise_results, g1, g2, p_col='p_fdr'):
    if pairwise_results is None or len(pairwise_results) == 0:
        return np.nan
    mask = ((pairwise_results['Group_1'] == g1) & (pairwise_results['Group_2'] == g2)) | (
        (pairwise_results['Group_1'] == g2) & (pairwise_results['Group_2'] == g1)
    )
    row = pairwise_results.loc[mask]
    if len(row) == 0:
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
        p_val = _get_pairwise_p(pairwise_results, champion, g, p_col='p_fdr')
        if np.isnan(p_val):
            label = ''
        else:
            label = _p_to_sig_symbol(p_val)
        if label == '':
            continue
        ax.text(i, y, label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

    ax.set_ylim(y_low, y + y_range * max(0.05, y_pad_fraction * 0.6))


def _safe_percentile(x, q):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, q))


def summarize_across_blocks(df, group_col, value_col, group_order=None):
    rows = []
    if group_order is None:
        groups = sorted(df[group_col].dropna().unique().tolist())
    else:
        groups = [g for g in list(group_order) if g in df[group_col].unique()]

    for g in groups:
        data = df[df[group_col] == g][value_col].astype(float).values
        data = data[~np.isnan(data)]
        rows.append({
            group_col: g,
            'n_blocks': int(data.size),
            'mean': float(np.mean(data)) if data.size > 0 else np.nan,
            'std': float(np.std(data)) if data.size > 0 else np.nan,
            'min': float(np.min(data)) if data.size > 0 else np.nan,
            'q25': _safe_percentile(data, 25),
            'median': float(np.median(data)) if data.size > 0 else np.nan,
            'q75': _safe_percentile(data, 75),
            'max': float(np.max(data)) if data.size > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out['IQR'] = out['q75'] - out['q25']
    return out


def write_statistical_tests_csv(output_path, overall_row, pairwise_df, overall_title, pairwise_title):
    lines = []
    lines.append(f"# {overall_title}\n")
    overall_df = pd.DataFrame([overall_row])
    lines.append(overall_df.to_csv(index=False))
    lines.append("\n")
    lines.append(f"# {pairwise_title}\n")
    if pairwise_df is None:
        pairwise_df = pd.DataFrame()
    lines.append(pairwise_df.to_csv(index=False))
    with open(output_path, 'w') as f:
        f.writelines(lines)


# ============================================================================
# 可视化函数
# ============================================================================

def style_axis(ax):
    """设置坐标轴黑色加粗样式"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.tick_params(axis='both', colors='black', width=1.2)

def plot_method_boxplot(df, metric='mean_same_any', output_path=None, show_points=True):
    """
    按 XAI 方法绘制箱线图
    
    Args:
        df: all_model_statistics.csv 数据
        metric: 要展示的指标
        output_path: 保存路径
        show_points: 是否显示散点
    """
    pairwise_results = None
    
    # 绘图
    fig, ax = plt.subplots(figsize=(2.35, 2.5))
    
    # 箱线图
    # 根据当前指标的中位数，对 XAI 方法从小到大排序
    available_xai = [xai for xai in XAI_METHODS if xai in df['xai_method'].unique()]
    medians = (
        df[df['xai_method'].isin(available_xai)]
        .groupby('xai_method')[metric]
        .median()
    )
    xai_list = sorted(available_xai, key=lambda x: medians.loc[x])
    positions = range(len(xai_list))

    # 使用 XAI_COLORS 配置的颜色
    colors = [XAI_COLORS.get(xai, '#888888') for xai in xai_list]

    bp = ax.boxplot(
        [df[df['xai_method'] == xai][metric].values for xai in xai_list],
        positions=positions,
        tick_labels=xai_list,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_zorder(1)

    for k in ['whiskers', 'caps', 'medians', 'fliers']:
        for artist in bp.get(k, []):
            artist.set_zorder(2)
    
    # 添加散点
    if show_points:
        for i, xai in enumerate(xai_list):
            y = df[df['xai_method'] == xai][metric].values
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.9, s=6, color=colors[i], zorder=3)
    
    # 设置网格线：只显示横线
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # 标题和标签
    metric_name = metric.replace('_', ' ').title()
    ax.set_ylabel(metric_name, )
    #title = f'{metric_name} across XAI Methods'
    #ax.set_title(title)
    
    # 使用更友好的XAI方法名称作为x轴标签
    display_labels = [XAI_DISPLAY_NAMES.get(x, x) for x in xai_list]
    ax.set_xticks(list(positions))
    ax.set_xticklabels(display_labels, rotation=30, ha='center')

    pairwise_results = pairwise_wilcoxon_paired(
        df,
        block_col='cancer_type',
        group_col='xai_method',
        value_col=metric,
        group_order=xai_list,
    )
    group_scores = df.groupby('xai_method')[metric].median().to_dict()
    champion = max(list(xai_list), key=lambda g: (group_scores.get(g, -np.inf), str(g)))
    add_champion_significance_annotations(ax, pairwise_results, xai_list, champion)
    
    style_axis(ax)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"已保存: {output_path}")
    
    return fig, pairwise_results


def plot_category_boxplot(df, metric='total_hit_same_any', output_path=None):
    """
    按XAI类别（Gradient/Propagation/Perturbation）分组的箱线图
    
    Args:
        df: all_model_statistics.csv 数据
        metric: 要展示的指标
        output_path: 保存路径
    """
    # 添加类别列
    df_copy = df.copy()
    if 'category' not in df_copy.columns:
        df_copy['category'] = df_copy['xai_method'].map(XAI_CATEGORIES)
    
    # 过滤掉没有类别映射的行
    df_copy = df_copy[df_copy['category'].notna()]
    
    # 按类别中位数排序
    category_order = (
        df_copy.groupby('category')[metric]
        .median()
        .sort_values()
        .index.tolist()
    )
    
    fig, ax = plt.subplots(figsize=(1.8, 2.5))
    
    # 使用 CATEGORY_COLORS 配置的颜色
    colors = [CATEGORY_COLORS.get(cat, '#888888') for cat in category_order]
    
    # 箱线图
    bp = ax.boxplot([df_copy[df_copy['category'] == cat][metric].values for cat in category_order],
                     positions=range(len(category_order)),
                     patch_artist=True,
                     widths=0.6,
                     showfliers=False,
                     boxprops=dict(edgecolor="black", linewidth=1),
                     whiskerprops=dict(color="black", linewidth=1),
                     capprops=dict(color="black", linewidth=1),
                     medianprops=dict(color="black", linewidth=1))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_zorder(1)

    for k in ['whiskers', 'caps', 'medians', 'fliers']:
        for artist in bp.get(k, []):
            artist.set_zorder(2)
    
    # 添加散点
    for i, cat in enumerate(category_order):
        y = df_copy[df_copy['category'] == cat][metric].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.9, s=7, color=colors[i], zorder=3)
    
    # 设置网格线：只显示横线
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # 标签
    metric_name = metric.replace('_', ' ').title()
    #title = f'{metric_name} across XAI Categories'
    #ax.set_title(title)

    pairwise_df = pairwise_wilcoxon_paired(
        df_copy,
        block_col='cancer_type',
        group_col='category',
        value_col=metric,
        group_order=category_order,
    )

    ax.set_ylabel(metric_name)
    ax.set_xlabel('XAI Category')
    ax.set_xticks(list(range(len(category_order))))
    ax.set_xticklabels(category_order, rotation=30, ha='center')

    add_significance_annotations_from_results(ax, pairwise_df, category_order, max_annotations=None)
    
    style_axis(ax)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"已保存: {output_path}")
    
    return fig, pairwise_df


def plot_heatmap(pivot_file, output_path=None, title=None, df_raw=None, metric_col=None):
    """绘制 cancer_type × XAI 方法 的 hit_same_any 热力图
    行: 15个癌种
    列: 6个XAI方法
    值: 对应癌种的 hit_same_any 在所有模型上的中位数
    """
    df_pivot = pd.read_csv(pivot_file, index_col=0)

    cancers = [ct for ct in CANCER_TYPES if ct in df_pivot.index]
    methods = [m for m in XAI_METHODS if m in df_pivot.columns]

    df_pivot = df_pivot.loc[cancers, methods]
    df_pivot = df_pivot.rename(columns={m: XAI_DISPLAY_NAMES.get(m, m) for m in methods})
    method_display_order = [XAI_DISPLAY_NAMES.get(m, m) for m in methods]

    pivot = df_pivot.T
    pivot = pivot.reindex(index=method_display_order)

    pw_by_cancer = {}
    champion_by_cancer = {}
    if df_raw is not None and metric_col is not None:
        for cancer in cancers:
            sub = df_raw[(df_raw['cancer_type'] == cancer) & (df_raw['xai_method'].isin(methods))].copy()
            if len(sub) == 0:
                continue
            if 'model_idx' not in sub.columns:
                continue

            sub['xai_display'] = sub['xai_method'].map(lambda x: XAI_DISPLAY_NAMES.get(x, x))
            order = method_display_order

            pw = pairwise_wilcoxon_paired(
                sub,
                block_col='model_idx',
                group_col='xai_display',
                value_col=metric_col,
                group_order=order,
            )
            wide = (
                sub.pivot_table(index='model_idx', columns='xai_display', values=metric_col, aggfunc='median')
                .reindex(columns=order)
                .dropna(axis=0, how='any')
            )
            scores = wide.median(axis=0).to_dict() if wide.shape[0] > 0 else None
            if scores is None or len(scores) == 0:
                continue
            champion = max(list(order), key=lambda g: (scores.get(g, -np.inf), str(g)))
            pw_by_cancer[cancer] = pw
            champion_by_cancer[cancer] = champion

    annot = pivot.copy().astype(object)
    for m in annot.index:
        for c in annot.columns:
            val = annot.loc[m, c]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                annot.loc[m, c] = ''
                continue
            label = ''
            if c in pw_by_cancer and c in champion_by_cancer:
                champion = champion_by_cancer[c]
                if m != champion:
                    p_val = _get_pairwise_p(pw_by_cancer[c], champion, m, p_col='p_fdr')
                    if not np.isnan(p_val):
                        label = _p_to_sig_symbol(p_val)
            annot.loc[m, c] = f"{float(val):.1f}\n{label}" if label != '' else f"{float(val):.1f}"

    fig, ax = plt.subplots(figsize=(6.89, 3))
    custom_cmap = LinearSegmentedColormap.from_list(
        'xai_db',
        ["#1a2a6c", "white", "#b21f1f"]
    )
    sns.heatmap(
        pivot,
        annot=annot.values,
        fmt='',
        cmap=custom_cmap,
        linewidths=0.8,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 9},
        cbar_kws={'label': metric_col.replace('_', ' ').title() if metric_col else 'Value'}
    )

    #if title is None:
        #title = Path(pivot_file).stem.replace('pivot_', '').replace('_', ' ').title()
    #ax.set_title(title, fontweight='bold')
    #ax.set_xlabel('Cancer Type', fontweight='bold')
    #ax.set_ylabel('XAI Method', fontweight='bold')
    plt.xticks(rotation=30, ha='center')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"已保存: {output_path}")

    return fig


def compare_xai_within_each_db(df, db_cols, aggregation='s2'):
    """在每个数据库内比较 6 种 XAI 的差异，并做配对统计检验。

    设计：以 cancer_type 作为 block（配对），在每个数据库列上对 6 个 XAI 做 Friedman + 两两 Wilcoxon。
    """
    xai_list = [xai for xai in XAI_METHODS if xai in df['xai_method'].unique()]
    if not xai_list:
        return None, None

    xai_display_order = [XAI_DISPLAY_NAMES.get(x, x) for x in xai_list]

    friedman_rows = []
    pairwise_all = []

    # 预聚合：对每个 (cancer_type, xai_method) 取中位数，避免每个数据库重复 groupby
    cols = [col for _, col in db_cols if col in df.columns]
    if len(cols) == 0:
        return None, None
    grouped = (
        df.groupby(['cancer_type', 'xai_method'], as_index=False)[cols]
        .median()
    )

    for db_name, col in db_cols:
        if col not in df.columns:
            continue

        # 为保证配对检验成立，这里统一以 cancer_type 为 block，使用预聚合后的每癌种中位数
        df_test = grouped[['cancer_type', 'xai_method', col]].copy()
        df_test = df_test[df_test['xai_method'].isin(xai_list)].copy()
        df_test['xai_display'] = df_test['xai_method'].map(lambda x: XAI_DISPLAY_NAMES.get(x, x))
        block_col = 'cancer_type'

        stat, p_value, kendalls_w, n_blocks = friedman_test_paired(
            df_test,
            block_col=block_col,
            group_col='xai_display',
            value_col=col,
            group_order=xai_display_order,
        )
        friedman_rows.append({
            'database': db_name,
            'db_col': col,
            'aggregation': 'per_cancer_median',
            'Statistic': stat,
            'p_value': p_value,
            'kendalls_w': kendalls_w,
            'n_blocks': n_blocks,
            'n_groups': len(xai_list),
        })

        pw = pairwise_wilcoxon_paired(
            df_test,
            block_col=block_col,
            group_col='xai_display',
            value_col=col,
            group_order=xai_display_order,
        )
        if pw is not None and len(pw) > 0:
            pw = pw.copy()
            pw['database'] = db_name
            pw['db_col'] = col
            pw['aggregation'] = 'per_cancer_median'
            pairwise_all.append(pw)

    friedman_df = pd.DataFrame(friedman_rows) if len(friedman_rows) > 0 else None
    pairwise_df = pd.concat(pairwise_all, ignore_index=True) if len(pairwise_all) > 0 else None
    return friedman_df, pairwise_df


def plot_db_xai_hit_heatmap(df, output_path=None, aggregation='model'):
    """绘制 数据库 × XAI 方法 的 hit_same_any 热力图

    行: 6个XAI方法
    列: 4个数据库 (OncoKB, DGIdb, OpenTargets, CancerMine)
    值: 对应数据库的 hit_same_any 在所有模型上的中位数
    """
    # 定义数据库列和显示名称
    db_cols = [
        ('OncoKB',       'oncokb_hit_same_any'),
        ('DGIdb',        'dgidb_hit_same_any'),
        ('OpenTargets',  'opentargets_hit_same_any'),
        ('CancerMine',   'cancermine_hit_same_any'),
    ]

    # 过滤出实际存在的列
    db_cols = [(name, col) for name, col in db_cols if col in df.columns]
    if not db_cols:
        print("⚠ 在 all_model_statistics.csv 中未找到 *_hit_same_any 相关列，跳过数据库×XAI热力图绘制")
        return None

    xai_list = [xai for xai in XAI_METHODS if xai in df['xai_method'].unique()]
    if not xai_list:
        print("⚠ 数据中未找到任何XAI方法，跳过数据库×XAI热力图绘制")
        return None

    # 计算矩阵: 行=XAI, 列=数据库
    # aggregation == 's2' 时预聚合为每个 (cancer_type, xai_method) 的中位数，避免重复 groupby
    values = []
    grouped = None
    if aggregation == 's2':
        cols = [col for _, col in db_cols]
        grouped = (
            df.groupby(['cancer_type', 'xai_method'], as_index=False)[cols]
            .median()
        )

    for xai in xai_list:
        row_vals = []
        if aggregation == 's2' and grouped is not None:
            sub = grouped[grouped['xai_method'] == xai]
            for _, col in db_cols:
                tmp = sub[col].values
                val = float(np.median(tmp)) if tmp.size > 0 else np.nan
                row_vals.append(val)
        else:
            sub = df[df['xai_method'] == xai]
            for _, col in db_cols:
                data = sub[col].values
                if len(data) == 0:
                    row_vals.append(np.nan)
                else:
                    row_vals.append(float(np.median(data)))
        values.append(row_vals)

    col_labels = [name for name, _ in db_cols]
    index_labels = [XAI_DISPLAY_NAMES.get(x, x) for x in xai_list]
    mat = pd.DataFrame(values, index=index_labels, columns=col_labels)

    # 显著性标注：对每个数据库列，在癌种维度做配对检验（champion vs others）
    annot = mat.copy().astype(object)
    if aggregation == 's2':
        xai_display_order = [XAI_DISPLAY_NAMES.get(x, x) for x in xai_list]
        pw_by_db = {}
        champion_by_db = {}

        for db_name, col in db_cols:
            if col not in df.columns:
                continue

            if grouped is None:
                df_test = (
                    df.groupby(['cancer_type', 'xai_method'], as_index=False)[col]
                    .median()
                )
            else:
                df_test = grouped[['cancer_type', 'xai_method', col]].copy()

            df_test = df_test[df_test['xai_method'].isin(xai_list)].copy()
            df_test['xai_display'] = df_test['xai_method'].map(lambda x: XAI_DISPLAY_NAMES.get(x, x))

            pw = pairwise_wilcoxon_paired(
                df_test,
                block_col='cancer_type',
                group_col='xai_display',
                value_col=col,
                group_order=xai_display_order,
            )

            wide = (
                df_test.pivot_table(index='cancer_type', columns='xai_display', values=col, aggfunc='median')
                .reindex(columns=xai_display_order)
                .dropna(axis=0, how='any')
            )
            scores = wide.median(axis=0).to_dict() if wide.shape[0] > 0 else None
            if scores is None or len(scores) == 0:
                continue

            champion = max(list(xai_display_order), key=lambda g: (scores.get(g, -np.inf), str(g)))
            pw_by_db[db_name] = pw
            champion_by_db[db_name] = champion

        for xai_disp in annot.index:
            for db_name in annot.columns:
                val = mat.loc[xai_disp, db_name]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    annot.loc[xai_disp, db_name] = ''
                    continue
                label = ''
                if db_name in pw_by_db and db_name in champion_by_db:
                    champion = champion_by_db[db_name]
                    if xai_disp != champion:
                        p_val = _get_pairwise_p(pw_by_db[db_name], champion, xai_disp, p_col='p_fdr')
                        if not np.isnan(p_val):
                            label = _p_to_sig_symbol(p_val)
                annot.loc[xai_disp, db_name] = f"{float(val):.1f}\n{label}" if label != '' else f"{float(val):.1f}"
    else:
        for xai_disp in annot.index:
            for db_name in annot.columns:
                val = mat.loc[xai_disp, db_name]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    annot.loc[xai_disp, db_name] = ''
                else:
                    annot.loc[xai_disp, db_name] = f"{float(val):.1f}"

    fig, ax = plt.subplots(figsize=(2.35, 2.6))
    custom_cmap = LinearSegmentedColormap.from_list(
        'xai_db',
        ['#FFB347', '#E74C3C', '#8E44AD']   # 黄 → 红 → 蓝
    )
    sns.heatmap(mat, annot=annot.values, fmt='', cmap=custom_cmap,
                linewidths=0.5, linecolor='white', ax=ax)

    #if aggregation == 's2':
    #    title = 'Database × XAI hit_same_any (median of per-cancer medians)'
    #else:
    #    title = 'Database × XAI hit_same_any (median)'
    #ax.set_title(title)
    ax.set_xlabel('Database')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    #ax.set_ylabel('XAI Method')
    
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
        print(f"已保存: {output_path}")

    return fig, mat
# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("生物学合理性可视化与统计检验")
    print("=" * 80)
    
    # 创建输出目录
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    all_stats_file = OUTPUT_DIR / "all_model_statistics.csv"
    
    if not all_stats_file.exists():
        print(f"\n⚠ 未找到文件: {all_stats_file}")
        print("请先运行 02_calculate_gene_scores.py --all")
        return
    
    print(f"\n加载数据: {all_stats_file}")
    df = pd.read_csv(all_stats_file)
    print(f"  数据形状: {df.shape}")
    print(f"  XAI方法: {df['xai_method'].unique()}")
    print(f"  癌种: {df['cancer_type'].unique()}")
    
    # 核心指标：只使用 total_hit_same_any（总当前癌种命中数）
    metric = 'total_hit_same_any'

    df_cancer = (
        df.groupby(['cancer_type', 'xai_method'], as_index=False)[metric]
        .median()
    )
    df_category_cancer = df_cancer.copy()
    df_category_cancer['category'] = df_category_cancer['xai_method'].map(XAI_CATEGORIES)
    df_category_cancer = df_category_cancer[df_category_cancer['category'].notna()]
    df_category_cancer = (
        df_category_cancer.groupby(['cancer_type', 'category'], as_index=False)[metric]
        .median()
    )

    category_order = (
        df_category_cancer.groupby('category')[metric]
        .median()
        .sort_values()
        .index.tolist()
    )

    # ====== Supplementary CSVs (match Prognostic_comparison_plots style) ======
    methods = [m for m in XAI_METHODS if m in df_cancer['xai_method'].unique()]
    method_display_order = [XAI_DISPLAY_NAMES.get(m, m) for m in methods]

    df_fig1 = df_cancer.copy()
    df_fig1['Method'] = df_fig1['xai_method'].map(lambda x: XAI_DISPLAY_NAMES.get(x, x))
    df_fig1_plot = df_fig1[['cancer_type', 'Method', metric]].rename(columns={'cancer_type': 'Cancer', metric: metric})
    fig1_plot_path = viz_dir / 'supplementary_fig1_plot_data_method_per_cancer_median.csv'
    df_fig1_plot.to_csv(fig1_plot_path, index=False)
    print(f"  已保存: {fig1_plot_path}")

    df_fig1_summary = summarize_across_blocks(df_fig1_plot, group_col='Method', value_col=metric, group_order=method_display_order)
    df_fig1_summary = df_fig1_summary.rename(columns={'n_blocks': 'n_cancers'})
    fig1_summary_path = viz_dir / 'supplementary_fig1_summary_across_cancers_by_method.csv'
    df_fig1_summary.to_csv(fig1_summary_path, index=False)
    print(f"  已保存: {fig1_summary_path}")

    df_fig2_plot = df_category_cancer.copy()
    df_fig2_plot = df_fig2_plot[['cancer_type', 'category', metric]].rename(columns={'cancer_type': 'Cancer', 'category': 'Category', metric: metric})
    fig2_plot_path = viz_dir / 'supplementary_fig2_plot_data_category_per_cancer_median.csv'
    df_fig2_plot.to_csv(fig2_plot_path, index=False)
    print(f"  已保存: {fig2_plot_path}")

    df_fig2_summary = summarize_across_blocks(df_fig2_plot, group_col='Category', value_col=metric, group_order=category_order)
    df_fig2_summary = df_fig2_summary.rename(columns={'n_blocks': 'cancer_number'})
    fig2_summary_path = viz_dir / 'supplementary_fig2_summary_across_cancers_by_category.csv'
    df_fig2_summary.to_csv(fig2_summary_path, index=False)
    print(f"  已保存: {fig2_summary_path}")

    # Fig4 热图数据文件（对齐 Prognostic_comparison_plots 风格）
    pivot_median_file = viz_dir / "cancer_xai_hit_heatmap_data.csv"
    pivot_df = df_cancer.pivot(index='cancer_type', columns='xai_method', values=metric)
    desired_row_order = [ct for ct in CANCER_TYPES if ct in pivot_df.index]
    pivot_df = pivot_df.loc[desired_row_order]
    desired_col_order = [xai for xai in XAI_METHODS if xai in pivot_df.columns]
    pivot_df = pivot_df[desired_col_order]
    pivot_df.to_csv(pivot_median_file, index=True)
    print(f"  已保存: {pivot_median_file}")
    
    # ========== 目标1: 总体评估不同XAI的生物学合理性（箱线图）==========
    print("\n" + "=" * 80)
    print("1. 总体评估不同XAI的生物学合理性（箱线图）")
    print("=" * 80)
    
    output_path = viz_dir / f"fig1_boxplot_xai_{metric}.png"
    fig, pairwise_results = plot_method_boxplot(df_cancer, metric=metric, output_path=output_path)
    plt.close()

    # 额外输出（匹配 Prognostic_comparison_plots）：statistical_tests_XAI_methods.csv
    df_test_m = df_cancer.copy()
    df_test_m['Method'] = df_test_m['xai_method'].map(lambda x: XAI_DISPLAY_NAMES.get(x, x))
    stat_m, p_m, w_m, n_m = friedman_test_paired(
        df_test_m,
        block_col='cancer_type',
        group_col='Method',
        value_col=metric,
        group_order=method_display_order,
    )
    pw_m = pairwise_wilcoxon_paired(
        df_test_m,
        block_col='cancer_type',
        group_col='Method',
        value_col=metric,
        group_order=method_display_order,
    )
    if pw_m is not None and len(pw_m) > 0:
        pw_m = pw_m[[
            'Group_1', 'Group_2', 'Statistic', 'p_value', 'n_blocks',
            'Median_1', 'Median_2', 'Median_Diff', 'p_fdr', 'sig_fdr'
        ]]
    overall_row_m = {
        'Test': 'Friedman (paired by Cancer)',
        'Statistic': stat_m,
        'p_value': p_m,
        'Effect_Size_Kendalls_W': w_m,
        'n_cancers': n_m,
        'Significant': 'Yes' if (p_m is not None and float(p_m) < 0.05) else 'No',
    }
    stat_path_m = viz_dir / 'statistical_tests_XAI_methods.csv'
    write_statistical_tests_csv(
        stat_path_m,
        overall_row=overall_row_m,
        pairwise_df=pw_m,
        overall_title='Overall Test: Friedman Test (paired by Cancer)',
        pairwise_title='Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction',
    )
    print(f"  已保存: {stat_path_m}")
    
    # ========== 目标2: 总体评估三种类别XAI的生物学合理性（箱线图）==========
    print("\n" + "=" * 80)
    print("2. 总体评估三种类别XAI的生物学合理性（箱线图）")
    print("=" * 80)
    
    output_path = viz_dir / f"fig2_boxplot_category_{metric}.png"
    fig, category_pairwise_df = plot_category_boxplot(df_category_cancer, metric=metric, output_path=output_path)
    plt.close()

    # 额外输出（匹配 Prognostic_comparison_plots）：statistical_tests_XAI_categories.csv
    stat_c, p_c, w_c, n_c = friedman_test_paired(
        df_category_cancer,
        block_col='cancer_type',
        group_col='category',
        value_col=metric,
        group_order=category_order,
    )
    pw_c = pairwise_wilcoxon_paired(
        df_category_cancer,
        block_col='cancer_type',
        group_col='category',
        value_col=metric,
        group_order=category_order,
    )
    if pw_c is not None and len(pw_c) > 0:
        pw_c = pw_c[[
            'Group_1', 'Group_2', 'Statistic', 'p_value', 'n_blocks',
            'Median_1', 'Median_2', 'Median_Diff', 'p_fdr', 'sig_fdr'
        ]]
    overall_row_c = {
        'Test': 'Friedman (paired by Cancer)',
        'Statistic': stat_c,
        'p_value': p_c,
        'Effect_Size_Kendalls_W': w_c,
        'n_cancers': n_c,
        'Significant': 'Yes' if (p_c is not None and float(p_c) < 0.05) else 'No',
    }
    stat_path_c = viz_dir / 'statistical_tests_XAI_categories.csv'
    write_statistical_tests_csv(
        stat_path_c,
        overall_row=overall_row_c,
        pairwise_df=pw_c,
        overall_title='Overall Test: Friedman Test (paired by Cancer)',
        pairwise_title='Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction',
    )
    print(f"  已保存: {stat_path_c}")
    
    # ========== 目标3: 分数据库比较不同XAI的生物学合理性（热图）==========
    print("\n" + "=" * 80)
    print("3. 分数据库比较不同XAI的生物学合理性（热图）")
    print("=" * 80)
    
    output_path = viz_dir / f"fig3_heatmap_database_xai_{metric}.png"
    _, fig3_mat = plot_db_xai_hit_heatmap(df, output_path=output_path, aggregation='s2')
    plt.close()

    # 同时输出：每个数据库内比较6个XAI（配对检验，block=cancer_type）
    db_cols = [
        ('OncoKB',       'oncokb_hit_same_any'),
        ('DGIdb',        'dgidb_hit_same_any'),
        ('OpenTargets',  'opentargets_hit_same_any'),
        ('CancerMine',   'cancermine_hit_same_any'),
    ]
    db_cols = [(name, col) for name, col in db_cols if col in df.columns]

    friedman_db, pairwise_db = compare_xai_within_each_db(df, db_cols=db_cols, aggregation='s2')

    # 额外输出（匹配 Prognostic_comparison_plots）：Fig3 热图数据 + within_database_method_tests_*.csv
    # Fig3 热图数据（XAI × Database），与 plot_db_xai_hit_heatmap(aggregation='s2') 完全一致
    fig3_data_path = viz_dir / 'database_xai_hit_heatmap_data.csv'
    fig3_mat.to_csv(fig3_data_path, index=True)
    print(f"  已保存: {fig3_data_path}")

    if friedman_db is not None and len(friedman_db) > 0:
        within_overall = friedman_db.copy()
        within_overall = within_overall.rename(columns={
            'database': 'Database',
            'Statistic': 'Statistic',
            'p_value': 'p_value',
            'kendalls_w': 'Effect_Size_Kendalls_W',
            'n_blocks': 'n_blocks',
        })
        within_overall['Test'] = 'Friedman (paired by Cancer)'
        within_overall['k_methods'] = int(fig3_mat.shape[0])
        within_overall = within_overall[['Database', 'Test', 'Statistic', 'p_value', 'Effect_Size_Kendalls_W', 'n_blocks', 'k_methods']]
        out_path = viz_dir / 'within_database_method_tests_overall.csv'
        within_overall.to_csv(out_path, index=False)
        print(f"  已保存: {out_path}")

    if pairwise_db is not None and len(pairwise_db) > 0:
        within_pairwise = pairwise_db.copy()
        within_pairwise = within_pairwise.rename(columns={'database': 'Database'})
        within_pairwise = within_pairwise[['Group_1', 'Group_2', 'Statistic', 'p_value', 'n_blocks', 'p_fdr', 'sig_fdr', 'Database']]
        out_path = viz_dir / 'within_database_method_tests_pairwise.csv'
        within_pairwise.to_csv(out_path, index=False)
        print(f"  已保存: {out_path}")
    
    # ========== 目标4: 分癌症比较不同XAI的生物学合理性（热图）==========
    print("\n" + "=" * 80)
    print("4. 分癌症比较不同XAI的生物学合理性（热图）")
    print("=" * 80)
    
    output_path = viz_dir / f"fig4_heatmap_cancer_xai_{metric}.png"
    fig = plot_heatmap(pivot_median_file, output_path=output_path, df_raw=df, metric_col=metric)
    plt.close()

    # 额外输出（匹配 Prognostic_comparison_plots）：within_cancer_method_tests_*.csv + supplementary_fig4_paired_wide_data_used_for_tests.csv
    cancers = [ct for ct in CANCER_TYPES if ct in df['cancer_type'].unique()]
    within_overall_rows = []
    within_pairwise_rows = []
    paired_wide_frames = []

    for cancer in cancers:
        sub = df[(df['cancer_type'] == cancer) & (df['xai_method'].isin(methods))].copy()
        if len(sub) == 0:
            continue
        if 'model_idx' not in sub.columns:
            continue

        sub['Method'] = sub['xai_method'].map(lambda x: XAI_DISPLAY_NAMES.get(x, x))
        wide = (
            sub.pivot_table(index='model_idx', columns='Method', values=metric, aggfunc='median')
            .reindex(columns=method_display_order)
            .dropna(axis=0, how='any')
        )
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue

        wide_reset = wide.reset_index().rename(columns={'model_idx': 'replicate_id'})
        paired_wide_frames.append(wide_reset)

        # 统计检验用的长表
        long = wide.reset_index().melt(id_vars=['model_idx'], var_name='Method', value_name=metric)

        stat_wc, p_wc, w_wc, n_wc = friedman_test_paired(
            long,
            block_col='model_idx',
            group_col='Method',
            value_col=metric,
            group_order=method_display_order,
        )
        within_overall_rows.append({
            'Cancer': cancer,
            'Test': 'Friedman (paired by model_idx)',
            'Statistic': stat_wc,
            'p_value': p_wc,
            'Effect_Size_Kendalls_W': w_wc,
            'n_blocks': n_wc,
            'k_methods': len(method_display_order),
        })

        pw_wc = pairwise_wilcoxon_paired(
            long,
            block_col='model_idx',
            group_col='Method',
            value_col=metric,
            group_order=method_display_order,
        )
        if pw_wc is not None and len(pw_wc) > 0:
            pw_wc = pw_wc[['Group_1', 'Group_2', 'Statistic', 'p_value', 'n_blocks', 'p_fdr', 'sig_fdr']].copy()
            pw_wc['Cancer'] = cancer
            within_pairwise_rows.append(pw_wc)

    if len(within_overall_rows) > 0:
        within_overall_df = pd.DataFrame(within_overall_rows)
        out_path = viz_dir / 'within_cancer_method_tests_overall.csv'
        within_overall_df.to_csv(out_path, index=False)
        print(f"  已保存: {out_path}")

    if len(within_pairwise_rows) > 0:
        within_pairwise_df = pd.concat(within_pairwise_rows, ignore_index=True)
        out_path = viz_dir / 'within_cancer_method_tests_pairwise.csv'
        within_pairwise_df.to_csv(out_path, index=False)
        print(f"  已保存: {out_path}")

    if len(paired_wide_frames) > 0:
        paired_wide_df = pd.concat(paired_wide_frames, ignore_index=True)
        out_path = viz_dir / 'supplementary_fig4_paired_wide_data_used_for_tests.csv'
        paired_wide_df.to_csv(out_path, index=False)
        print(f"  已保存: {out_path}")
    
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n所有输出保存在: {viz_dir}")
    print("\n主要文件(对齐 Prognostic_comparison_plots):")
    print(f"  - statistical_tests_XAI_methods.csv")
    print(f"  - statistical_tests_XAI_categories.csv")
    print(f"  - supplementary_fig1_plot_data_method_per_cancer_median.csv")
    print(f"  - supplementary_fig1_summary_across_cancers_by_method.csv")
    print(f"  - supplementary_fig2_plot_data_category_per_cancer_median.csv")
    print(f"  - supplementary_fig2_summary_across_cancers_by_category.csv")
    print(f"  - database_xai_hit_heatmap_data.csv")
    print(f"  - within_database_method_tests_overall.csv")
    print(f"  - within_database_method_tests_pairwise.csv")
    print(f"  - cancer_xai_hit_heatmap_data.csv")
    print(f"  - within_cancer_method_tests_overall.csv")
    print(f"  - within_cancer_method_tests_pairwise.csv")
    print(f"  - supplementary_fig4_paired_wide_data_used_for_tests.csv")


if __name__ == "__main__":
    main()
