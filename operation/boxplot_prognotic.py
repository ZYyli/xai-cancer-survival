import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap

# 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 可选：强制使用 Arial 字体（BBRC 推荐）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 设置绘图风格（与03_visualize_and_test.py一致）
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 8


CATEGORY_COLORS = {
    'Gradient': '#e6cf88',
    'Perturbation': '#e6889f',
    'Propagation': '#88cee6'
}

XAI_COLORS = {
    'IG': '#F49E39',
    'shap': '#D3C0A3',
    'deepshap': '#9271B1',
    'LRP': '#A6DAEF',
    'DeepLIFT': '#66c1a4',
    'PFI': '#E26472'
}

METHOD_COLORS = {
    'IG': XAI_COLORS['IG'],
    'G-SHAP': XAI_COLORS['shap'],
    'D-SHAP': XAI_COLORS['deepshap'],
    'LRP': XAI_COLORS['LRP'],
    'DeepLIFT': XAI_COLORS['DeepLIFT'],
    'PFI': XAI_COLORS['PFI']
}


def _apply_axis_style(ax):
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.1)
    ax.tick_params(axis='both', colors='black', width=1.2)


def _save_figure(fig, out_file_png):
    fig.savefig(out_file_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_file_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()


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


cancer_list = [
    'COADREAD', 'LUSC', 'HNSC', 'STAD', 'BLCA', 'BRCA', 'LUAD', 'PAAD',
    'LIHC', 'SKCM', 'KIRC', 'UCEC', 'KIRP', 'GBMLGG', 'LGG'
]


methods_dirs = {
    'G-SHAP': "/home/zuoyiyi/SNN/TCGA/shap_results_2",
    'IG': "/home/zuoyiyi/SNN/TCGA/IG_results_2",
    'LRP': "/home/zuoyiyi/SNN/TCGA/LRP_results_2",
    'PFI': "/home/zuoyiyi/SNN/TCGA/PFI_results_2",
    'D-SHAP': "/home/zuoyiyi/SNN/TCGA/deepshap_results_2",
    'DeepLIFT': "/home/zuoyiyi/SNN/TCGA/DeepLIFT_results_2"
}

col_name = 'cox_prognostic_factors_raw'

output_dir = "/home/zuoyiyi/SNN/TCGA/Prognostic_comparison_plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 统计检验函数
# ============================================================================


def friedman_test_paired(df, block_col='Cancer', group_col='Method', value_col='N_Factors', group_order=None):
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
    kendalls_w = stat / (n * (k - 1)) if (n > 0 and k > 1) else np.nan

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


def pairwise_wilcoxon_paired(df, block_col='Cancer', group_col='Method', value_col='N_Factors', group_order=None):
    wide = df.pivot_table(index=block_col, columns=group_col, values=value_col, aggfunc='median')
    if group_order is not None:
        wide = wide.reindex(columns=list(group_order))
    wide = wide.dropna(axis=0, how='any')

    methods = list(wide.columns)
    results = []
    for m1, m2 in combinations(methods, 2):
        x = wide[m1].values
        y = wide[m2].values
        stat, p_value = _wilcoxon_safe(x, y)

        results.append({
            'Group_1': m1,
            'Group_2': m2,
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

    group_to_idx = {g: i for i, g in enumerate(list(group_order))}
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
        ax.plot([g1_idx, g2_idx], [y, y], 'k-', lw=1.2)

        sig_symbol = _p_to_sig_symbol(row['p_fdr'])
        if sig_symbol == '':
            continue

        ax.text((g1_idx + g2_idx) / 2, y, sig_symbol,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

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


def pairwise_wilcoxon_paired_from_wide(wide, group_order=None):
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
            'n_blocks': int(len(x))
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df['sig_fdr'] = results_df['p_fdr'] < 0.05
    return results_df


def within_cancer_pairwise_tests(df_long, cancer_order, hue_order, value_col='N_Factors', x_col='Cancer', hue_col='Method'):
    k = int(len(hue_order))
    if k <= 0:
        return None, None, {}, {}

    per_cancer_rows_overall = []
    per_cancer_rows_pairwise = []
    pw_by_cancer = {}
    champion_by_cancer = {}

    for cancer in list(cancer_order):
        sub = df_long[df_long[x_col] == cancer].copy()
        if len(sub) == 0:
            continue

        sub['replicate_id'] = sub['repeat'].astype(str) + '_' + sub['fold'].astype(str)
        wide = sub.pivot_table(index='replicate_id', columns=hue_col, values=value_col, aggfunc='median')
        wide = wide.reindex(columns=list(hue_order))
        wide = wide.dropna(axis=0, how='any')
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue

        arrays = [wide[m].values for m in hue_order]
        stat, p_value = friedmanchisquare(*arrays)
        n_blocks = int(wide.shape[0])
        k_groups = int(wide.shape[1])
        kendalls_w = float(stat) / (n_blocks * (k_groups - 1)) if (n_blocks > 0 and k_groups > 1) else np.nan

        per_cancer_rows_overall.append({
            'Cancer': cancer,
            'Test': 'Friedman (paired by repeat+fold)',
            'Statistic': float(stat),
            'p_value': float(p_value),
            'Effect_Size_Kendalls_W': float(kendalls_w) if kendalls_w is not None else np.nan,
            'n_blocks': n_blocks,
            'k_methods': k_groups
        })

        pairwise = pairwise_wilcoxon_paired_from_wide(wide, group_order=hue_order)
        if pairwise is not None and len(pairwise) > 0:
            pairwise = pairwise.copy()
            pairwise['Cancer'] = cancer
            per_cancer_rows_pairwise.append(pairwise)

        scores = wide.median(axis=0).to_dict()
        champion = max(list(hue_order), key=lambda g: (scores.get(g, -np.inf), str(g)))
        pw_by_cancer[cancer] = pairwise
        champion_by_cancer[cancer] = champion

    overall_df = pd.DataFrame(per_cancer_rows_overall) if len(per_cancer_rows_overall) > 0 else pd.DataFrame()
    pairwise_df = pd.concat(per_cancer_rows_pairwise, axis=0, ignore_index=True) if len(per_cancer_rows_pairwise) > 0 else pd.DataFrame()
    return overall_df, pairwise_df, pw_by_cancer, champion_by_cancer


def within_category_method_tests(df_cancer_median, method_category_map, output_dir):
    rows_overall = []
    rows_pairwise = []

    df = df_cancer_median.copy()
    df['Category'] = df['Method'].map(method_category_map)

    for category, sub in df.groupby('Category'):
        methods = sorted(sub['Method'].dropna().unique().tolist())
        if len(methods) < 2:
            continue

        wide = sub.pivot_table(index='Cancer', columns='Method', values='N_Factors', aggfunc='median')
        wide = wide.dropna(subset=methods, how='any')
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

    out_overall = os.path.join(output_dir, 'within_category_method_tests_overall.csv')
    out_pairwise = os.path.join(output_dir, 'within_category_method_tests_pairwise.csv')
    overall_df.to_csv(out_overall, index=False)
    pairwise_df.to_csv(out_pairwise, index=False)

    if len(overall_df) > 0:
        print("\n" + "="*80)
        print("类别内部：方法差异检验（基于每癌种的Cancer×Method median，配对检验）")
        print("="*80)
        print(f"总体检验结果已保存到: {out_overall}")
        print(f"两两Wilcoxon结果已保存到: {out_pairwise}")
        print("\n总体检验:")
        print(overall_df.to_string(index=False))

        if len(pairwise_df) > 0:
            sig = pairwise_df[pairwise_df['sig_fdr']]
            print("\n两两比较（FDR<0.05）:")
            if len(sig) > 0:
                print(sig[['Category', 'Method_1', 'Method_2', 'p_value', 'p_fdr', 'Median_Diff']].to_string(index=False))
            else:
                print("  无显著差异的配对")

# 收集所有原始数据 (nested CV)
all_original_counts = []
for method, base_dir in methods_dirs.items():
    for cancer in cancer_list:
        file_raw = os.path.join(base_dir, cancer, f"{cancer}_detailed_results.csv")
        if os.path.exists(file_raw):
            df_raw = pd.read_csv(file_raw)
            if col_name in df_raw.columns:
                if 'repeat' in df_raw.columns and 'fold' in df_raw.columns:
                    sub = df_raw[['repeat', 'fold', col_name]].dropna(subset=[col_name]).copy()
                    all_original_counts.extend([(cancer, method, int(r), int(f), float(v)) for r, f, v in sub.values])
                else:
                    counts = df_raw[col_name].dropna()
                    all_original_counts.extend([(cancer, method, 0, int(i), float(v)) for i, v in enumerate(counts.values)])

# 生成图：方法箱线图 → 类别箱线图 → 热图 → 代表癌症箱线图
if all_original_counts:
    df_original = pd.DataFrame(all_original_counts, columns=['Cancer', 'Method', 'repeat', 'fold', 'N_Factors'])
    df_cancer_median = (
        df_original
        .groupby(['Cancer', 'Method'], as_index=False)['N_Factors']
        .median()
    )

    hue_order = ['D-SHAP', 'IG', 'DeepLIFT', 'G-SHAP', 'PFI', 'LRP']

    within_cancer_overall, within_cancer_pairwise, pw_by_cancer, champion_by_cancer = within_cancer_pairwise_tests(
        df_original,
        cancer_list,
        hue_order,
        value_col='N_Factors',
        x_col='Cancer',
        hue_col='Method',
    )

    if within_cancer_overall is not None and len(within_cancer_overall) > 0:
        out_overall = os.path.join(output_dir, 'within_cancer_method_tests_overall.csv')
        within_cancer_overall.to_csv(out_overall, index=False)
    if within_cancer_pairwise is not None and len(within_cancer_pairwise) > 0:
        out_pairwise = os.path.join(output_dir, 'within_cancer_method_tests_pairwise.csv')
        within_cancer_pairwise.to_csv(out_pairwise, index=False)


    # 生成第一张图：6种XAI方法整合15癌种（每癌种×方法的median点）
    method_order = df_cancer_median.groupby('Method')['N_Factors'].median().sort_values().index.tolist()
    colors_ordered = [METHOD_COLORS.get(m) for m in method_order]

    print("\n" + "="*80)
    print("第一张图：XAI方法比较的统计检验")
    print("="*80)
    stat, p_value, kendalls_w, n_blocks = friedman_test_paired(
        df_cancer_median, block_col='Cancer', group_col='Method', value_col='N_Factors', group_order=method_order
    )

    fig1, ax1 = plt.subplots(figsize=(2.35, 2.6))
    ax1 = sns.boxplot(
        data=df_cancer_median,
        x='Method',
        y='N_Factors',
        order=method_order,
        palette=colors_ordered,
        showfliers=False,
        ax=ax1,
        linecolor="black", linewidth=1
    )
    for i, m in enumerate(method_order):
        y = df_cancer_median[df_cancer_median['Method'] == m]['N_Factors'].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax1.scatter(x, y, alpha=0.9, s=6, color=colors_ordered[i], zorder=3)
    
    # 设置箱体透明度
    for patch in ax1.patches:
        patch.set_alpha(0.7)

    # 添加显著性标注（显示所有显著配对）
    pairwise_results_methods = pairwise_wilcoxon_paired(
        df_cancer_median, block_col='Cancer', group_col='Method', value_col='N_Factors', group_order=method_order
    )
    method_scores = df_cancer_median.groupby('Method')['N_Factors'].median().to_dict()
    champion = max(list(method_order), key=lambda g: (method_scores.get(g, -np.inf), str(g)))
    add_champion_significance_annotations(ax1, pairwise_results_methods, method_order, champion)

    _apply_axis_style(ax1)
    #ax1.set_title("Number of Prognostic Factors by XAI Method\nAggregated", fontweight='bold')
    ax1.set_ylabel("Number of Prognostic Factors")
    ax1.set_xlabel("XAI Method", labelpad=15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    plt.tight_layout()

    # 保存第一张图
    out_file_aggregated = os.path.join(output_dir, "aggregated_XAI_prognostic_factors_boxplot.png")
    _save_figure(fig1, out_file_aggregated)
    print(f"\n📊 整合XAI方法箱线图已保存到: {out_file_aggregated}")

    # 保存统计检验结果
    if pairwise_results_methods is not None:
        stats_file_methods = os.path.join(output_dir, "statistical_tests_XAI_methods.csv")
        
        # 添加总体检验结果
        overall_stats = pd.DataFrame([{
            'Test': 'Friedman (paired by Cancer)',
            'Statistic': stat,
            'p_value': p_value,
            "Effect_Size_Kendalls_W": kendalls_w,
            'n_cancers': n_blocks,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        }])
        with open(stats_file_methods, 'w') as f:
            f.write("# Overall Test: Friedman Test (paired by Cancer)\n")
            overall_stats.to_csv(f, index=False)
            f.write("\n# Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction\n")
            pairwise_results_methods.to_csv(f, index=False)
        print(f"📊 统计检验结果已保存到: {stats_file_methods}")

    fig1_plot_data = df_cancer_median.copy()
    fig1_plot_data['Cancer'] = pd.Categorical(fig1_plot_data['Cancer'], categories=cancer_list, ordered=True)
    fig1_plot_data['Method'] = pd.Categorical(fig1_plot_data['Method'], categories=method_order, ordered=True)
    fig1_plot_data = fig1_plot_data.sort_values(['Cancer', 'Method'])
    fig1_plot_data_file = os.path.join(output_dir, "supplementary_fig1_plot_data_method_per_cancer_median.csv")
    fig1_plot_data.to_csv(fig1_plot_data_file, index=False)
    print(f"📊 图1作图数据已保存到: {fig1_plot_data_file}")

    fig1_summary = _describe_by_group(fig1_plot_data, ['Method'], 'N_Factors')
    fig1_summary = fig1_summary.rename(columns={'n_models': 'n_cancers'})
    fig1_summary['Method'] = pd.Categorical(fig1_summary['Method'], categories=method_order, ordered=True)
    fig1_summary = fig1_summary.sort_values(['Method'])
    fig1_summary_file = os.path.join(output_dir, "supplementary_fig1_summary_across_cancers_by_method.csv")
    fig1_summary.to_csv(fig1_summary_file, index=False)
    print(f"📊 图1跨癌种汇总统计已保存到: {fig1_summary_file}")

    # 生成第二张图：3类XAI类别整合15癌种（每癌种×类别的median点）
    method_category_map = {
        'IG': 'Gradient',
        'G-SHAP': 'Gradient',
        'D-SHAP': 'Propagation',
        'LRP': 'Propagation',
        'DeepLIFT': 'Propagation',
        'PFI': 'Perturbation'
    }
    within_category_method_tests(df_cancer_median, method_category_map, output_dir)

    df_category = df_cancer_median.copy()
    df_category['Category'] = df_category['Method'].map(method_category_map)
    df_category = df_category.groupby(['Cancer', 'Category'], as_index=False)['N_Factors'].median()
    category_order = df_category.groupby('Category')['N_Factors'].median().sort_values().index.tolist()

    print("\n" + "="*80)
    print("第二张图：XAI类别比较的统计检验")
    print("="*80)
    stat_cat, p_value_cat, kendalls_w_cat, n_blocks_cat = friedman_test_paired(
        df_category, block_col='Cancer', group_col='Category', value_col='N_Factors', group_order=category_order
    )

    fig2, ax2 = plt.subplots(figsize=(1.8, 2.8))
    category_colors = [CATEGORY_COLORS.get(c) for c in category_order]
    ax2 = sns.boxplot(
        data=df_category,
        x='Category',
        y='N_Factors',
        order=category_order,
        palette=category_colors,
        showfliers=False,
        ax=ax2,
        linecolor="black", linewidth=1
    )
    for i, cat in enumerate(category_order):
        y = df_category[df_category['Category'] == cat]['N_Factors'].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax2.scatter(x, y, alpha=0.9, s=7, color=category_colors[i], zorder=3)
    
    # 设置箱体透明度
    for patch in ax2.patches:
        patch.set_alpha(0.7)

    # 添加显著性标注（显示所有显著配对）
    pairwise_results_category = pairwise_wilcoxon_paired(
        df_category, block_col='Cancer', group_col='Category', value_col='N_Factors', group_order=category_order
    )
    pairwise_results_category = add_significance_annotations_from_results(
        ax2, pairwise_results_category, category_order, max_annotations=None
    )

    _apply_axis_style(ax2)
    #ax2.set_title("Number of Prognostic Factors by XAI Category\nAggregated Across 15 Cancer Types", fontweight='bold')
    ax2.set_ylabel("Number of Prognostic Factors")
    ax2.set_xlabel("XAI Category", labelpad=15)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
    plt.tight_layout()

    out_file_category = os.path.join(output_dir, "XAI_category_prognostic_factors_boxplot.png")
    _save_figure(fig2, out_file_category)
    print(f"\n📊 XAI类别箱线图已保存到: {out_file_category}")

    if pairwise_results_category is not None:
        stats_file_category = os.path.join(output_dir, "statistical_tests_XAI_categories.csv")
        
        # 添加总体检验结果
        overall_stats_cat = pd.DataFrame([{
            'Test': 'Friedman (paired by Cancer)',
            'Statistic': stat_cat,
            'p_value': p_value_cat,
            "Effect_Size_Kendalls_W": kendalls_w_cat,
            'n_cancers': n_blocks_cat,
            'Significant': 'Yes' if p_value_cat < 0.05 else 'No'
        }])
        with open(stats_file_category, 'w') as f:
            f.write("# Overall Test: Friedman Test (paired by Cancer)\n")
            overall_stats_cat.to_csv(f, index=False)
            f.write("\n# Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction\n")
            pairwise_results_category.to_csv(f, index=False)
        print(f"📊 统计检验结果已保存到: {stats_file_category}")

    fig2_plot_data = df_category.copy()
    fig2_plot_data['Cancer'] = pd.Categorical(fig2_plot_data['Cancer'], categories=cancer_list, ordered=True)
    fig2_plot_data['Category'] = pd.Categorical(fig2_plot_data['Category'], categories=category_order, ordered=True)
    fig2_plot_data = fig2_plot_data.sort_values(['Cancer', 'Category'])
    fig2_plot_data_file = os.path.join(output_dir, "supplementary_fig2_plot_data_category_per_cancer_median.csv")
    fig2_plot_data.to_csv(fig2_plot_data_file, index=False)
    print(f"📊 图2作图数据已保存到: {fig2_plot_data_file}")

    fig2_summary = _describe_by_group(fig2_plot_data, ['Category'], 'N_Factors')
    fig2_summary = fig2_summary.rename(columns={'n_models': 'cancer_number'})
    fig2_summary['Category'] = pd.Categorical(fig2_summary['Category'], categories=category_order, ordered=True)
    fig2_summary = fig2_summary.sort_values(['Category'])
    fig2_summary_file = os.path.join(output_dir, "supplementary_fig2_summary_across_cancers_by_category.csv")
    fig2_summary.to_csv(fig2_summary_file, index=False)
    print(f"📊 图2跨癌种汇总统计已保存到: {fig2_summary_file}")

    # 生成第三张图：15种癌症中6种XAI方法比较热图
    print("\n" + "="*80)
    print("第三张图：15种癌症中6种XAI方法比较热图")
    print("="*80)
    mat = (
        df_cancer_median
        .pivot(index='Method', columns='Cancer', values='N_Factors')
        .reindex(index=hue_order, columns=cancer_list)
    )

    annot = mat.copy().astype(object)
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

    fig3, ax3 = plt.subplots(figsize=(6.89, 3))
    custom_cmap = LinearSegmentedColormap.from_list(
        'xai_db',
        ["#1a2a6c", "white", "#b21f1f"]
    )
    sns.heatmap(
        mat,
        annot=annot.values,
        fmt='',
        cmap=custom_cmap,
        linewidths=0.8,
        linecolor='white',
        ax=ax3,
        annot_kws={'size': 8},
        cbar_kws={'label': 'Number of Prognostic Factors'}
    )

    #ax3.set_title("Number of Prognostic Factors Across 15 Cancers\nComparison of Six XAI Methods", fontweight='bold')
    ax3.set_xlabel("Cancer Type")
    ax3.set_ylabel("XAI Method")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='center')
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
    plt.tight_layout()

    out_file_original = os.path.join(output_dir, "prognostic_factors_heatmap.png")
    _save_figure(fig3, out_file_original)
    print(f"📊 第三张图（热图）已保存到: {out_file_original}")

    # 生成第四张图：LGG单癌种的6种XAI方法比较（50个抽样模型）
    print("\n" + "="*80)
    print("第四张图：LGG单癌种的6种XAI方法比较")
    print("="*80)

    lgg_data = df_original[df_original['Cancer'] == 'LGG'].copy()
    if len(lgg_data) > 0:
        # 按照预后因子数量中位数排序
        lgg_method_order = lgg_data.groupby('Method')['N_Factors'].median().sort_values().index.tolist()
        lgg_colors_ordered = [METHOD_COLORS.get(m) for m in lgg_method_order]

        # 统计检验
        lgg_data['replicate_id'] = lgg_data['repeat'].astype(str) + '_' + lgg_data['fold'].astype(str)
        lgg_wide = lgg_data.pivot_table(index='replicate_id', columns='Method', values='N_Factors', aggfunc='median')
        lgg_wide = lgg_wide.reindex(columns=lgg_method_order).dropna(axis=0, how='any')

        stat_lgg, p_value_lgg, kendalls_w_lgg = np.nan, 1.0, np.nan
        n_blocks_lgg = int(lgg_wide.shape[0])
        if lgg_wide.shape[0] >= 2 and lgg_wide.shape[1] >= 2:
            arrays_lgg = [lgg_wide[m].values for m in lgg_method_order]
            stat_lgg, p_value_lgg = friedmanchisquare(*arrays_lgg)
            k_lgg = int(lgg_wide.shape[1])
            kendalls_w_lgg = float(stat_lgg) / (n_blocks_lgg * (k_lgg - 1)) if (n_blocks_lgg > 0 and k_lgg > 1) else np.nan

        fig4, ax4 = plt.subplots(figsize=(2.5, 2.6))
        sns.violinplot(
            data=lgg_data,
            x='Method',
            y='N_Factors',
            order=lgg_method_order,
            palette=lgg_colors_ordered,
            ax=ax4,
            inner=None,
            cut=0,
            linewidth=0.8
        )
        for coll in ax4.collections:
            try:
                coll.set_alpha(0.5)
            except Exception:
                pass
        ax4 = sns.boxplot(
            data=lgg_data,
            x='Method',
            y='N_Factors',
            order=lgg_method_order,
            palette=lgg_colors_ordered,
            showfliers=False,
            ax=ax4,
            linecolor="black", linewidth=1
        )
        for i, m in enumerate(lgg_method_order):
            y = lgg_data[lgg_data['Method'] == m]['N_Factors'].values
            x = np.random.normal(i, 0.04, size=len(y))
            ax4.scatter(x, y, alpha=0.9, s=3, color=lgg_colors_ordered[i], zorder=3)
        for patch in ax4.patches:
            patch.set_alpha(0.7)

        pairwise_results_lgg = pairwise_wilcoxon_paired_from_wide(lgg_wide, group_order=lgg_method_order)
        if pairwise_results_lgg is not None and len(pairwise_results_lgg) > 0:
            lgg_method_scores = lgg_data.groupby('Method')['N_Factors'].median().to_dict()
            champion_lgg = max(list(lgg_method_order), key=lambda g: (lgg_method_scores.get(g, -np.inf), str(g)))
            add_champion_significance_annotations(ax4, pairwise_results_lgg, lgg_method_order, champion_lgg)

        _apply_axis_style(ax4)
        #ax4.set_title("Number of Prognostic Factors by XAI Method\nLGG Cancer Type (50 Sampled Models)", fontweight='bold')
        ax4.set_ylabel("Number of Prognostic Factors")
        ax4.set_xlabel("XAI Method", labelpad=15)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=30)
        plt.tight_layout()

        out_file_lgg = os.path.join(output_dir, "LGG_XAI_prognostic_factors_boxplot.png")
        _save_figure(fig4, out_file_lgg)
        print(f"\n📊 LGG单癌种XAI方法箱线图已保存到: {out_file_lgg}")

        fig4_plot_data = lgg_data.copy()
        fig4_plot_data['Method'] = pd.Categorical(fig4_plot_data['Method'], categories=lgg_method_order, ordered=True)
        fig4_plot_data = fig4_plot_data.sort_values(['replicate_id', 'Method'])
        fig4_plot_data_file = os.path.join(output_dir, "supplementary_fig4_plot_data_LGG_models.csv")
        fig4_plot_data.to_csv(fig4_plot_data_file, index=False)
        print(f"📊 图4作图数据已保存到: {fig4_plot_data_file}")

        fig4_summary = _describe_by_group(fig4_plot_data, ['Method'], 'N_Factors')
        fig4_summary['Method'] = pd.Categorical(fig4_summary['Method'], categories=lgg_method_order, ordered=True)
        fig4_summary = fig4_summary.sort_values(['Method'])
        fig4_summary_file = os.path.join(output_dir, "supplementary_fig4_summary_across_models_by_method.csv")
        fig4_summary.to_csv(fig4_summary_file, index=False)
        print(f"📊 图4跨模型汇总统计已保存到: {fig4_summary_file}")

        fig4_paired_wide_file = os.path.join(output_dir, "supplementary_fig4_paired_wide_data_used_for_tests.csv")
        lgg_wide.reset_index().to_csv(fig4_paired_wide_file, index=False)
        print(f"📊 图4配对检验使用的数据已保存到: {fig4_paired_wide_file}")

        if pairwise_results_lgg is not None and len(pairwise_results_lgg) > 0:
            stats_file_lgg = os.path.join(output_dir, "statistical_tests_LGG_XAI_methods.csv")
            overall_stats_lgg = pd.DataFrame([{
                'Test': 'Friedman (paired by repeat+fold)',
                'Statistic': stat_lgg,
                'p_value': p_value_lgg,
                "Effect_Size_Kendalls_W": kendalls_w_lgg,
                'n_models': n_blocks_lgg,
                'Significant': 'Yes' if p_value_lgg < 0.05 else 'No'
            }])
            with open(stats_file_lgg, 'w') as f:
                f.write("# Overall Test: Friedman Test (paired by repeat+fold)\n")
                overall_stats_lgg.to_csv(f, index=False)
                f.write("\n# Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction\n")
                pairwise_results_lgg.to_csv(f, index=False)
            print(f"📊 LGG统计检验结果已保存到: {stats_file_lgg}")
    else:
        print("警告：未找到LGG数据")

    print("\n" + "="*80)
    print("所有分析完成！")
    print("="*80)