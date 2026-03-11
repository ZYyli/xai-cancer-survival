#!/usr/bin/env python3
"""
bootstrap_boxplot_analysis.py

用途：
- 读取每种癌症的100次bootstrap c-index结果
- 绘制15种癌症的c-index分布箱线图（每种癌症一个箱线图，展示100次bootstrap的分布）
- 生成汇总表（均值、中位数、标准差、置信区间等）
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

def load_bootstrap_results(results_dir):
    """
    从bootstrap结果文件夹中读取所有癌症的c-index数组
    Returns:
        dict: {cancer_name: {'cindex_array': array}}
    """
    bootstrap_data = {}
    for cancer_folder in os.listdir(results_dir):
        cancer_path = os.path.join(results_dir, cancer_folder)
        if not os.path.isdir(cancer_path):
            continue
        cindex_file = os.path.join(cancer_path, 'cindex_array.npy')
        if os.path.exists(cindex_file):
            try:
                cindex_array = np.load(cindex_file)
                valid_indices = ~np.isnan(cindex_array)
                bootstrap_data[cancer_folder] = {
                    'cindex_array': cindex_array[valid_indices]
                }
                print(f"✅ {cancer_folder}: 加载 {len(cindex_array[valid_indices])} 个有效结果")
            except Exception as e:
                print(f"⚠️ {cancer_folder} 读取失败: {e}")
        else:
            print(f"⚠️ {cancer_folder} 缺少文件: cindex_array.npy")
    return bootstrap_data

def test_cindex_vs_random(bootstrap_data):
    """
    对每种癌症的 c-index 进行单样本 t 检验，检验是否显著大于 0.5（随机水平）
    使用 FDR（Benjamini-Hochberg 方法）进行多重检验校正
    
    参数:
    - bootstrap_data: dict，{cancer_name: {'cindex_array': array}}
    
    返回:
    - ttest_results: dict，{cancer_name: {'t_stat': float, 'p_raw': float, 
                                         'p_fdr': float, 'sig_fdr': str, 
                                         'mean': float, 'std': float, 'n': int}}
    """
    print("\n=== C-index vs 随机水平(0.5)的统计检验 ===")
    print("检验方法: 单样本单侧 t 检验")
    print("H0: μ = 0.5 (模型无预测能力)")
    print("H1: μ > 0.5 (模型优于随机)")
    print("-" * 70)
    
    ttest_results = {}
    p_values_for_correction = []
    cancer_list = []
    
    for cancer_name, data in bootstrap_data.items():
        cindex_values = data['cindex_array']
        
        if len(cindex_values) < 3:
            print(f"{cancer_name}: 样本量不足 (n={len(cindex_values)})")
            ttest_results[cancer_name] = {
                't_stat': np.nan, 'p_raw': np.nan, 'mean': np.nan,
                'std': np.nan, 'n': len(cindex_values)
            }
            continue
        
        # 单样本 t 检验：检验均值是否显著大于 0.5
        # alternative='greater' 表示单侧检验（H1: mean > 0.5）
        t_stat, p_value = stats.ttest_1samp(cindex_values, 0.5, alternative='greater')
        
        mean_cindex = np.mean(cindex_values)
        std_cindex = np.std(cindex_values, ddof=1)
        
        # 收集 p 值用于多重检验校正
        p_values_for_correction.append(p_value)
        cancer_list.append(cancer_name)
        
        ttest_results[cancer_name] = {
            't_stat': t_stat,
            'p_raw': p_value,
            'mean': mean_cindex,
            'std': std_cindex,
            'n': len(cindex_values)
        }
        
        print(f"{cancer_name:8s}: mean={mean_cindex:.4f}±{std_cindex:.4f}, "
              f"t={t_stat:6.3f}, p={p_value:.6f} "
              f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # FDR 多重检验校正（Benjamini-Hochberg 方法）
    if len(p_values_for_correction) > 0:
        reject_fdr, pvals_corrected_fdr, _, _ = multipletests(
            p_values_for_correction, alpha=0.05, method='fdr_bh'
        )
        
        print("\n" + "=" * 70)
        print("多重检验校正结果 (FDR - Benjamini-Hochberg 方法):")
        print("=" * 70)
        print(f"{'Cancer':<12s} {'Raw p':<12s} {'FDR p':<12s} {'Significance':<12s}")
        print("-" * 70)
        
        for i, cancer_name in enumerate(cancer_list):
            p_fdr = pvals_corrected_fdr[i]
            
            # 根据 FDR 校正后的 p 值确定显著性标记
            if p_fdr < 0.001:
                sig_fdr = '***'
            elif p_fdr < 0.01:
                sig_fdr = '**'
            elif p_fdr < 0.05:
                sig_fdr = '*'
            else:
                sig_fdr = ''
            
            # 更新结果，添加 FDR 校正后的 p 值和显著性
            ttest_results[cancer_name]['p_fdr'] = p_fdr
            ttest_results[cancer_name]['sig_fdr'] = sig_fdr
            
            print(f"{cancer_name:<12s} {ttest_results[cancer_name]['p_raw']:<12.6f} "
                  f"{p_fdr:<12.6f} {sig_fdr:<12s}")
        
        print("-" * 70)
        print(f"FDR 校正后显著的癌症: {sum(reject_fdr)}/{len(reject_fdr)} "
              f"({sum(reject_fdr)/len(reject_fdr)*100:.1f}%)")
    
    return ttest_results

def create_bootstrap_boxplot(bootstrap_data, output_dir, ttest_results=None, figsize=(12, 6)):
    """生成15种癌症的c-index箱线图"""
    plot_data = []
    for cancer, data in bootstrap_data.items():
        for value in data['cindex_array']:
            plot_data.append({'Cancer': cancer, 'C-index': value})
    df_plot = pd.DataFrame(plot_data)

    # === 统计平均 C-index ===
    summary_results = {}
    for cancer, data in bootstrap_data.items():
        cindex_values = data['cindex_array']
        summary_results[cancer] = {
            'mean_cindex': np.mean(cindex_values),
            'median_cindex': np.median(cindex_values),
            'n_bootstrap': len(cindex_values)
        }

    # === 按平均 C-index 排序 ===
    cancer_order = sorted(summary_results.keys(), key=lambda c: summary_results[c]['mean_cindex'])

    # 打印结果
    for cancer in cancer_order:
        print(f"📊 {cancer}: Mean C-index={summary_results[cancer]['mean_cindex']:.4f}, "
            f"Median={summary_results[cancer]['median_cindex']:.4f}")

    # 绘图
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    box_plot = sns.boxplot(
        data=df_plot,
        x='Cancer',
        y='C-index',
        order=cancer_order,
        color='#135AA9',
        linewidth=1.5,
        flierprops={"marker": "x", "markersize": 5}
    )

    # 绘制随机水平参考线（y=0.5）
    box_plot.axhline(
        y=0.5, 
        color='#D32F2F',  # 红色
        linestyle=':', 
        linewidth=2.0,
        label='Random (C-index=0.5)',
        zorder=2
    )

    # 绘制整体平均线
    overall_mean = df_plot['C-index'].mean()
    box_plot.axhline(
        y=overall_mean,
        color='#E98E1E',
        linestyle='--',
        linewidth=1.5,
        label=f'Overall: {overall_mean:.3f}',
        zorder=2
    )

    # 美化
    box_plot.set_title(
        'Concordance Index Distribution for Different Cancers (bootstrap)',
        fontsize=14, fontweight='bold', pad=20
    )
    box_plot.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    box_plot.set_ylabel('C-index', fontsize=12, fontweight='bold')
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=30, ha='right', fontsize=12)
    box_plot.set_ylim(0.3, 1.0)

    # 标注显著性星号（如果提供了 t 检验结果）
    if ttest_results is not None:
        # 获取当前 y 轴范围
        y_min, y_max = box_plot.get_ylim()
        # 扩大 y 轴范围以容纳星号
        box_plot.set_ylim(y_min, y_max + 0.05)
        
        for i, cancer in enumerate(cancer_order):
            if cancer in ttest_results and 'sig_fdr' in ttest_results[cancer]:
                sig = ttest_results[cancer]['sig_fdr']
                
                if sig:  # 只标注显著的
                    # 获取箱线图顶部位置
                    cancer_data = df_plot[df_plot['Cancer'] == cancer]['C-index']
                    # 计算箱线图的上须位置（Q3 + 1.5*IQR，或最大值，取较小者）
                    q1 = cancer_data.quantile(0.25)
                    q3 = cancer_data.quantile(0.75)
                    iqr = q3 - q1
                    upper_whisker = min(q3 + 1.5 * iqr, cancer_data.max())
                    
                    # 星号位置在上须上方
                    star_y = upper_whisker + 0.02
                    
                    box_plot.text(i, star_y, sig, 
                                 ha='center', va='bottom', 
                                 fontsize=14, fontweight='bold', color='black')

    # 图例
    box_plot.legend(fontsize=11, loc='upper left', frameon=False, facecolor='white', framealpha=0.8)

    plt.tight_layout()
    plt.grid(False)
    output_file = os.path.join(output_dir, 'bootstrap_cindex_boxplot.png')
    box_plot.figure.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"📊 箱线图已保存到: {output_file}")
    return summary_results

def generate_bootstrap_summary_table(bootstrap_data, output_dir, ttest_results=None):
    """
    生成汇总统计表
    
    参数:
    - bootstrap_data: dict，bootstrap 数据
    - output_dir: 输出目录
    - ttest_results: dict，统计检验结果（可选）
    """
    summary_stats = []
    for cancer, data in bootstrap_data.items():
        cindex_values = data['cindex_array']
        
        row = {
            'Cancer': cancer,
            'N_Bootstrap': len(cindex_values),
            'Mean': np.mean(cindex_values),
            'Median': np.median(cindex_values),
            'Std': np.std(cindex_values, ddof=1),
            'Min': np.min(cindex_values),
            'Max': np.max(cindex_values),
            'Q25': np.percentile(cindex_values, 25),
            'Q75': np.percentile(cindex_values, 75),
            'CI_Lower': np.percentile(cindex_values, 2.5),
            'CI_Upper': np.percentile(cindex_values, 97.5)
        }
        
        # 如果提供了统计检验结果，添加到汇总表
        if ttest_results is not None and cancer in ttest_results:
            row['t_statistic'] = ttest_results[cancer].get('t_stat', np.nan)
            row['p_value_raw'] = ttest_results[cancer].get('p_raw', np.nan)
            row['p_value_fdr'] = ttest_results[cancer].get('p_fdr', np.nan)
            row['significance_fdr'] = ttest_results[cancer].get('sig_fdr', '')
        
        summary_stats.append(row)
    
    df_summary = pd.DataFrame(summary_stats).sort_values('Cancer').reset_index(drop=True)
    summary_file = os.path.join(output_dir, 'bootstrap_cindex_summary.csv')
    df_summary.to_csv(summary_file, index=False, float_format='%.6f')
    print(f"📋 汇总表已保存到: {summary_file}")
    return df_summary

def main():
    parser = argparse.ArgumentParser(description='Bootstrap C-index分析（箱线图+汇总表）')
    parser.add_argument('--input_dir', type=str, required=True, help='Bootstrap结果目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("🚀 开始分析Bootstrap C-index结果...")

    # 1. 加载数据
    bootstrap_data = load_bootstrap_results(args.input_dir)
    if not bootstrap_data:
        print("❌ 未找到有效数据")
        return

    # 2. 执行统计检验（C-index vs 0.5）
    print("\n📈 执行统计检验...")
    ttest_results = test_cindex_vs_random(bootstrap_data)
    
    # 保存统计检验结果到单独的 CSV
    ttest_rows = []
    for cancer, result in ttest_results.items():
        ttest_rows.append({
            'cancer': cancer,
            'n_bootstrap': result.get('n', 0),
            'mean_cindex': result.get('mean', np.nan),
            'std_cindex': result.get('std', np.nan),
            't_statistic': result.get('t_stat', np.nan),
            'p_value_raw': result.get('p_raw', np.nan),
            'p_value_fdr': result.get('p_fdr', np.nan),
            'significance_fdr': result.get('sig_fdr', '')
        })
    if ttest_rows:
        ttest_df = pd.DataFrame(ttest_rows)
        ttest_file = os.path.join(args.output_dir, 'cindex_vs_random_ttest_results.csv')
        ttest_df.to_csv(ttest_file, index=False, float_format='%.6f')
        print(f"✅ 统计检验结果已保存至: {ttest_file}")

    # 3. 生成箱线图（带显著性标记）
    print("\n📊 生成箱线图...")
    create_bootstrap_boxplot(bootstrap_data, args.output_dir, ttest_results=ttest_results)

    # 4. 生成汇总表（包含统计检验结果）
    print("\n📋 生成汇总表...")
    generate_bootstrap_summary_table(bootstrap_data, args.output_dir, ttest_results=ttest_results)

    print("\n🎉 分析完成！结果已保存到:", args.output_dir)
    print("\n📁 输出文件:")
    print(f"  • bootstrap_cindex_boxplot.png/pdf - 箱线图（带显著性星号）")
    print(f"  • bootstrap_cindex_summary.csv - 汇总表（含统计检验）")
    print(f"  • cindex_vs_random_ttest_results.csv - 统计检验详细结果")

if __name__ == "__main__":
    main()
