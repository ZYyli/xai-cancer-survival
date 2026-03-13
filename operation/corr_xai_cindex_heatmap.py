"""
扇形热图：15种癌症 × 6种XAI方法的 Cox预后因子数量 vs C-index 相关性
- 横向：15种癌症
- 纵向：6种XAI方法
- 颜色：Spearman ρ 值
- 标注：显著性星号
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

# 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 可选：强制使用 Arial 字体（BBRC 推荐）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['font.size'] = 8

# ============================== 数据路径配置 ==============================
CANCER_LIST = ['COADREAD', 'LUSC', 'HNSC', 'STAD', 'BLCA', 'BRCA', 'LUAD', 'PAAD',
    'LIHC', 'SKCM', 'KIRC', 'UCEC', 'KIRP', 'GBMLGG', 'LGG']

TCGA_DIR = Path(os.environ.get('TCGA_DIR', Path(__file__).resolve().parents[1])).resolve()

XAI_METHODS = {
    'LRP': str(TCGA_DIR / 'LRP_bootstrap_results'),
    'PFI': str(TCGA_DIR / 'PFI_bootstrap_results'),
    'IG': str(TCGA_DIR / 'IG_bootstrap_results'),
    'DeepLIFT': str(TCGA_DIR / 'DeepLIFT_bootstrap_results'),
    'G-SHAP': str(TCGA_DIR / 'shap_bootstrap_results'),
    'D-SHAP': str(TCGA_DIR / 'deepshap_bootstrap_results')
}

OUTPUT_DIR = str(Path(os.environ.get('OUTPUT_DIR', str(TCGA_DIR / 'Prognostic_comparison_plots'))).resolve())

# ============================== 颜色配置 ==============================
HEATMAP_COLORS = ["#43A8A8", "white", "#D75F5F"]


def load_correlation_data():
    """
    直接从各XAI方法的all_cancers_bootstrap_summary.csv读取已计算好的Spearman相关系数
    返回：
    - df_corr: 相关系数矩阵 (癌症 × XAI方法)
    - df_p: p值矩阵
    - df_q: q值矩阵（BH-FDR）
    - df_sig: 显著性矩阵 (FDR q < 0.05)
    """
    xai_names = list(XAI_METHODS.keys())
    
    # 初始化结果矩阵
    corr_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    p_matrix = np.full((len(CANCER_LIST), len(xai_names)), np.nan)
    
    # 相关系数列名（Cox预后因子 vs C-index 的 Spearman）
    CORR_COL = 'corr_cindex_cox_prognostic_raw_spearman'
    P_COL = 'corr_cindex_cox_prognostic_raw_spearman_p'
    
    for j, (xai_name, base_dir) in enumerate(XAI_METHODS.items()):
        # 读取该XAI方法的汇总文件
        summary_file = os.path.join(base_dir, "all_cancers_bootstrap_summary.csv")
        
        if not os.path.exists(summary_file):
            print(f"⚠️ 汇总文件不存在: {summary_file}")
            continue
        
        try:
            df_summary = pd.read_csv(summary_file)
            
            # 检查必要的列
            if CORR_COL not in df_summary.columns or P_COL not in df_summary.columns:
                print(f"⚠️ {xai_name} 缺少相关系数列: {CORR_COL} 或 {P_COL}")
                print(f"   可用列: {[c for c in df_summary.columns if 'corr' in c.lower()]}")
                continue
            
            # 遍历癌症类型
            for i, cancer in enumerate(CANCER_LIST):
                row = df_summary[df_summary['cancer'] == cancer]
                
                if len(row) == 0:
                    print(f"⚠️ {xai_name} 中未找到癌症: {cancer}")
                    continue
                
                rho = row[CORR_COL].values[0]
                p_value = row[P_COL].values[0]
                
                corr_matrix[i, j] = rho
                p_matrix[i, j] = p_value
                
            print(f"✅ {xai_name}: 成功读取 {(~np.isnan(corr_matrix[:, j])).sum()} 个癌症的相关系数")
                
        except Exception as e:
            print(f"❌ 读取 {xai_name} 汇总文件失败: {e}")
    
    # 转换为DataFrame
    df_corr = pd.DataFrame(corr_matrix, index=CANCER_LIST, columns=xai_names)
    df_p = pd.DataFrame(p_matrix, index=CANCER_LIST, columns=xai_names)

    q = _bh_fdr(df_p.to_numpy())
    df_q = pd.DataFrame(q, index=df_p.index, columns=df_p.columns)
    df_sig = (df_q < 0.05) & df_q.notna()
    
    return df_corr, df_p, df_q, df_sig


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


def create_fan_heatmap(df_corr, df_sig, output_path):
    """
    创建热图
    - 横向：癌症类型
    - 纵向：XAI方法
    """
    fig, ax = plt.subplots(figsize=(6.89, 3))

    cancer_order = list(CANCER_LIST)
    xai_order = ['D-SHAP', 'IG', 'DeepLIFT', 'G-SHAP', 'PFI', 'LRP']

    custom_cmap = LinearSegmentedColormap.from_list('corr_rect', HEATMAP_COLORS)
    custom_cmap = custom_cmap.copy()
    custom_cmap.set_bad('lightgray')

    mat = df_corr.T.reindex(index=xai_order, columns=cancer_order)
    df_sig_xy = df_sig.reindex(index=cancer_order, columns=xai_order)
    mask = mat.isna()

    annot = pd.DataFrame('', index=mat.index, columns=mat.columns)
    for xai in mat.index:
        for cancer in mat.columns:
            val = mat.loc[xai, cancer]
            if pd.isna(val):
                continue
            is_sig = False
            try:
                is_sig = bool(df_sig_xy.loc[cancer, xai])
            except Exception:
                is_sig = False
            label = '*' if is_sig else ''
            annot.loc[xai, cancer] = f"{float(val):.2f}\n{label}" if label != '' else f"{float(val):.2f}"

    sns.heatmap(
        mat,
        mask=mask,
        annot=annot.values,
        fmt='',
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 8},
        cbar_kws={'label': 'Spearman ρ'}
    )

    ax.set_xlabel("Cancer Type")
    ax.set_ylabel("XAI Method")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"✅ 长方形热图已保存: {output_path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("加载数据并计算Spearman相关系数...")
    print("=" * 60)
    
    df_corr, df_p, df_q, df_sig = load_correlation_data()
    
    # 打印相关系数矩阵
    print("\n=== Spearman相关系数矩阵 ===")
    print(df_corr.round(3).to_string())
    
    print("\n=== 显著性矩阵 (FDR q < 0.05) ===")
    print(df_sig.to_string())

    rows = []
    for cancer in df_corr.index:
        for xai in df_corr.columns:
            r = df_corr.loc[cancer, xai]
            p = df_p.loc[cancer, xai]
            qv = df_q.loc[cancer, xai]
            sig = df_sig.loc[cancer, xai]
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
    out_csv = os.path.join(OUTPUT_DIR, "cindex_vs_cox_prognostic_heatmap_data.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ 热图补充数据已保存到: {out_csv}")
    
    output_path = os.path.join(OUTPUT_DIR, "heatmap_cindex_vs_cox_factors.png")
    create_fan_heatmap(df_corr, df_sig, output_path)
    
    # 统计显著相关的数量
    n_sig_positive = ((df_corr > 0) & df_sig).sum().sum()
    n_sig_negative = ((df_corr < 0) & df_sig).sum().sum()
    n_total = (~df_corr.isna()).sum().sum()
    
    print(f"\n=== 统计摘要 ===")
    print(f"总计算组合数: {n_total}")
    print(f"显著正相关数: {n_sig_positive}")
    print(f"显著负相关数: {n_sig_negative}")
    print(f"显著比例: {(n_sig_positive + n_sig_negative) / n_total * 100:.1f}%")


if __name__ == "__main__":
    main()