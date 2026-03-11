"""
补充计算 PFI 的 Spearman 相关性
从已有的 detailed_results.csv 计算，无需重新运行 PFI 分析
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

# 配置
PFI_DIR = "/home/zuoyiyi/SNN/TCGA/PFI_bootstrap_results"
CANCER_LIST = ['BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC', 'KIRP', 
               'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC']

def compute_correlations():
    """从 detailed_results.csv 计算相关性并更新汇总文件"""
    
    all_summaries = []
    
    for cancer in CANCER_LIST:
        detail_file = os.path.join(PFI_DIR, cancer, f"{cancer}_bootstrap_detailed_results.csv")
        
        if not os.path.exists(detail_file):
            print(f"⚠️ {cancer}: 详细结果文件不存在")
            continue
        
        df = pd.read_csv(detail_file)
        
        # 检查必要列
        if 'cindex' not in df.columns or 'cox_prognostic_factors_raw' not in df.columns:
            print(f"⚠️ {cancer}: 缺少必要列")
            continue
        
        cindex = df['cindex'].dropna()
        cox_prog_raw = df['cox_prognostic_factors_raw'].dropna()
        
        # 确保长度一致
        min_len = min(len(cindex), len(cox_prog_raw))
        cindex = cindex[:min_len]
        cox_prog_raw = cox_prog_raw[:min_len]
        
        # 计算相关性
        spearman_rho, spearman_p = spearmanr(cindex, cox_prog_raw)
        pearson_r, pearson_p = pearsonr(cindex, cox_prog_raw)
        
        # 计算其他统计量
        summary = {
            'cancer': cancer,
            'n_bootstrap_models': len(df),
            'cindex_mean': df['cindex'].mean(),
            'cindex_std': df['cindex'].std(),
            'cindex_min': df['cindex'].min(),
            'cindex_max': df['cindex'].max(),
            
            # PFI 预后因子统计
            'pfi_prognostic_factors_fdr_mean': df['pfi_prognostic_factors_fdr'].mean() if 'pfi_prognostic_factors_fdr' in df.columns else np.nan,
            'pfi_prognostic_factors_fdr_std': df['pfi_prognostic_factors_fdr'].std() if 'pfi_prognostic_factors_fdr' in df.columns else np.nan,
            'pfi_prognostic_factors_raw_mean': df['pfi_prognostic_factors_raw'].mean() if 'pfi_prognostic_factors_raw' in df.columns else np.nan,
            'pfi_prognostic_factors_raw_std': df['pfi_prognostic_factors_raw'].std() if 'pfi_prognostic_factors_raw' in df.columns else np.nan,
            
            # Cox 预后因子统计
            'cox_prognostic_factors_fdr_mean': df['cox_prognostic_factors_fdr'].mean() if 'cox_prognostic_factors_fdr' in df.columns else np.nan,
            'cox_prognostic_factors_fdr_std': df['cox_prognostic_factors_fdr'].std() if 'cox_prognostic_factors_fdr' in df.columns else np.nan,
            'cox_risk_factors_fdr_mean': df['cox_risk_factors_fdr'].mean() if 'cox_risk_factors_fdr' in df.columns else np.nan,
            'cox_risk_factors_fdr_std': df['cox_risk_factors_fdr'].std() if 'cox_risk_factors_fdr' in df.columns else np.nan,
            'cox_protective_factors_fdr_mean': df['cox_protective_factors_fdr'].mean() if 'cox_protective_factors_fdr' in df.columns else np.nan,
            'cox_protective_factors_fdr_std': df['cox_protective_factors_fdr'].std() if 'cox_protective_factors_fdr' in df.columns else np.nan,
            'cox_prognostic_factors_raw_mean': df['cox_prognostic_factors_raw'].mean() if 'cox_prognostic_factors_raw' in df.columns else np.nan,
            'cox_prognostic_factors_raw_std': df['cox_prognostic_factors_raw'].std() if 'cox_prognostic_factors_raw' in df.columns else np.nan,
            'cox_risk_factors_raw_mean': df['cox_risk_factors_raw'].mean() if 'cox_risk_factors_raw' in df.columns else np.nan,
            'cox_risk_factors_raw_std': df['cox_risk_factors_raw'].std() if 'cox_risk_factors_raw' in df.columns else np.nan,
            'cox_protective_factors_raw_mean': df['cox_protective_factors_raw'].mean() if 'cox_protective_factors_raw' in df.columns else np.nan,
            'cox_protective_factors_raw_std': df['cox_protective_factors_raw'].std() if 'cox_protective_factors_raw' in df.columns else np.nan,
            
            # 相关性 - 这是关键补充部分
            'corr_cindex_pfi_prognostic_raw_pearson': pearsonr(cindex, df['pfi_prognostic_factors_raw'][:min_len])[0] if 'pfi_prognostic_factors_raw' in df.columns else np.nan,
            'corr_cindex_pfi_prognostic_raw_pearson_p': pearsonr(cindex, df['pfi_prognostic_factors_raw'][:min_len])[1] if 'pfi_prognostic_factors_raw' in df.columns else np.nan,
            'corr_cindex_pfi_prognostic_raw_spearman': spearmanr(cindex, df['pfi_prognostic_factors_raw'][:min_len])[0] if 'pfi_prognostic_factors_raw' in df.columns else np.nan,
            'corr_cindex_pfi_prognostic_raw_spearman_p': spearmanr(cindex, df['pfi_prognostic_factors_raw'][:min_len])[1] if 'pfi_prognostic_factors_raw' in df.columns else np.nan,
            
            'corr_cindex_cox_prognostic_raw_pearson': pearson_r,
            'corr_cindex_cox_prognostic_raw_pearson_p': pearson_p,
            'corr_cindex_cox_prognostic_raw_spearman': spearman_rho,
            'corr_cindex_cox_prognostic_raw_spearman_p': spearman_p,
        }
        
        all_summaries.append(summary)
        print(f"✅ {cancer}: Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.4f}")
    
    # 保存新的汇总文件
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        output_file = os.path.join(PFI_DIR, "all_cancers_bootstrap_summary.csv")
        
        # 备份旧文件
        if os.path.exists(output_file):
            backup_file = output_file.replace('.csv', '_backup.csv')
            os.rename(output_file, backup_file)
            print(f"\n📁 旧文件已备份为: {backup_file}")
        
        summary_df.to_csv(output_file, index=False)
        print(f"\n✅ 新汇总文件已保存: {output_file}")
        print(f"   包含 {len(all_summaries)} 种癌症的相关性数据")
        
        # 打印相关性矩阵
        print("\n=== Cox预后因子 vs C-index Spearman相关性 ===")
        for s in all_summaries:
            sig = "*" if s['corr_cindex_cox_prognostic_raw_spearman_p'] < 0.05 else ""
            print(f"  {s['cancer']}: ρ = {s['corr_cindex_cox_prognostic_raw_spearman']:.3f}{sig}")


if __name__ == "__main__":
    compute_correlations()
