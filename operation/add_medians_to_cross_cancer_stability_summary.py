import os
import pandas as pd
import numpy as np


def _compute_medians_for_row(data_dir, cancer_type, xai_method, top_k):
    raw_path = os.path.join(
        data_dir,
        str(cancer_type),
        str(xai_method),
        f"pairwise_stability_raw_top{int(top_k)}.csv",
    )
    if not os.path.exists(raw_path):
        return np.nan, np.nan, np.nan

    df_raw = pd.read_csv(raw_path)
    if 'kuncheva' not in df_raw.columns or 'jaccard' not in df_raw.columns or 'rbo_09' not in df_raw.columns:
        return np.nan, np.nan, np.nan

    kuncheva_median = float(df_raw['kuncheva'].median()) if len(df_raw) > 0 else np.nan
    jaccard_median = float(df_raw['jaccard'].median()) if len(df_raw) > 0 else np.nan
    rbo_09_median = float(df_raw['rbo_09'].median()) if len(df_raw) > 0 else np.nan
    return kuncheva_median, jaccard_median, rbo_09_median


def add_medians_to_summary(data_dir):
    summary_path = os.path.join(data_dir, 'cross_cancer_stability_summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)

    if 'kuncheva_median' not in df.columns:
        df['kuncheva_median'] = np.nan
    if 'jaccard_median' not in df.columns:
        df['jaccard_median'] = np.nan
    if 'rbo_09_median' not in df.columns:
        df['rbo_09_median'] = np.nan

    required_cols = ['cancer_type', 'xai_method', 'top_k']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in summary: {c}")

    keys = df[required_cols].drop_duplicates().to_dict('records')
    medians_map = {}
    for rec in keys:
        k = (rec['cancer_type'], rec['xai_method'], int(rec['top_k']))
        medians_map[k] = _compute_medians_for_row(data_dir, rec['cancer_type'], rec['xai_method'], rec['top_k'])

    for idx, row in df.iterrows():
        k = (row['cancer_type'], row['xai_method'], int(row['top_k']))
        kun_m, jac_m, rbo_m = medians_map.get(k, (np.nan, np.nan, np.nan))
        df.at[idx, 'kuncheva_median'] = kun_m
        df.at[idx, 'jaccard_median'] = jac_m
        df.at[idx, 'rbo_09_median'] = rbo_m

    df.to_csv(summary_path, index=False)
    return summary_path


if __name__ == '__main__':
    data_dir = '/home/zuoyiyi/SNN/TCGA/stability_analysis_4'
    out_path = add_medians_to_summary(data_dir)
    print(f"✅ Updated: {out_path}")
