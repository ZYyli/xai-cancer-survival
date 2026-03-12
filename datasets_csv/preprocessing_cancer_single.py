import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

def add_bins(slide_data, label_col, n_bins, eps):
    """
    对生存数据进行分箱处理，并为每个患者分配一个复合标签。
    """
    assert 'case_id' in slide_data.columns and 'censorship' in slide_data.columns

    if slide_data[label_col].isna().any():
        raise ValueError(f"{label_col} 列中存在 NaN 值，请先处理")

    patients_df = slide_data.drop_duplicates(['case_id']).copy()
    uncensored_df = patients_df[patients_df['censorship'] < 1]
    
    # 对未删失患者的生存时间进行分位数分箱
    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False, duplicates='drop')
    
    # 调整分箱边界，确保所有数据点都被包含
    q_bins[-1] = slide_data[label_col].max() + eps
    q_bins[0] = slide_data[label_col].min() - eps

    # 使用调整后的边界对所有患者进行分箱
    disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    
    patients_df.insert(2, 'label', disc_labels.values.astype(int))
    
    return q_bins, patients_df

def series_intersection(s1, s2):
    """
    找出两个pandas Series的交集，并返回为Series。
    """
    return pd.Series(list(set(s1) & set(s2)))

def plot_cum_mad(cum_mad, cancer, top_k, mad_dir):
    """
    绘制累积MAD曲线并保存。
    """
    plt.plot(np.arange(1, len(cum_mad)+1), cum_mad)
    plt.xlabel('Number of Genes')
    plt.ylabel('Cumulative MAD Proportion')
    plt.axvline(top_k, color='red', linestyle='--')
    plt.title(f'{cancer} MAD cumulative plot')
    plt.savefig(f'{mad_dir}/{cancer}_cum_mad.png', format='png')
    plt.close()

def process_cancer_single(cancer, data_root, mad_dir, output_dir, label_col, top_k, eps, n_bins):
    """
    针对单一癌症，进行数据预处理和MAD特征选择，并保存单个处理好的文件。
    """
    cancer_file = f'tcga_{cancer}_all_clean.csv.zip'
    file_path = os.path.join(data_root, cancer_file)

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"正在处理癌症: {cancer}")
    slide_data = pd.read_csv(file_path, header=0, index_col=0, sep=',', low_memory=False)
    print("原始数据结构：", slide_data.shape)
    print("原始数据索引：", slide_data.index)

    if 'case_id' not in slide_data:
        slide_data.index = slide_data.index.str[:12]
        slide_data['case_id'] = slide_data.index
        slide_data = slide_data.reset_index(drop=True)
    
    # 分箱处理
    q_bins, slide_data = add_bins(slide_data, label_col, n_bins, eps)
    slide_data = slide_data.reset_index(drop=True)
    
    # 构建生存时间分箱和删失状态字典
    label_dict = {}
    key_count = 0
    for i in range(len(q_bins) - 1):
        for c in [0, 1]:
            label_dict.update({(i, c): key_count})
            key_count += 1

    #更新 slide_data 的标签列.
    #最后disc_label仅存储生存时间分箱标签，不包含删失信息。
    #最后label存储复合标签，结合了分箱+删失信息。    
    for i in slide_data.index:
        key = slide_data.loc[i, 'label']
        slide_data.at[i, 'disc_label'] = key
        censorship = slide_data.loc[i, 'censorship']
        key = (key, int(censorship))
        slide_data.at[i, 'label'] = label_dict[key]

        #(0, 0): 0,  # 分箱0 + 事件发生
        #(0, 1): 1,  # 分箱0 + 删失
        #(1, 0): 2,  # 分箱1 + 事件发生
        #(1, 1): 3,  # 分箱1 + 删失
        #(2, 0): 4,  # 分箱2 + 事件发生
        #(2, 1): 5,  # 分箱2 + 删失
        #(3, 0): 6,  # 分箱3 + 事件发生
        #(3, 1): 7   # 分箱3 + 删失


    bins = q_bins
    num_classes = len(label_dict)
    print("生存时间分箱边界：", bins)
    print("结合了分箱+删失复合标签类别数：", num_classes)
    print("分箱后数据结构：", slide_data.shape)

    #利用分子特征数据库（Molecular Signatures Database, MSigDB）的基因家族类别基因集来限制特征数量.
    signatures = pd.read_csv(os.path.join(data_root, 'signatures.csv'))
    #仅提取RNAseq相关的基因名
    rna_columns = [col for col in slide_data.columns if col.endswith('_rnaseq')]
    #提取RNAseq特征列
    genomic_rna = slide_data[rna_columns]
    print(f"rna基因数: {genomic_rna.shape[1]}")
    
    omic_from_signatures = np.unique(np.concatenate([signatures[col].dropna().unique() for col in signatures.columns]))
    rnaseq_overlap = sorted(series_intersection(np.concatenate([omic_from_signatures + mode for mode in ['_rnaseq']]), genomic_rna.columns))
    
    if len(rnaseq_overlap) == 0:
        print(f"⚠️{cancer}: 没有交集基因，跳过！")
        return

    # 对全数据集做MAD筛选并排序
    genomic_rna_select = genomic_rna[rnaseq_overlap]
    mad = stats.median_abs_deviation(genomic_rna_select, axis=0)
    cum_mad = np.cumsum(np.sort(mad)[::-1]) / np.sum(mad)
    plot_cum_mad(cum_mad, cancer, top_k, mad_dir)

    auto_top_k = np.where(cum_mad >= 0.8)[0][0] + 1
    print(f"自动选择基因数: {auto_top_k}")

    print(f"指定基因数: {top_k}")
    top_indices = np.argpartition(mad, -top_k)[-top_k:]
    # 对 top_indices 内部按 MAD 排序（降序）
    top_indices = top_indices[np.argsort(mad[top_indices])[::-1]]
    high_var_genes = genomic_rna_select.columns[top_indices]

    slide_data_metadata = [col for col in slide_data.columns if not col.endswith('_rnaseq') and not col.endswith('_cnv') and not col.endswith('_mut')]
    # 保留高变异特征
    rna_data = pd.concat([slide_data[slide_data_metadata], genomic_rna_select[high_var_genes]], axis=1)
    print(f"筛选后slide_data结构: {rna_data.shape}")
    print(f"筛选后slide_data索引: {rna_data.index[:14]}")
    print(f"筛选后slide_data列名示例: {rna_data.columns[:14]}")

    # 填充缺失值
    if 'age' in rna_data.columns:
        median_age = rna_data['age'].median()
        rna_data['age'] = rna_data['age'].fillna(median_age)
    
    if rna_data.isnull().values.any():
        print("警告: 数据中仍存在缺失值，请检查。")
    
    # 保存为单个文件
    cancer_upper = cancer.upper()
    output_file = f'{output_dir}/{cancer_upper}_preprocessed.csv'
    rna_data.to_csv(output_file, index=False)
    print(f'{cancer_upper}预处理数据保存完成至：{output_file}\n')

if __name__ == "__main__":
    # 参数配置
    cancer_types = ['BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC']
    data_root = '/home/zuoyiyi/SNN/TCGA/datasets_csv'
    label_col = 'survival_months'
    top_k = 2000
    eps = 1e-6
    n_bins = 4
    mad_dir = 'mad_cancer_single'
    output_dir = 'preprocess_cancer_single'
    os.makedirs(mad_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for cancer in cancer_types:
        cancer_lower = cancer.lower()
        process_cancer_single(cancer_lower, data_root, mad_dir, output_dir, label_col, top_k, eps, n_bins)
    
    print(f"\n所有癌症预处理完成!")