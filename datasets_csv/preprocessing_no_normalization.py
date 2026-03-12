import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from split import generate_survival_five_fold_split
import matplotlib.pyplot as plt
from scipy import stats

def add_bins(slide_data, label_col):
    #用于检查输入的数据框 slide_data 是否包含 'case_id'（患者ID）和 'censorship'（删失信息）这两列
    assert 'case_id' in slide_data.columns and 'censorship' in slide_data.columns

    if slide_data[label_col].isna().any():
        raise ValueError(f"{label_col} 列中存在 NaN 值，请先处理")

    #patients_df 是一个每个患者只有一行记录的数据框。
    patients_df = slide_data.drop_duplicates(['case_id']).copy()
    uncensored_df = patients_df[patients_df['censorship'] < 1]
    # pandas.qcut对未删失患者的生存时间进行分位数分箱。分箱边界(q_bins)。每个数据点所属箱子的整数索引(disc_labels)。
    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
    #最后一个分箱的上边界调整为整个数据集中 label_col 的最大值加上 eps，以确保最大的数据点也能被正确分箱。
    q_bins[-1] = slide_data[label_col].max() + eps
    #第一个分箱的下边界
    q_bins[0] = slide_data[label_col].min() - eps

    #使用调整后的分箱边界 q_bins，通过 pandas.cut 函数对 所有 患者 (patients_df) 的 label_col 进行分箱。
    disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    ##disc_labels 现在包含了所有患者（无论是否删失）的分箱整数标签
    #第3列（索引为2）插入一个名为 'label' 的新列,值为disc_labels（分箱标签），并将其数据类型转换为整数。
    patients_df.insert(2, 'label', disc_labels.values.astype(int))
        
    #函数最后返回两个结果：
    #q_bins最终用于分箱的边界值；
    #patients_df为处理后的患者数据框，其中每个患者一行，并增加了'label'列(表示分箱后的类别)。
    return q_bins, patients_df

#取基因特征和signatures的交集
def series_intersection(s1, s2):
    return pd.Series(list(set(s1) & set(s2)))

def plot_cum_mad(cum_mad, cancer, top_k, mad_dir, i):
    plt.plot(np.arange(1, len(cum_mad)+1), cum_mad)
    plt.xlabel('Number of Genes')
    plt.ylabel('Cumulative MAD Proportion')
    plt.axvline(top_k, color='red', linestyle='--')
    plt.title(f'{cancer} MAD cumulative plot')
    plt.savefig(f'{mad_dir}/{cancer}_cum_mad_train_{i}.png', format='png')
    plt.close()

# 主处理逻辑
def process_cancer(cancer, data_root, split_root, mad_dir, output_dir, label_col, top_k, eps, n_bins):
    cancer_file= f'tcga_{cancer}_all_clean.csv.zip'
    file_path = os.path.join(data_root, cancer_file)

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"正在处理癌症: {cancer}")
    slide_data = pd.read_csv(file_path, header=0, index_col=0, sep=',', low_memory=False)
    print("原始数据结构：\n",slide_data.shape)
    print("原始数据索引：\n",slide_data.index)

    if 'case_id' not in slide_data:
        slide_data.index = slide_data.index.str[:12]
        slide_data['case_id'] = slide_data.index
        slide_data = slide_data.reset_index(drop=True)

    #print("修改前 censorship 列的值分布：\n", slide_data['censorship'].value_counts())
    #修改censorship的0/1值，使其调换
    #slide_data['censorship'] = slide_data['censorship'].map({0: 1, 1: 0})
    #print("修改后 censorship 列的值分布：\n", slide_data['censorship'].value_counts())
    
    q_bins, slide_data = add_bins(slide_data, label_col)
    slide_data = slide_data.reset_index(drop=True)
    print("分箱后数据索引：",slide_data.index)
    print("分箱后数据行名：",slide_data.columns)

    #构建生存时间分箱和删失状态字典
    label_dict = {}
    key_count = 0
    for i in range(len(q_bins)-1):
        for c in [0, 1]:
            label_dict.update({(i, c):key_count})
            key_count+=1

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
    num_classes=len(label_dict)

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

    #从签名文件中提取所有基因名
    omic_from_signatures = []
    for col in signatures.columns:
        #提取每列的非空唯一值
        omic = signatures[col].dropna().unique()
        omic_from_signatures.append(omic)
    #合并所有签名中的唯一基因名，用于后续特征筛选
    omic_from_signatures = np.unique(np.concatenate(omic_from_signatures))

    rnaseq_overlap = np.concatenate([omic_from_signatures+mode for mode in ['_rnaseq']])
    #筛选交集基因
    rnaseq_overlap = sorted(series_intersection(rnaseq_overlap, genomic_rna))
    if len(rnaseq_overlap) == 0:
        print(f"⚠️{cancer}: 没有交集基因，跳过！")
        return

    # 对全数据集做MAD筛选并排序
    genomic_rna_select = genomic_rna[rnaseq_overlap]
    mad = stats.median_abs_deviation(genomic_rna_select, axis=0)
    cum_mad = np.cumsum(np.sort(mad)[::-1]) / np.sum(mad)
    # 绘制解释方差曲线
    plot_cum_mad(cum_mad, cancer, top_k, mad_dir, i = 0)
    ##拐点：若曲线在 2000 附近趋于平缓，说明增加基因数对信息量提升有限

    #自动选择解释80%方差的基因数
    aoto_top_k = np.where(cum_mad >= 0.8)[0][0] + 1  # 达到80%解释力的最小基因数
    print(f"自动选择基因数: {aoto_top_k}")

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

    # 数据分割
    split_file_path = os.path.join(split_root, cancer)
    if not os.path.exists(split_file_path) or len(os.listdir(split_file_path)) < 5:
        os.makedirs(split_file_path, exist_ok=True)
        generate_survival_five_fold_split(rna_data, split_file_path)
    else:
        print(f"{cancer} 的分割文件已存在，跳过生成。")

    for i in range(5):
        split_path = os.path.join(split_file_path, f'splits_{i}.csv')
        print(f"\n正在读取分割文件: {split_path}")
        case_ids_df = pd.read_csv(split_path)
        
        ##训练集和验证集数据准备,读取训练集和验证集的 case_id
        train_case_ids = case_ids_df[case_ids_df['set'] == 'train']['case_id'].dropna().tolist()
        val_case_ids = case_ids_df[case_ids_df['set'] == 'val']['case_id'].dropna().tolist()

        train_data = rna_data[rna_data['case_id'].isin(train_case_ids)]
        val_data = rna_data[rna_data['case_id'].isin(val_case_ids)]

        median_age = train_data['age'].median()

        train_missing = train_data.isnull().values.any()
        print(f"是否存在空白值: {train_missing}")
        if train_missing:
            train_data['age'] = train_data['age'].fillna(median_age)
            print(train_data.isnull().sum())
            
        val_missing = val_data.isnull().values.any()
        print(f"是否存在空白值: {val_missing}")
        if val_missing:
            val_data['age'] = val_data['age'].fillna(median_age)
            print(val_data.isnull().sum())

        print(f"合并后train数据索引示例: {train_data.index[:14]}")
        print(f"合并后train数据列名示例: {train_data.columns[:14]}")
        print(f"合并后train数据结构: {train_data.shape}")
        print(f"合并后val数据索引示例: {val_data.index[:14]}")
        print(f"合并后val数据列名示例: {val_data.columns[:14]}")
        print(f"合并后val数据结构: {val_data.shape}")

        train_data.to_csv(f'{output_dir}/{cancer}_{i}_train.csv', index=False)
        val_data.to_csv(f'{output_dir}/{cancer}_{i}_val.csv', index=False)

    print(f'{cancer}预处理数据保存完成。\n')

if __name__ == "__main__":
    # 参数配置
    cancer_types = ['blca', 'brca', 'coadread', 'gbmlgg', 'hnsc', 'kirc', 'kirp', 'lgg', 'lihc', 'luad', 'lusc', 'paad', 'skcm', 'stad', 'ucec']
    data_root = '/home/zuoyiyi/SNN/TCGA/datasets_csv'
    split_root = '/home/zuoyiyi/SNN/TCGA/splits_1'
    label_col = 'survival_months'
    top_k = 2000
    eps = 1e-6
    n_bins = 4
    mad_dir = 'mad_1'
    output_dir = 'preprocess_1'
    os.makedirs(mad_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for cancer in cancer_types:
        process_cancer(cancer, data_root, split_root, mad_dir, output_dir, label_col, top_k, eps, n_bins)
    print(f"\n15种癌症预处理完成!")
