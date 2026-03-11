import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr

# ====== 参数 ======
input_file = "/home/zuoyiyi/SNN/TCGA/operation/cindex_factors.csv"
out_dir = "/home/zuoyiyi/SNN/TCGA/correlation"
os.makedirs(out_dir, exist_ok=True)
corr_method = "pearson"  # "pearson" 或 "spearman"

# 读取数据
df = pd.read_csv(input_file, sep=',')
print("列名:", df.columns.tolist())

# 提取 C_index 主值
df["C_index"] = df["C_index"].str.split("±").str[0].astype(float)
df["Num_factors"] = pd.to_numeric(df["Num_factors"], errors="coerce")
df = df.dropna(subset=["C_index", "Num_factors"])

print("清洗后行数:", len(df))
print("各方法数据量:\n", df["Method"].value_counts())

# 确保列存在
required_cols = {"Cancer", "Method", "C_index", "Num_factors"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"输入文件必须包含列: {required_cols}")

# ====== 逐方法画散点图并计算相关性 ======
methods = df["Method"].unique()

for method in methods:
    df_m = df[df["Method"] == method]

    # 计算相关性
    if len(df_m) > 1:
        if corr_method == "pearson":
            corr, pval = pearsonr(df_m["Num_factors"], df_m["C_index"])
        else:
            corr, pval = spearmanr(df_m["Num_factors"], df_m["C_index"])
    else:
        corr, pval = float("nan"), float("nan")

    plt.figure(figsize=(7, 6))
    sns.regplot(
        x="Num_factors", y="C_index",
        data=df_m, scatter_kws={"s":70}, line_kws={"color":"red"}
    )
    plt.title(f"{method}\n{corr_method.capitalize()} r={corr:.2f}, p={pval:.3f}", fontsize=14)
    plt.xlabel("Number of Prognostic Factors", fontsize=12)
    plt.ylabel("C-index", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"cindex_vs_factors_{method}.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close()
    print(f"方法 {method}: 样本数={len(df_m)}, r={corr:.2f}, p={pval:.3f}")
    if os.path.exists(plot_path):
        print(f"图片保存成功: {plot_path}")
    else:
        print(f"图片保存失败: {plot_path}")

print("✅ 分析完成，已输出散点图。")