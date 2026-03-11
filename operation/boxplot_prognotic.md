# `boxplot_prognotic.py` 脚本说明

本文件说明脚本：`/home/zuoyiyi/SNN/TCGA/operation/boxplot_prognotic.py`。

内容包括：
- 数据来源是什么
- 对数据做了什么处理/聚合
- 四张图分别表示什么
- 使用了哪些统计检验、效应量与多重比较校正
- 脚本输出了哪些文件

---

## 1. 数据来源（输入数据是什么）

脚本从 6 个 XAI 方法对应的结果目录中读取每个癌症的 `*_detailed_results.csv`：

- `methods_dirs` 定义了 6 种方法及其目录：
  - `G-SHAP` -> `/home/zuoyiyi/SNN/TCGA/shap_results_2`
  - `IG` -> `/home/zuoyiyi/SNN/TCGA/IG_results_2`
  - `LRP` -> `/home/zuoyiyi/SNN/TCGA/LRP_results_2`
  - `PFI` -> `/home/zuoyiyi/SNN/TCGA/PFI_results_2`
  - `D-SHAP` -> `/home/zuoyiyi/SNN/TCGA/deepshap_results_2`
  - `DeepLIFT` -> `/home/zuoyiyi/SNN/TCGA/DeepLIFT_results_2`

- `cancer_list` 包含 15 个癌种：
  - `COADREAD`, `LUSC`, `HNSC`, `STAD`, `BLCA`, `BRCA`, `LUAD`, `PAAD`, `LIHC`, `SKCM`, `KIRC`, `UCEC`, `KIRP`, `GBMLGG`, `LGG`

- 每个癌种每个方法读取文件路径：
  - `.../<METHOD_DIR>/<CANCER>/<CANCER>_detailed_results.csv`

- 使用的核心指标列：
  - `col_name = 'cox_prognostic_factors_raw'`

该列代表每次抽样模型（重复实验/折叠）下，Cox 预后因子数量（raw）。

---

## 2. 数据组织形式与“配对单位”

### 2.1 原始数据点
每个 `*_detailed_results.csv` 通常包含以下列（不同方法前缀不同，但结构类似）：
- `repeat`：重复编号
- `fold`：折叠编号
- `cox_prognostic_factors_raw`：Cox 预后因子数量（raw）

脚本在读取时会保留 `repeat` 与 `fold`，并构造一个“配对单位”：
- `replicate_id = repeat + '_' + fold`

这个 `replicate_id` 用于在同一癌种内对齐不同 XAI 方法的“同一次抽样模型”，从而满足 **配对统计检验**（Friedman / Wilcoxon signed-rank）的前提。

### 2.2 避免伪重复（pseudo-replication）的聚合版本
在比较“跨癌种的总体方法差异”（图1）与“跨癌种的类别差异”（图2）时，脚本会先对原始数据进行聚合：

- 先构造原始长表：`df_original`，包含（至少）
  - `Cancer`, `Method`, `repeat`, `fold`, `N_Factors`

- 再按每个癌种、每个方法取中位数：
  - `df_cancer_median = df_original.groupby(['Cancer','Method'])['N_Factors'].median()`

这一步的含义是：
- 每个癌种每个方法最终只贡献 1 个代表值（中位数），避免把同一癌种下的多个重复/折叠当成独立样本，造成样本量“虚增”。

---

## 3. 四张图分别做了什么

输出目录：
- `output_dir = /home/zuoyiyi/SNN/TCGA/Prognostic_comparison_plots`

说明：脚本的 `_save_figure(fig, out_file_png)` 会同时保存：
- `*.png`
- 对应的 `*.pdf`

### 3.1 图1：6 种 XAI 方法（跨 15 癌种）箱线图

**图像输出**：
- `aggregated_XAI_prognostic_factors_boxplot.png` / `.pdf`

**目的**：比较 6 种 XAI 方法在 15 个癌种上的总体差异。

**使用的数据**：
- 原始长表：`df_original`（每行代表一个抽样模型：`Cancer + Method + repeat + fold` 对应一个 `N_Factors`）
- 用于图1的作图点：`df_cancer_median`
  - 构建方式：`df_original.groupby(['Cancer','Method']).median()`
  - 含义：每个癌种每个方法仅保留 1 个代表值（中位数）

**数据处理**：
- 每个 (Cancer, Method) 取 `N_Factors` 中位数，避免把同一癌种的多个重复/折叠当作独立样本（避免 pseudo-replication）。

**统计检验**（跨癌种配对）：
- 总体检验：Friedman test（配对）
  - block（配对单位）：`Cancer`
  - group（比较组）：`Method`
  - 效应量：Kendall’s W
- 两两比较：Wilcoxon signed-rank test（配对，block=癌种）
- 多重比较校正：FDR（BH）

**显著性标注**：
- 图上使用 `* / ** / ***` 标注：来自两两比较的 `p_fdr`。
- 仅针对“跨癌种中位数最大”的方法（champion）与其它方法的比较进行标注。

**输出文件（统计结果）**：
- `statistical_tests_XAI_methods.csv`
  - 作用：保存图1的总体 Friedman 结果 + 两两 Wilcoxon（含 FDR 校正）的结果。

**输出文件（补充材料：作图数据与汇总）**：
- `supplementary_fig1_plot_data_method_per_cancer_median.csv`
  - 作用：图1箱线图的“真实数据点”（每个癌种×每种方法一个中位数点）。
- `supplementary_fig1_summary_across_cancers_by_method.csv`
  - 作用：对上述作图点按 `Method` 做描述性统计（含 `n_cancers`、均值、中位数、IQR 等）。

---

### 3.2 图2：3 类 XAI 类别（跨 15 癌种）箱线图

**图像输出**：
- `XAI_category_prognostic_factors_boxplot.png` / `.pdf`

**目的**：将方法归并为 3 类后，比较 3 类在 15 个癌种上的总体差异。

**类别映射（以脚本为准）**：
- `Gradient-based`：`IG`, `G-SHAP`
- `Propagation-based`：`D-SHAP`, `LRP`, `DeepLIFT`
- `Perturbation-based`：`PFI`

**使用的数据**：
- `df_cancer_median`（每癌种×方法一个中位数）
- 构造 `df_category`：
  - 先给 `df_cancer_median` 增加 `Category`
  - 再按 `(Cancer, Category)` 取 `N_Factors` 中位数

**统计检验**（跨癌种配对）：
- 总体检验：Friedman test（配对，block=癌种，group=类别）
  - 效应量：Kendall’s W
- 两两比较：Wilcoxon signed-rank test（配对，block=癌种）
- 多重比较校正：FDR（BH）

**显著性标注**：
- 图上对所有显著配对绘制连线 `* / ** / ***`（来自 `p_fdr`）。

**输出文件（统计结果）**：
- `statistical_tests_XAI_categories.csv`
  - 作用：保存图2的总体 Friedman + 两两 Wilcoxon（含 FDR 校正）。

**输出文件（补充材料：作图数据与汇总）**：
- `supplementary_fig2_plot_data_category_per_cancer_median.csv`
  - 作用：图2箱线图的“真实数据点”（每个癌种×每个类别一个中位数点）。
- `supplementary_fig2_summary_across_cancers_by_category.csv`
  - 作用：对上述作图点按 `Category` 做描述性统计（均值/中位数/IQR 等）。

---

### 3.3 图3：15 个癌种 × 6 种 XAI 方法热图

**图像输出**：
- `prognostic_factors_heatmap.png` / `.pdf`

**目的**：展示每个癌种在 6 种方法下的 `N_Factors`（以每癌种×方法的中位数代表）。

**使用的数据**：
- `df_cancer_median`（每癌种×方法一个中位数）
- 热图矩阵 `mat`：
  - `df_cancer_median.pivot(index='Method', columns='Cancer', values='N_Factors')`
  - 并按固定顺序重排方法（`hue_order`）与癌种（`cancer_list`）

**额外标注（显著性符号的来源）**：
- 脚本在前面先对每个癌种单独做“方法差异检验”（基于 `df_original` 的模型级别数据），输出：
  - `within_cancer_method_tests_overall.csv`
  - `within_cancer_method_tests_pairwise.csv`
- 热图中的 `* / ** / ***` 标注来自：
  - 在每个癌种内，选出该癌种“中位数最大的方法”（champion）
  - 再从该癌种的两两 Wilcoxon（FDR 校正）中取 `p_fdr`，与 champion 对比得到显著性符号

说明：图3本身不额外产生新的统计检验文件，主要输出热图图像；其显著性标注依赖上述“癌种内检验”的结果。

**输出文件（癌种内检验，供热图标注与补充材料使用）**：
- `within_cancer_method_tests_overall.csv`
  - 作用：每个癌种一行的总体 Friedman（配对单位=repeat+fold）及 Kendall’s W。
- `within_cancer_method_tests_pairwise.csv`
  - 作用：每个癌种的两两 Wilcoxon（配对）+ FDR 校正，含 `p_fdr` 与是否显著。

### 3.4 图4：LGG 单癌种 6 种 XAI 方法箱线图（50 个抽样模型）

**图像输出**：
- `LGG_XAI_prognostic_factors_boxplot.png` / `.pdf`

**目的**：以 LGG 为代表癌种，展示 6 种方法在 50 个抽样模型上的 `N_Factors` 分布。

**使用的数据**：
- `lgg_data = df_original[df_original['Cancer'] == 'LGG']`
  - 每个点对应一个抽样模型（由 `repeat` 与 `fold` 标识）

**数据处理**：
- 构造 `replicate_id = repeat + '_' + fold` 作为配对单位
- 为配对统计检验构造 `lgg_wide`：
  - `pivot_table(index='replicate_id', columns='Method', values='N_Factors', aggfunc='median')`
  - 并 `dropna(how='any')`：仅保留 6 种方法都齐全的模型（保证配对检验前提）

**统计检验**（LGG 内模型级别配对）：
- 总体检验：Friedman test（配对，block=`replicate_id`）
  - 效应量：Kendall’s W
- 两两比较：Wilcoxon signed-rank test（配对，block=`replicate_id`）
- 多重比较校正：FDR（BH）

**显著性标注**：
- 仅对 LGG 内“中位数最大的方法”（champion）与其它方法的比较标注 `* / ** / ***`（来自 `p_fdr`）。

**输出文件（统计结果）**：
- `statistical_tests_LGG_XAI_methods.csv`
  - 作用：保存图4的总体 Friedman + 两两 Wilcoxon（含 FDR 校正）。

**输出文件（补充材料：作图数据与汇总）**：
- `supplementary_fig4_plot_data_LGG_models.csv`
  - 作用：图4箱线图的“真实数据点”（LGG 下每个模型×方法的 `N_Factors`）。
- `supplementary_fig4_summary_across_models_by_method.csv`
  - 作用：对上述作图点按 `Method` 做描述性统计（样本量/均值/中位数/IQR 等）。
- `supplementary_fig4_paired_wide_data_used_for_tests.csv`
  - 作用：保存配对统计检验实际使用的矩阵 `lgg_wide`（严格配对后的输入数据），便于复现统计检验。

---

## 4. 类别内部方法差异检验（额外分析输出）

脚本还会在每个类别内部做“方法差异检验”，输出：

- `within_category_method_tests_overall.csv`
  - 类别内若方法数>=3：Friedman（配对，block=癌种）并输出 Kendall’s W
  - 类别内若方法数=2：Wilcoxon（配对，block=癌种）并输出 rank-biserial 效应量
- `within_category_method_tests_pairwise.csv`
  - 类别内所有方法两两 Wilcoxon（配对，block=癌种）
  - 输出 `Median_Diff`（方向与大小）
  - 输出 `Effect_Size_RankBiserial` 与 `n_nonzero`

### 4.1 Wilcoxon 的效应量（rank-biserial）
- `Effect_Size_RankBiserial` 范围：`[-1, 1]`
- 绝对值越大：差异越强
- 正负号代表方向（基于 `Method_1 - Method_2`）：
  - `> 0` 倾向于 `Method_1 > Method_2`
  - `< 0` 倾向于 `Method_1 < Method_2`

---

## 5. 颜色与散点叠加（可视化细节）

### 5.1 颜色
- 类别颜色：`CATEGORY_COLORS`
- 方法颜色：`XAI_COLORS`，并用 `METHOD_COLORS` 将脚本内部方法名（如 `G-SHAP/D-SHAP`）映射到对应颜色。

### 5.2 图2/图3散点
散点叠加方式与 `03_visualize_and_test.py` 保持一致：
- jitter：`x = np.random.normal(i, 0.04, size=len(y))`
- `ax.scatter(..., alpha=0.8, s=20, color=..., zorder=3)`

---

## 6. 输出文件清单

图像：
- `aggregated_XAI_prognostic_factors_boxplot.png` / `.pdf`（图1）
- `XAI_category_prognostic_factors_boxplot.png` / `.pdf`（图2）
- `prognostic_factors_heatmap.png` / `.pdf`（图3）
- `LGG_XAI_prognostic_factors_boxplot.png` / `.pdf`（图4）

统计结果（主比较）：
- `statistical_tests_XAI_methods.csv`
- `statistical_tests_XAI_categories.csv`
- `statistical_tests_LGG_XAI_methods.csv`

补充材料（作图数据与汇总）：
- `supplementary_fig1_plot_data_method_per_cancer_median.csv`
- `supplementary_fig1_summary_across_cancers_by_method.csv`
- `supplementary_fig2_plot_data_category_per_cancer_median.csv`
- `supplementary_fig2_summary_across_cancers_by_category.csv`
- `supplementary_fig4_plot_data_LGG_models.csv`
- `supplementary_fig4_summary_across_models_by_method.csv`
- `supplementary_fig4_paired_wide_data_used_for_tests.csv`

统计结果（补充分析）：
- `within_category_method_tests_overall.csv`
- `within_category_method_tests_pairwise.csv`
- `within_cancer_method_tests_overall.csv`
- `within_cancer_method_tests_pairwise.csv`
