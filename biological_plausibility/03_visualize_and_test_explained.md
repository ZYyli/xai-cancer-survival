# `03_visualize_and_test.py` 说明文档

本文档基于脚本 `biological_plausibility/scripts/03_visualize_and_test.py` 的**当前实现**整理：

- **作图前如何处理/聚合数据**
- **每张图使用的数据口径（Fig1–Fig4）**
- **脚本输出了哪些文件（PNG/PDF/CSV）**
- **每个 CSV 的列名与含义**（按脚本实际写出的 DataFrame 为准）

## 1. 输入数据与依赖

### 1.1 输入文件

- **输入 CSV**：`OUTPUT_DIR/all_model_statistics.csv`
  - 由 `02_calculate_gene_scores.py` 生成。
  - 每行对应一个：`(cancer_type, xai_method, model_idx)` 的统计结果（以及若干数据库列）。

### 1.2 脚本依赖的关键列

- `cancer_type`：癌种。
- `xai_method`：XAI 方法（例如 `IG`, `shap`, `deepshap`, `LRP`, `DeepLIFT`, `PFI`）。
- `model_idx`：模型/重复实验索引（用于“癌种内配对检验”，尤其 Fig4 的 per-cancer Wilcoxon）。
- `total_hit_same_any`：脚本固定使用的核心指标（见下节）。

Fig3（数据库 × XAI）若存在以下列则会纳入（脚本内固定候选 4 个 DB，缺失则跳过）：

- `oncokb_hit_same_any`
- `dgidb_hit_same_any`
- `opentargets_hit_same_any`
- `cancermine_hit_same_any`

### 1.3 核心指标：`total_hit_same_any`

- 脚本固定：`metric = 'total_hit_same_any'`。
- 含义：某模型 top genes 在各数据库中“当前癌种支持”的命中数汇总；它代表“证据命中总量/次数”，不是 unique gene 数。

## 2. 全局映射与显示

### 2.1 XAI 类别映射（Fig2）

脚本内 `XAI_CATEGORIES` 把方法映射到 3 类：

- `Gradient-based`
- `Propagation-based`
- `Perturbation-based`

### 2.2 显示名称映射（Fig1/Fig3/Fig4 坐标轴显示）

`XAI_DISPLAY_NAMES` 用于统一展示名：

- `shap` -> `G-SHAP`
- `deepshap` -> `D-SHAP`

## 3. 作图前的数据聚合口径

脚本会把模型粒度 `df` 先聚合到癌种层面，得到“每癌种一个点”，再做跨癌种比较：

### 3.1 `df_cancer`（per-cancer median）

从模型粒度汇总到癌种粒度：

```text
df_cancer: grouped by (cancer_type, xai_method)
           value = median over models
```

对应代码：

`df.groupby(['cancer_type','xai_method'])[total_hit_same_any].median()`

### 3.2 `df_category_cancer`（per-cancer median + 类别内再 median）

步骤：

```text
1) df_cancer 添加 category = map(xai_method)
2) grouped by (cancer_type, category)
3) 对同类别内的多个方法再取 median
```

这样每个癌种在每个类别下只贡献一个值，避免类别内方法数不同导致偏倚。

## 4. Fig1–Fig4：数据处理与输出

输出目录统一为：`viz_dir = OUTPUT_DIR/visualizations`

### Fig1：XAI 方法箱线图

- **绘图输入**：`df_cancer`
- **点的含义**：每个散点 = “某癌种 + 某 XAI” 的 `total_hit_same_any`（癌种内模型 median）。
- **排序**：XAI 方法按跨癌种的 median 从小到大排序。
- **显著性标注**：做“癌种配对”的 Wilcoxon + FDR，并在图上以 `champion vs others` 星号标注。

图像输出：

- `fig1_boxplot_xai_total_hit_same_any.png/.pdf`

补充 CSV 输出：

1) `supplementary_fig1_plot_data_method_per_cancer_median.csv`

- **列名**：`Cancer`, `Method`, `total_hit_same_any`
- **含义**：每癌种×方法一个值（癌种内模型 median）。

2) `supplementary_fig1_summary_across_cancers_by_method.csv`

- **列名**：`Method`, `n_cancers`, `mean`, `std`, `min`, `q25`, `median`, `q75`, `max`, `IQR`
- **含义**：对 Fig1 的 per-cancer 点，按方法做描述性统计。

3) `statistical_tests_XAI_methods.csv`

这是一个“带两段 CSV 的文本文件”，格式为：

- **Overall 段**（Friedman, 以 cancer_type 为配对 block）
  - **列名**：`Test`, `Statistic`, `p_value`, `Effect_Size_Kendalls_W`, `n_cancers`, `Significant`
- **Pairwise 段**（Wilcoxon signed-rank + FDR, cancer_type 配对）
  - **列名**：`Group_1`, `Group_2`, `Statistic`, `p_value`, `n_blocks`, `Median_1`, `Median_2`, `Median_Diff`, `p_fdr`, `sig_fdr`

### Fig2：XAI 类别箱线图

- **绘图输入**：`df_category_cancer`
- **点的含义**：每个散点 = “某癌种 + 某类别” 的 `total_hit_same_any`（类别内方法 median 后的值）。
- **排序**：类别按跨癌种 median 从小到大排序。
- **显著性标注**：同样做配对检验（以 cancer_type 为 block）并标注显著性。

图像输出：

- `fig2_boxplot_category_total_hit_same_any.png/.pdf`

补充 CSV 输出：

1) `supplementary_fig2_plot_data_category_per_cancer_median.csv`

- **列名**：`Cancer`, `Category`, `total_hit_same_any`

2) `supplementary_fig2_summary_across_cancers_by_category.csv`

- **列名**：`Category`, `cancer_number`, `mean`, `std`, `min`, `q25`, `median`, `q75`, `max`, `IQR`

3) `statistical_tests_XAI_categories.csv`

- **Overall 段列名**：`Test`, `Statistic`, `p_value`, `Effect_Size_Kendalls_W`, `n_cancers`, `Significant`
- **Pairwise 段列名**：`Group_1`, `Group_2`, `Statistic`, `p_value`, `n_blocks`, `Median_1`, `Median_2`, `Median_Diff`, `p_fdr`, `sig_fdr`

### Fig3：数据库 × XAI 热力图

- **绘图输入**：模型粒度 `df`。
- **热图矩阵**：
  - 行：XAI（展示名）
  - 列：数据库（OncoKB/DGIdb/OpenTargets/CancerMine，存在才纳入）
  - 值：脚本默认 `aggregation='s2'`，即：

```text
1) 先对每个 (cancer_type, xai_method) 在该数据库列上取 median
2) 再对这些 per-cancer medians 在癌种维度取 median
```

图像输出：

- `fig3_heatmap_database_xai_total_hit_same_any.png/.pdf`

数据与统计输出：

1) `database_xai_hit_heatmap_data.csv`

- **列名**：4 个数据库列名（`OncoKB`, `DGIdb`, `OpenTargets`, `CancerMine`）
- **index**：XAI 展示名（`IG`, `G-SHAP`, `D-SHAP`, ...）
- **值**：每个单元格是该 XAI 在该 DB 列上的“median of per-cancer medians”。

2) `within_database_method_tests_overall.csv`

- **列名**：`Database`, `Test`, `Statistic`, `p_value`, `Effect_Size_Kendalls_W`, `n_blocks`, `k_methods`
- **含义**：在每个数据库列内，比较 6 种 XAI（block=cancer_type）的 Friedman 结果。

3) `within_database_method_tests_pairwise.csv`

- **列名**：`Group_1`, `Group_2`, `Statistic`, `p_value`, `n_blocks`, `p_fdr`, `sig_fdr`, `Database`
- **含义**：数据库内的两两 Wilcoxon 配对比较（FDR 校正）。

### Fig4：癌种 × XAI 热力图

Fig4 的绘图数据先写成 pivot 文件：

- `cancer_xai_hit_heatmap_data.csv`
  - **index**：`cancer_type`
  - **columns**：`xai_method`（内部名）
  - **值**：`df_cancer` 的 per-cancer median

绘图函数 `plot_heatmap()` 会：

- 按 `CANCER_TYPES` 过滤/排序癌种
- 按 `XAI_METHODS` 过滤方法，并把列名映射成展示名
- 转置为 `XAI × Cancer` 的矩阵绘制热图
- 额外：对每个癌种，在该癌种内做 `model_idx` 配对的 Wilcoxon（champion vs others），把显著性星号写到每个单元格的注释里

图像输出：

- `fig4_heatmap_cancer_xai_total_hit_same_any.png/.pdf`

额外统计输出：

1) `within_cancer_method_tests_overall.csv`

- **列名**：`Cancer`, `Test`, `Statistic`, `p_value`, `Effect_Size_Kendalls_W`, `n_blocks`, `k_methods`
- **含义**：在每个癌种内（block=model_idx），比较 6 种 XAI 的 Friedman 结果。

2) `within_cancer_method_tests_pairwise.csv`

- **列名**：`Group_1`, `Group_2`, `Statistic`, `p_value`, `n_blocks`, `p_fdr`, `sig_fdr`, `Cancer`
- **含义**：每个癌种内的两两 Wilcoxon 配对比较（FDR 校正）。

3) `supplementary_fig4_paired_wide_data_used_for_tests.csv`

- **列名**：`replicate_id` + 6 个方法列（展示名，顺序为 `method_display_order`）
- **含义**：脚本在每个癌种内构造的配对 wide 矩阵（index=model_idx），为了落盘将 `model_idx` 重命名为 `replicate_id` 并 `reset_index()`；随后把 15 个癌种的 wide 矩阵直接纵向拼接保存（该文件**不包含 Cancer 列**）。

## 5. 输出文件清单（汇总）

### 图像（PNG+PDF）

- `fig1_boxplot_xai_total_hit_same_any.png/.pdf`
- `fig2_boxplot_category_total_hit_same_any.png/.pdf`
- `fig3_heatmap_database_xai_total_hit_same_any.png/.pdf`
- `fig4_heatmap_cancer_xai_total_hit_same_any.png/.pdf`

### 关键 CSV / 文本输出

- `supplementary_fig1_plot_data_method_per_cancer_median.csv`
- `supplementary_fig1_summary_across_cancers_by_method.csv`
- `supplementary_fig2_plot_data_category_per_cancer_median.csv`
- `supplementary_fig2_summary_across_cancers_by_category.csv`
- `cancer_xai_hit_heatmap_data.csv`
- `database_xai_hit_heatmap_data.csv`
- `statistical_tests_XAI_methods.csv`（两段 CSV 文本：overall + pairwise）
- `statistical_tests_XAI_categories.csv`（两段 CSV 文本：overall + pairwise）
- `within_database_method_tests_overall.csv`
- `within_database_method_tests_pairwise.csv`
- `within_cancer_method_tests_overall.csv`
- `within_cancer_method_tests_pairwise.csv`
- `supplementary_fig4_paired_wide_data_used_for_tests.csv`
