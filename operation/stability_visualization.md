# stability_visualization.py 说明文档

本文档解释 `operation/stability_visualization.py` 的：

- 数据来源（输入文件、目录结构、字段含义）
- 每张图（Fig1–Fig5）使用的数据与处理流程
- 使用的统计检验方法、配对方式、效应量与多重比较校正
- 输出文件（图像与统计结果 CSV）

---

## 1. 数据来源与目录结构

脚本默认使用 `--data_dir` 指定的稳定性分析目录（默认值：`/home/zuoyiyi/SNN/TCGA/stability_analysis_4`）。

### 1.1 汇总表（作图主数据）

- **文件**：`{data_dir}/cross_cancer_stability_summary.csv`
- **粒度**：每行对应一个 `(cancer_type, xai_method, top_k)` 的稳定性汇总
- **典型字段**：
  - `cancer_type`：癌症类型（如 BLCA、BRCA 等）
  - `xai_method`：XAI 方法原始名称（如 `shap`, `deepshap`, `IG`, `LRP`, `DeepLIFT`, `PFI`）
  - `top_k`：特征选择的 k
  - `n_rankings`：用于稳定性计算的 ranking 数量
  - `kuncheva_mean`, `kuncheva_std`
  - `jaccard_mean`, `jaccard_std`
  - `rbo_09_mean`, `rbo_09_std`
  - **新增的中位数列**（当前分析/作图统一口径）：
    - `kuncheva_median`
    - `jaccard_median`
    - `rbo_09_median`

脚本读入后会增加两个派生列：

- `xai_category`：将方法映射到类别（`XAI_CATEGORIES`）
- `xai_display`：将方法映射到显示名称（`XAI_DISPLAY_NAMES`）

### 1.2 明细表（用于 Fig3 的配对检验）

用于“每个癌症内部 6 种 XAI 方法”的配对检验，脚本会读取：

- **文件模式**：`{data_dir}/{cancer_type}/{xai_method}/pairwise_stability_raw_top{top_k}.csv`
- **典型字段**：
  - `cancer_type`
  - `pair_id`
  - `repeat1`, `fold1`, `repeat2`, `fold2`（用于构造更稳健的配对键）
  - `kuncheva`, `jaccard`, `rbo_09`（明细指标）

脚本会优先使用组合键：

- `pair_key = repeat1_fold1_repeat2_fold2`

若缺少 repeat/fold 信息则退回使用 `pair_id` 作为配对块（block）。

---

## 2. 指标列选择（mean / median）

为了统一口径，Fig1–Fig3 默认使用 `*_median` 列作图；Fig4/5 也支持中位数口径。

脚本提供 `_resolve_metric_column(df, metric)`：

- 若 `metric` 列存在，直接使用
- 若 `metric` 以 `_median` 结尾但列不存在，则回退到对应的 `*_mean` 列，并打印提示

---

## 3. 统计检验方法总览

脚本主要使用三类统计流程：

### 3.1 Friedman 检验（重复测量/配对）

- **目的**：检验多组（>2 组）在配对设计下是否整体存在差异
- **实现**：`scipy.stats.friedmanchisquare`
- **配对块（block）**：
  - Fig1/Fig2：`cancer_type`
  - Fig3：`pair_key`（或退回 `pair_id`）
- **效应量**：Kendall’s W

`W = statistic / (n_blocks * (k_groups - 1))`

### 3.2 配对 Wilcoxon signed-rank（两两比较）

- **目的**：在配对设计下对两组进行两两比较
- **实现**：`scipy.stats.wilcoxon`（封装为 `_wilcoxon_safe`）
- **效应量**：rank-biserial（封装为 `_rank_biserial_paired`）

### 3.3 多重比较校正（FDR）

- **方法**：Benjamini–Hochberg
- **实现**：`statsmodels.stats.multitest.multipletests(method='fdr_bh')`
- **输出字段**：`p_fdr`、`sig_fdr`

### 3.4 显著性标注

- Fig1：CLD（compact letter display），用字母（a/b/ab…）表示显著性分组
- Fig2：括号线 + 星号（由 `add_significance_annotations_from_results` 添加）
- Fig3：热图单元格显示“数值 + 换行 + CLD字母”（每个癌症列内单独计算字母）

---

## 4. Fig1–Fig5 逐图说明

### Fig1：6 种 XAI 方法稳定性箱线图

- **函数**：`plot_xai_method_boxplot(top_k=..., metric='kuncheva_median')`
- **作图数据**：来自 summary
  - 筛选 `top_k == top_k`
  - y = `metric`（默认 `kuncheva_median`）
  - x = `xai_display`
  - 每个方法一组点（约 15 个癌症）
- **排序**：按每个方法的中位数从低到高排序
- **统计检验**：
  - Friedman（block=`cancer_type`，group=`xai_display`）
  - 两两配对 Wilcoxon + FDR
  - CLD 字母标注（`a` 对应该图中位数最高的方法）
- **输出**：
  - `fig1_xai_method_boxplot_top{top_k}.png/.pdf`
  - `statistical_tests_xai_methods_top{top_k}.csv`

### Fig2：3 类 XAI 类别稳定性箱线图

- **函数**：`plot_xai_category_boxplot(top_k=..., metric='kuncheva_median')`
- **作图数据**：来自 summary，但会避免伪重复
  - 先筛选 `top_k == top_k`
  - 再聚合：对每个 `(cancer_type, xai_category)` 取 `metric` 的 median
    - 原因：每个类别包含多个方法，不聚合会导致同一癌症在同一类别出现多次（伪重复）
- **统计检验**：
  - Friedman（block=`cancer_type`，group=`xai_category`）
  - 两两配对 Wilcoxon + FDR
  - 括号线 + 星号标注
- **输出**：
  - `fig2_xai_category_boxplot_top{top_k}.png/.pdf`
  - `statistical_tests_xai_categories_top{top_k}.csv`

### Fig3：癌症 × XAI 稳定性热图（15×6）

- **函数**：`plot_stability_heatmap(top_k=..., metric='kuncheva_median')`
- **热图数值（作图）**：来自 summary
  - 筛选 `top_k == top_k`
  - pivot：行=`xai_display`，列=`cancer_type`，值=`metric`

#### Fig3 的显著性检验（每癌症列内）

- **检验数据**：来自 raw 明细表 `pairwise_stability_raw_top{top_k}.csv`
- **配对块（block）**：
  - 优先 `pair_key = repeat1_fold1_repeat2_fold2`
  - 否则使用 `pair_id`
- **每个癌症单独做**：
  - Friedman（paired by block）
  - 两两配对 Wilcoxon + FDR（在该癌症内部校正 15 个比较）
- **CLD 字母**：按 raw 数据计算列中位数作为排序依据，生成每癌症的 CLD 显著性字母
- **注释写回热图**：每格显示：`value`（summary）+ 换行 + `letter`（CLD）

- **输出**：
  - `fig3_stability_heatmap_top{top_k}.png/.pdf`
  - `within_cancer_method_tests_overall_top{top_k}_{metric}.csv`
  - `within_cancer_method_tests_pairwise_top{top_k}_{metric}.csv`

### Fig4：Kuncheva vs Top-k

- **函数**：`plot_kuncheva_vs_topk(metric='kuncheva_median')`
- **作图数据**：来自 summary
- **处理**：对每个 `(xai_display, top_k)` 在癌症维度上再聚合
  - 若 `metric` 为 `*_median`：画 `median` + IQR（q25–q75）
  - 若 `metric` 为 `*_mean`：画 `mean` ± SE
- **输出**：`fig4_kuncheva_vs_topk.png/.pdf`

### Fig5：RBO vs Top-k

- **函数**：`plot_rbo_vs_topk(metric='rbo_09_median')`
- **作图数据**：来自 summary
- **处理**：对每个 `(xai_display, top_k)` 在癌症维度上再聚合
  - 若 `metric` 为 `*_median`：画 `median` + IQR（q25–q75）
  - 若 `metric` 为 `*_mean`：画 `mean` ± SE
- **输出**：`fig5_rbo_vs_topk.png/.pdf`

---

## 5. 主要输出文件清单（默认输出目录）

脚本 `--output_dir` 默认设置为：

`/home/zuoyiyi/SNN/TCGA/stability_comparison_plots_nestedcv/visualization`

常见输出：

- Fig1：`fig1_xai_method_boxplot_top{top_k}.png/.pdf`
- Fig2：`fig2_xai_category_boxplot_top{top_k}.png/.pdf`
- Fig3：`fig3_stability_heatmap_top{top_k}.png/.pdf`
- Fig4：`fig4_kuncheva_vs_topk.png/.pdf`
- Fig5：`fig5_rbo_vs_topk.png/.pdf`

统计结果：

- `statistical_tests_xai_methods_top{top_k}.csv`
- `statistical_tests_xai_categories_top{top_k}.csv`
- `within_cancer_method_tests_overall_top{top_k}_{metric}.csv`
- `within_cancer_method_tests_pairwise_top{top_k}_{metric}.csv`

---

## 6. 备注：关于 mean 与 median

- Fig1–Fig3 默认使用 `*_median`，因为箱线图与配对秩检验对异常值更稳健。
- Fig4/5 在 `*_median` 模式下使用 `Median ± IQR`；在 `*_mean` 模式下使用 `Mean ± SE`。
