# 04_visualize_2.py — 图形、数据来源、处理流程与输出

本文档用于说明 `scripts/04_visualize_2.py` 生成的每一张图：使用了哪些输入数据、做了哪些处理/统计操作、会生成哪些输出文件，以及输出 CSV 的列名。

## 0. 全局配置与约定

### 输入（来自 `config.py`）

- `OUTPUT_DIR`
  - 总输出目录，其中包含按 cancer / XAI 划分的子目录。
  - 脚本从如下路径读取输入：`OUTPUT_DIR/{cancer}/{xai}/...`
- `XAI_METHODS`
  - XAI 方法的目录名列表。
- `CANCER_TYPES`
  - 用于跨癌种分析的癌种列表。

### 脚本参数

- `--out-dir`
  - 如果提供，则图与 CSV 都写入该目录。
  - 否则默认写入：`OUTPUT_DIR/visualizations/`
- `--n-models`
  - 默认 `N_SAMPLED_MODELS = 50`。

### 采样规则（配对模型）

脚本中很多步骤需要在不同 XAI 方法之间做配对比较。

- 对于给定 cancer，脚本会找出在 **所有** XAI 方法下都存在的 `model_idx`。
- 然后对这些 `model_idx` 排序，取前 `--n-models` 个（确定性）。

### 常用输入文件（按 cancer + XAI）

1. `model_statistics.csv`
   - 路径：`OUTPUT_DIR/{cancer}/{xai}/model_statistics.csv`
   - 用于 Fig5。
2. `gene_validation.csv`
   - 路径：`OUTPUT_DIR/{cancer}/{xai}/gene_validation.csv`
   - 用于 Fig6 与 Fig7。


## Fig5 — 单癌种：不同 XAI 的 Total Supported Hits

### 图展示的内容

针对单一癌种（脚本中硬编码为 `CANCER = "LIHC"`），使用模型层面的指标对不同 XAI 方法进行比较：

- `METRIC_COL = "total_hit_same_any"`

### 输入数据

- 对 `CANCER` 的所有 `XAI_METHODS` 读取 `model_statistics.csv`。
- 必需列：
  - `model_idx`
  - `total_hit_same_any`（或由 `METRIC_COL` 指定的列）

### 处理流程

1. 构建长表 `(model_idx, xai_method, total_hit_same_any)`。
2. 只保留在所有 XAI 方法下都存在的配对 `model_idx`，然后取前 `--n-models` 个。
3. 将 `model_idx` 映射为 `(repeat, fold)`：
   - `repeat = model_idx // 5`
   - `fold = model_idx % 5`
   - `replicate_id = f"{repeat}_{fold}"`
4. 统计检验（按 `replicate_id` 配对）：
   - Overall：Friedman test
   - Pairwise：Wilcoxon signed-rank test + BH-FDR correction
5. 绘图：
   - LGG 风格：violin + box + jitter points（单面板）

### 输出文件

- 图：
  - `fig5_LIHC_XAI_databases_supported_boxplot.png`
  - `fig5_LIHC_XAI_databases_supported_boxplot.pdf`
- 绘图数据：
  - `supplementary_fig5_plot_data_LIHC_models.csv`
- 汇总表：
  - `supplementary_fig5_summary_across_models_by_method.csv`
- 统计检验结果（CSV-with-sections）：
  - `statistical_tests_LIHC_XAI_methods.csv`

### 输出 CSV 列名

#### `supplementary_fig5_plot_data_LIHC_models.csv`
由 `build_plot_data_csv(...)` 生成，随后将 `N_Factors -> total_hit_same_any` 重命名。

列名：
- `Cancer`
- `Method`（展示名，例如 `G-SHAP`）
- `repeat`
- `fold`
- `total_hit_same_any`
- `replicate_id`

#### `supplementary_fig5_summary_across_models_by_method.csv`
由 `summarize_across_models_by_method(...)` 生成。

列名：
- `Method`
- `n_models`
- `mean`
- `std`
- `min`
- `q25`
- `median`
- `q75`
- `max`
- `IQR`

#### `statistical_tests_LIHC_XAI_methods.csv`
该文件包含 2 个部分（section），每个部分之前会有一行以 `#` 开头的标题。

部分 1（overall Friedman）：
- `Test`
- `Statistic`
- `p_value`
- `Effect_Size_Kendalls_W`
- `n_replicates`
- `Significant`

部分 2（pairwise Wilcoxon + FDR）：
- `Group_1`
- `Group_2`
- `Statistic`
- `p_value`
- `n_blocks`
- `p_fdr`
- `sig_fdr`


## Fig6a / Fig6b — 跨癌种 Breadth (U) 与 Depth (H/U)

### 图展示的内容

在所有 `CANCER_TYPES` 上，使用从每个模型 top genes 计算得到的两个“生物学可信度”指标来比较不同 XAI 方法：

- **Breadth (U)**：支持基因数（支持定义为 `n_same_any >= 1`）。
- **Depth (H/U)**：在支持基因中，平均每个基因对应的数据库 hit 数。

### 输入数据

- 每个 `(Cancer, XAI)` 的 `gene_validation.csv`。
- 必需列：
  - `model_idx`
  - `gene`
  - `n_same_any`

### 处理流程

对每个 cancer：

1. 在不同 XAI 方法之间选择配对的 `model_idx`（与上面的配对逻辑一致）。
2. 对每个 model_idx 计算：
   - `supported = (n_same_any >= 1)`
   - `U = sum(supported)`
   - `H = sum(n_same_any)`
   - `Depth = H / U`（若 `U <= 0` 则置为 NaN）
3. 对 (Cancer, XAI) 聚合：对 sampled models 取 median。
4. 跨癌种 summary（按 XAI）：对各癌种的 median 再取 median。
5. 跨癌种配对检验：
   - Overall：Friedman test（按 Cancer 配对）
   - Pairwise：Wilcoxon signed-rank + BH-FDR
6. 绘图：
   - Fig6a：Breadth (U) 的跨癌种 boxplot
   - Fig6b：Depth (H/U) 的跨癌种 boxplot

### 输出文件

- 绘图数据（per model）：
  - `supplementary_fig6_plot_data_breadth_depth_models.csv`
- 各癌种中位数（按 method）：
  - `supplementary_fig6_median_by_cancer_method.csv`
- 跨癌种汇总表：
  - `supplementary_fig6a_summary_across_cancers_by_xai.csv`
  - `supplementary_fig6b_summary_across_cancers_by_xai.csv`
- 统计检验结果：
  - `statistical_tests_fig6a_breadth_U_XAI_methods.csv`
  - `statistical_tests_fig6b_depth_H_over_U_XAI_methods.csv`
- 图：
  - `fig6a_boxplot_xai_breadth_U.png` / `.pdf`
  - `fig6b_boxplot_xai_depth_H_over_U.png` / `.pdf`

### 输出 CSV 列名

#### `supplementary_fig6_plot_data_breadth_depth_models.csv`
列名：
- `Cancer`
- `xai_method`
- `model_idx`
- `U`
- `H`
- `Depth`

#### `supplementary_fig6_median_by_cancer_method.csv`
列名：
- `Cancer`
- `xai_method`
- `median_U`
- `q25_U`
- `q75_U`
- `median_Depth`
- `q25_Depth`
- `q75_Depth`
- `n_models`

#### `supplementary_fig6a_summary_across_cancers_by_xai.csv` 与 `supplementary_fig6b_summary_across_cancers_by_xai.csv`
由 `summarize_across_cancers_by_xai(...)` 生成。

列名：
- `xai_method`
- `n_cancers`
- `mean`
- `std`
- `min`
- `q25`
- `median`
- `q75`
- `max`
- `IQR`

#### `statistical_tests_fig6a_breadth_U_XAI_methods.csv` 与 `statistical_tests_fig6b_depth_H_over_U_XAI_methods.csv`
两者结构与 Fig5 一致，均包含 2 个部分（section）。

总体检验部分（Friedman paired by cancer）：
- `Test`
- `Statistic`
- `p_value`
- `Effect_Size_Kendalls_W`
- `n_cancers`
- `Significant`

两两比较部分（Wilcoxon + FDR）：
- `Group_1`
- `Group_2`
- `Statistic`
- `p_value`
- `n_blocks`
- `p_fdr`
- `sig_fdr`


## Fig7a — 癌种特异的 Discovery Preference（Known vs Transfer vs Novel）

### 图展示的内容

对每个 XAI 方法，统计 top genes 落入以下三类的比例：

- **Known**：在当前 cancer 上，OpenTargets 或 CancerMine 提供证据
- **Transfer**：存在证据，但仅出现在其他 cancer（OpenTargets 或 CancerMine）
- **Novel**：OpenTargets 与 CancerMine 均无证据

该图 **仅使用癌种特异数据库证据**：
- OpenTargets (`opentargets_class`)
- CancerMine (`cancermine_class`)

### 输入数据

- 每个 `(Cancer, XAI)` 的 `gene_validation.csv`。
- 必需列：
  - `model_idx`
  - `gene`
  - `opentargets_class`
  - `cancermine_class`

### 处理流程

对每个 sampled model_idx：

1. 定义分类辅助集合：
   - 同癌种证据：`same_only`, `same_and_other`
   - 仅其他癌种证据：`other_only`
2. 基因类别判定：
   - 若 OpenTargets 或 CancerMine ∈ {`same_only`, `same_and_other`} 则为 `Known`
   - 若不为 Known 且 OpenTargets 或 CancerMine 为 `other_only` 则为 `Transfer`
   - 否则为 `Novel`
3. 计算每个模型的比例：
   - `p_known`, `p_transfer`, `p_novel`
4. 聚合到 (Cancer, XAI)：对 sampled models 取 median。
5. 聚合到跨癌种（按 XAI）：对各癌种的 median 再取 median。
6. 绘制 100% 堆叠柱图（仅使用 median proportion）。

**排序规则（x 轴 XAI 顺序）**：

- Fig7a 的 XAI 顺序按 `median_p_known` **由小到大**排序（同值按方法名字符串排序）。

### 输出文件

- 图：
  - `fig7a_cancer_specific_100pct_stacked_bar_xai_discovery_preference.png`
  - `fig7a_cancer_specific_100pct_stacked_bar_xai_discovery_preference.pdf`
- 绘图数据（per model）：
  - `supplementary_fig7a_plot_data_cancer_specific_discovery_models.csv`
- 各癌种中位数：
  - `supplementary_fig7a_proportions_median_by_cancer_method.csv`
- 跨癌种汇总（per XAI）：
  - `supplementary_fig7a_summary_across_cancers_by_xai.csv`

### 输出 CSV 列名

#### `supplementary_fig7a_plot_data_cancer_specific_discovery_models.csv`
列名：
- `Cancer`
- `xai_method`
- `model_idx`
- `p_known`
- `p_transfer`
- `p_novel`

#### `supplementary_fig7a_proportions_median_by_cancer_method.csv`
列名：
- `Cancer`
- `xai_method`
- `n_models`
- `median_p_known`
- `q25_p_known`
- `q75_p_known`
- `median_p_transfer`
- `q25_p_transfer`
- `q75_p_transfer`
- `median_p_novel`
- `q25_p_novel`
- `q75_p_novel`

#### `supplementary_fig7a_summary_across_cancers_by_xai.csv`
列名：
- `xai_method`
- `n_cancers`
- `median_p_known`
- `q25_p_known`
- `q75_p_known`
- `median_p_transfer`
- `q25_p_transfer`
- `q75_p_transfer`
- `median_p_novel`
- `q25_p_novel`
- `q75_p_novel`


## Fig7b — 非癌种特异证据（OncoKB / DGIdb）

### 图展示的内容

对每个 XAI 方法，统计 top genes 中至少被一个 **非癌种特异数据库**支持的比例：

- **Supported**：基因在 OncoKB 或 DGIdb 中有证据（任一属于 `same_only`, `same_and_other`, `other_only`）
- **Not Supported**：否则

该图使用：
- OncoKB (`oncokb_class`)
- DGIdb (`dgidb_class`)

### 输入数据

- 每个 `(Cancer, XAI)` 的 `gene_validation.csv`。
- 必需列：
  - `model_idx`
  - `gene`
  - `oncokb_class`
  - `dgidb_class`

### 处理流程

对每个 sampled model_idx：

1. 支持判定：
   - `oncokb_class` in {`same_only`, `same_and_other`, `other_only`} 或
   - `dgidb_class` in {`same_only`, `same_and_other`, `other_only`}
2. 计算每个模型的比例：
   - `p_non_cancer_evidence = (# supported genes) / (# genes)`
3. 聚合到 (Cancer, XAI)：对 sampled models 取 median。
4. 聚合到跨癌种（按 XAI）：对各癌种的 median 再取 median。
5. 绘制 100% 堆叠柱图（Supported vs Not Supported，仅使用 median proportions）。

### 输出文件

- 图：
  - `fig7b_bar_non_cancer_specific_evidence.png`
  - `fig7b_bar_non_cancer_specific_evidence.pdf`
- 绘图数据（per model）：
  - `supplementary_fig7b_plot_data_non_cancer_specific_models.csv`
- 各癌种中位数：
  - `supplementary_fig7b_proportions_median_by_cancer_method.csv`
- 跨癌种汇总（per XAI）：
  - `supplementary_fig7b_summary_across_cancers_by_xai.csv`

### 输出 CSV 列名

#### `supplementary_fig7b_plot_data_non_cancer_specific_models.csv`
列名：
- `Cancer`
- `xai_method`
- `model_idx`
- `p_non_cancer_evidence`

#### `supplementary_fig7b_proportions_median_by_cancer_method.csv`
列名：
- `Cancer`
- `xai_method`
- `n_models`
- `median_p_non_cancer_evidence`
- `q25_p_non_cancer_evidence`
- `q75_p_non_cancer_evidence`

#### `supplementary_fig7b_summary_across_cancers_by_xai.csv`
列名：
- `xai_method`
- `n_cancers`
- `median_p_non_cancer_evidence`
- `q25_p_non_cancer_evidence`
- `q75_p_non_cancer_evidence`


## Fig6c — 气泡图：Breadth (U) 与 Depth (H/U)

### 图展示的内容

以 Fig6a/Fig6b 的跨癌种中位数结果为基础，在同一张图中展示每个 XAI 方法的两个指标：

- `Breadth (U)`
- `Depth (H/U)`

每个 XAI 方法对应两枚气泡（两行），气泡大小表示该指标在不同 XAI 间的相对大小（使用非线性映射增强差异）。

### 输入数据

- 直接使用 Fig6a/Fig6b 生成的跨癌种中位数表：
  - `supplementary_fig6a_summary_across_cancers_by_xai.csv`
  - `supplementary_fig6b_summary_across_cancers_by_xai.csv`

脚本内部会把两张表按 `xai_method` 合并成一个 bubble plot 的作图数据。

### 处理流程

1. 对 `Breadth (U)` 与 `Depth (H/U)` 分别取每个 XAI 的跨癌种 `median`。
2. 将每个指标在 6 个 XAI 内做 0-1 归一化后，用指数 `gamma` 做非线性拉伸，再映射为气泡大小区间 `[size_min, size_max]`。
3. 绘制气泡图（两个 y 层：`Depth (H/U)` 与 `Breadth (U)`）。

### 输出文件

- 图：
  - `fig6c_bubble_breadth_depth_by_xai.png`
  - `fig6c_bubble_breadth_depth_by_xai.pdf`
- 绘图数据：
  - `supplementary_fig6c_bubble_plot_data_breadth_depth_by_xai.csv`

### 输出 CSV 列名

#### `supplementary_fig6c_bubble_plot_data_breadth_depth_by_xai.csv`

列名（每行=一个 XAI × 一个 metric 气泡）：

- `xai_method`
- `metric`（`Breadth (U)` 或 `Depth (H/U)`）
- `value`（该指标的跨癌种 median）
- `x`（x 轴位置）
- `y`（y 轴层位置）
- `xai_label`
- `bubble_size`


## Fig8 — 雷达图：3 指标综合评估（XAI 方法 / XAI 类别）

Fig8 包含两套雷达图：

1. **按 XAI 方法**（6 种方法）
2. **按 XAI 类别**（3 类：Perturbation-based / Gradient-based / Propagation-based）

每套雷达图都使用同样的 3 个指标：

- `Prognostic factors`
- `DB-supported hits`
- `Stability (Kuncheva)`

### 指标数据来源（summary CSV）

#### 按 XAI 方法（6 种）

- 预后因子数量：
  - `TCGA_DIR/Prognostic_comparison_plots/supplementary_fig1_summary_across_cancers_by_method.csv`
- 数据库支持 hits：
  - `TCGA_DIR/biological_plausibility/outputs/visualizations/supplementary_fig1_summary_across_cancers_by_method.csv`
- 稳定性（Kuncheva, top100）：
  - `TCGA_DIR/stability_comparison_plots_nestedcv/visualization/supplementary_fig1_summary_across_cancers_by_method_top100_kuncheva_median.csv`

#### 按 XAI 类别（3 类）

- 预后因子数量：
  - `TCGA_DIR/Prognostic_comparison_plots/supplementary_fig2_summary_across_cancers_by_category.csv`
- 数据库支持 hits：
  - `TCGA_DIR/biological_plausibility/outputs/visualizations/supplementary_fig2_summary_across_cancers_by_category.csv`
- 稳定性（Kuncheva, top100）：
  - `TCGA_DIR/stability_comparison_plots_nestedcv/visualization/supplementary_fig2_summary_across_cancers_by_category_top100_kuncheva_median.csv`

### 归一化与显示（min-max + eps）

对每个指标（metric）单独在组内做 **Min-Max 归一化**得到 `score`：

\[
score = \frac{x - \min(x)}{\max(x) - \min(x)}
\]

- 若 `max(x) <= min(x)`（该指标下各方法/类别全相同），则该指标所有对象的 `score = 0.5`。

绘图时为了避免 0 分贴中心点导致“像缺失值”的歧义，使用 `eps=0.05` 做显示下限：

\[
display = eps + score \cdot (1-eps)
\]

另外，雷达图径向刻度的数值标签被隐藏（只保留圆圈刻度线）。

### 输出文件（按 XAI 方法）

- 单方法雷达图（2×3 子图，每个方法一张）：
  - `fig8_radar_minmax_prognostic_db_stability.png`
  - `fig8_radar_minmax_prognostic_db_stability.pdf`
- 汇聚线雷达图（6 条线叠加，带 legend，带半透明填充）：
  - `fig8_radar_minmax_prognostic_db_stability_overlay_xai.png`
  - `fig8_radar_minmax_prognostic_db_stability_overlay_xai.pdf`
- 补充作图数据：
  - `supplementary_fig8_radar_plot_data_xai_3metrics_minmax.csv`

### 输出文件（按 XAI 类别）

- 单类别雷达图（1×3 子图，每类一张）：
  - `fig8_radar_minmax_categories_prognostic_db_stability.png`
  - `fig8_radar_minmax_categories_prognostic_db_stability.pdf`
- 汇聚线雷达图（3 条线叠加，带 legend，带半透明填充）：
  - `fig8_radar_minmax_categories_prognostic_db_stability_overlay_categories.png`
  - `fig8_radar_minmax_categories_prognostic_db_stability_overlay_categories.pdf`
- 补充作图数据：
  - `supplementary_fig8_radar_plot_data_xai_categories_3metrics_minmax.csv`

### Fig8 补充 CSV 列名说明

#### `supplementary_fig8_radar_plot_data_xai_3metrics_minmax.csv`

每行 = 一个 `xai_method` × 一个 `metric`。

- `xai_method`
- `metric`
- `raw_value`（该指标的原始 median 值）
- `score`（该指标内 min-max 归一化后的分数）
- `xai_label`（展示名，如 `shap -> G-SHAP`）
- `rank_metric`（**每个 metric 内**，按 `raw_value` 从大到小的 dense rank，`1`=最好）
- `composite_score_mean`（三个指标的 `score` 平均）
- `composite_score_sum`（三个指标的 `score` 求和）
- `rank_overall`（按 `composite_score_mean` 从大到小的 dense rank，`1`=最好）

#### `supplementary_fig8_radar_plot_data_xai_categories_3metrics_minmax.csv`

每行 = 一个 `xai_category` × 一个 `metric`。

- `xai_category`
- `metric`
- `raw_value`
- `score`
- `rank_metric`（每个 metric 内 dense rank，按 `raw_value` 从大到小，`1`=最好）
- `composite_score_mean`
- `composite_score_sum`
- `rank_overall`（按 `composite_score_mean` 从大到小 dense rank，`1`=最好）
