# `operation/evaluate_plots_nested_cv.py` 说明文档

本文档用于解释脚本 `operation/evaluate_plots_nested_cv.py` 的用途、输入数据来源与格式、核心计算步骤、输出文件，以及各输出表格（CSV）的列名含义。

## 1. 脚本目标与整体流程

该脚本用于对 **Nested CV（10 repeats × 5 folds）** 的生存分析实验结果做“下游评估与可视化”，包括：

- **逐癌种**：
  - fold-level 的 C-index / time-dependent AUC（动态 AUC）统计
  - 基于所有外层测试集的 **整合 KM 曲线（Kaplan-Meier）** 与 **log-rank 检验**
  - 基于所有 fold 的 **平均动态 AUC 曲线**（并计算每个时间点的 Bootstrap 95% CI）
- **跨癌种（Pan-cancer）**：
  - 所有癌种合并后的整体平均动态 AUC 曲线（每个时间点合并所有癌种/所有 fold 的 AUC 值，再做 Bootstrap CI）
- **统计检验与汇总**：
  - C-index 是否显著优于随机水平 0.5（单样本单侧 t 检验 + FDR 校正）
  - C-index 分布箱线图（按癌种）并在图上标注显著性星号

脚本入口为 `main()`，默认输出目录为：

- `output_dir = "/home/zuoyiyi/SNN/TCGA/results_nested_cv_plots_1"`

## 2. 输入数据（来自 nested CV 训练输出）

### 2.1 结果目录结构

脚本默认从如下目录读取 nested CV 结果：

- `results_dir = "/home/zuoyiyi/SNN/TCGA/results_2"`

其目录结构假设为：

```
results_2/
  BLCA/
    repeat0_fold0_results.pkl
    repeat0_fold1_results.pkl
    ...
    detailed_results.csv
  BRCA/
    ...
  ...（共 15 种癌症类型，见 CANCER_TYPES）
```

其中 `repeat{r}_fold{f}_results.pkl` 的格式与 `operation/main.py` 产生的 `*_results.pkl` 一致，且必须包含键：

- `test_results`：一个 dict，以 `patient_id` 为 key；每个患者包含：
  - `risk`
  - `survival_months`
  - `censorship`

### 2.2 癌症类型列表

脚本内置分析 15 种癌症类型（`CANCER_TYPES`）：

- `BLCA`, `BRCA`, `COADREAD`, `GBMLGG`, `HNSC`, `KIRC`, `KIRP`, `LGG`, `LIHC`, `LUAD`, `LUSC`, `PAAD`, `SKCM`, `STAD`, `UCEC`

## 3. 核心数据表（脚本内部 DataFrame）

### 3.1 `load_nested_cv_results()`：逐患者记录表

函数：`load_nested_cv_results(results_dir, n_repeats=10, n_folds=5)`

- 读取每个癌症目录下的 `repeat{repeat}_fold{fold}_results.pkl`
- 解包 `test_results`
- 为每个患者生成一行记录

输出 DataFrame 列名：

- `cancer_type`：癌症类型（如 `BRCA`）
- `repeat`：重复编号（0..9）
- `fold`：fold 编号（0..4）
- `patient_id`：患者/样本标识（来自 `test_results` 的 key）
- `risk_score`：风险分数（来自 `test_results[patient_id]['risk']`）
- `survival_months`：生存时间（月）
- `censorship`：删失标记（1=删失；0=事件发生；脚本中用 `event_observed = 1 - censorship`）

说明：同一个 `patient_id` 可能在不同 repeat/fold 下出现多次；`plot_integrated_survival_curve()` 会对同一个患者跨重复的 `risk_score` 取均值以避免重复患者影响。

### 3.2 `load_cindex_data_nested_cv()`：读取每癌种 fold-level C-index

函数：`load_cindex_data_nested_cv(root_folder)`

- 对 `root_folder` 下每个癌症子目录读取：
  - `detailed_results.csv`
- 并将其整理为统一结构。

该函数输出 DataFrame 的列名：

- `Cancer_Type`：癌症类型（子目录名）
- `Concordance Index`：C-index（原列名 `test_cindex` 会被 rename）
- `repeat`
- `fold`

## 4. 计算与可视化方法

## 4.1 fold-level 分析：`fold_level_analysis()`

函数：`fold_level_analysis(data, cancer_type, cindex_data=None, n_repeats=10, n_folds=5)`

对每个 (repeat, fold)：

- **样本量过滤**：若该 fold 的记录数 `< 20`，跳过。
- **C-index**：优先从 `cindex_data` 中读取对应 `(Cancer_Type, repeat, fold)` 的 `Concordance Index`。
- **动态 AUC（mean_auc）**：
  - 取该 fold 最大随访时间 `max_time = max(survival_months)`
  - 时间点：`times = np.arange(12, max_time, 12)`（每 12 个月一个点）
  - 调用 `compute_dynamic_auc_single()`，内部使用 `sksurv.metrics.cumulative_dynamic_auc`。

输出 DataFrame `fold_results` 列名：

- `cancer_type`
- `repeat`
- `fold`
- `n_samples`：该 fold 测试集患者记录数
- `c_index`
- `mean_auc`

## 4.2 整合 KM 曲线（每癌种）：`plot_integrated_survival_curve()`

函数：`plot_integrated_survival_curve(data, cancer_type, save_dir)`

核心思想：把一个癌种的 50 个外层测试集（10×5）整合成一个“更稳健”的 KM 分层评估。

关键步骤：

1. 对每个 `patient_id`，对其跨重复（repeat）出现的 `risk_score` 取 **mean**：
   - `data_averaged = data.groupby('patient_id').agg({'risk_score':'mean', ...})`
2. 用 `data_averaged['risk_score'].median()` 作为 **全局阈值**，分成：
   - `high_risk`：`risk_score >= median`
   - `low_risk`：`risk_score < median`
3. 生存时间换算：`survival_years = survival_months / 12`
4. 对两组做 **log-rank test**（`lifelines.statistics.logrank_test`）
5. 绘制 KM 曲线（不显示置信区间），并在曲线上用 `|` 标记删失点。

输出：

- PNG 图：`cancer_{cancer_type}_survival_curve_integrated.png`

返回：

- `logrank_p`：log-rank 检验 p 值（用于写入总汇总表）

## 4.3 平均动态 AUC 曲线（每癌种）：`plot_average_auc_curves_all_folds()`

函数：`plot_average_auc_curves_all_folds(data, cancer_type, save_dir)`

思路：对该癌种所有 fold 的动态 AUC 曲线进行汇总。

- 先调用 `compute_average_auc_curves_all_folds(data)`：
  - 对每个 (repeat, fold) 算一条 AUC 曲线（时间点为每 12 个月）
  - 由于不同 fold 的最大随访时间不同，各曲线长度不同：
    - 脚本取 **最短公共长度** `min_length`，所有曲线截断到相同长度
  - 返回：
    - `times_common`（单位：年）
    - `auc_curves_trimmed`（shape: n_folds × n_timepoints）
- 在每个时间点上：
  - `auc_mean = mean(auc_curves, axis=0)`
  - 对该时间点的 AUC 值做 `bootstrap_ci(..., n_bootstrap=1000)` 得到 95% CI

输出：

- PNG 图：`cancer_{cancer_type}_dynamic_auc_average_all_folds.png`

返回：

- `mean_auc_overall`：该癌种的平均 AUC（对 `auc_mean` 再取均值）

## 4.4 泛癌种整体平均 AUC：`plot_overall_average_auc_all_cancers()`

函数：`plot_overall_average_auc_all_cancers(all_data, save_dir)`

思路：在时间点维度上把所有癌种的 AUC 值合并。

- 对每个癌种：计算其 `times` 与 `auc_curves`（同 4.3）
- 对每个时间点 `t`（按 0.01 年四舍五入）：
  - 收集所有癌种在该时间点的所有 fold AUC 值
  - 计算均值与 `bootstrap_ci(..., n_bootstrap=1000)`
  - 记录：
    - `n_cancers`：该时间点参与的癌症数
    - `n_folds`：该时间点参与的 fold 数（AUC 样本数）

输出：

- PNG 图：`overall_average_auc_all_cancers.png`
- CSV：`overall_average_auc_details.csv`

`overall_average_auc_details.csv` 列名含义：

- `time_years`：时间点（年）
- `mean_auc`：该时间点的总体平均 AUC
- `ci_lower` / `ci_upper`：Bootstrap 95% CI
- `n_cancers`：该时间点参与计算的癌症数
- `n_folds`：该时间点参与计算的 fold 数（AUC 样本数）

## 4.5 C-index 置信区间与显著性：`draw_cindex_boxplot_with_stars_nested_cv()`

函数：`draw_cindex_boxplot_with_stars_nested_cv(all_fold_results, output_dir)`

该函数主要做两件事：

1. **计算每癌种 C-index 的 Bootstrap 95% CI**（用于控制台报告）
   - 使用 `cindex_ci_nested_cv()`，其内部对每个癌种的 50 个 C-index 进行 `bootstrap_ci()`。
2. **统计检验：C-index 是否显著大于 0.5**
   - 使用 `test_cindex_vs_random()`：
     - 单样本、单侧 t 检验（`H1: mean > 0.5`）
     - 对 15 个癌种的 p 值做 FDR（Benjamini–Hochberg）校正
     - 得到每癌种的 `p_value_fdr` 与显著性星号 `significance`

然后绘制 C-index 箱线图（按癌种），并把 **FDR 校正显著**的癌种在箱线图上方标注星号。

输出：

- 图：
  - `nested_cv_cindex_boxplot.png`
  - `nested_cv_cindex_boxplot.tiff`
  - `nested_cv_cindex_boxplot.pdf`
- t 检验结果表：
  - `cindex_vs_random_ttest_results.csv`

`cindex_vs_random_ttest_results.csv` 列名含义：

- `cancer_type`
- `n_values`：该癌种参与检验的 C-index 数量（理论上为 50，可能因缺失而减少）
- `mean_cindex`
- `std_cindex`
- `t_statistic`
- `p_value_raw`：原始 p 值
- `p_value_fdr`：FDR 校正后的 p 值
- `significance`：显著性标记（`***`, `**`, `*` 或空字符串）

## 5. 主程序 `main()` 的输出文件清单

`main()` 会为每个癌种输出 3 个文件，并额外输出全局汇总文件。

### 5.1 每个癌种（每癌种 3 个文件）

- `{cancer}_fold_level_results.csv`
- `cancer_{cancer}_survival_curve_integrated.png`
- `cancer_{cancer}_dynamic_auc_average_all_folds.png`

其中 `{cancer}_fold_level_results.csv` 的列名含义：

- `cancer_type`
- `repeat`
- `fold`
- `n_samples`
- `c_index`
- `mean_auc`

### 5.2 所有癌种的汇总输出

- `all_cancers_fold_analysis_summary.csv`
- `nested_cv_cindex_boxplot.png/.pdf/.tiff`
- `cindex_vs_random_ttest_results.csv`
- `overall_average_auc_all_cancers.png`
- `overall_average_auc_details.csv`

`all_cancers_fold_analysis_summary.csv` 列名含义（每行=一个癌种）：

- `cancer_type`
- `n_folds`：实际参与 fold-level 分析的 fold 数（可能小于 50，因为脚本会过滤样本量不足或缺失数据的 fold）
- `integrated_logrank_p`：整合 KM 曲线的 log-rank p 值
- `mean_cindex` / `std_cindex`
- `ci_lower_cindex` / `ci_upper_cindex`：C-index 均值的 Bootstrap 95% CI（由 `bootstrap_ci` 对 fold-level C-index 重采样得到）
- `mean_auc` / `std_auc`
- `ci_lower_auc` / `ci_upper_auc`：mean_auc 均值的 Bootstrap 95% CI

## 6. 统计方法说明（Bootstrap 与 FDR）

### 6.1 `bootstrap_ci()`

- 对输入样本（如 50 个 fold 的 C-index）做有放回抽样 `n_bootstrap` 次（脚本常用 1000 次）
- 计算每次抽样的均值，形成 bootstrap 分布
- 用分位数得到置信区间：
  - 下界：`alpha/2` 分位数
  - 上界：`1-alpha/2` 分位数

### 6.2 `test_cindex_vs_random()`

- 对每癌种做单样本单侧 t 检验：`H1: mean(C-index) > 0.5`
- 对 15 个癌种 p 值做 BH-FDR 校正

## 7. 你可能需要确认的点（为了文档完全准确）

- 本脚本假设 `results_2/{cancer}/detailed_results.csv` 与 `repeat*_fold*_results.pkl` 均已存在；它们由上游训练脚本生成（例如 `operation/main.py`）。如果你的 `results_2` 目录结构/文件命名不同，需要同步修改 `results_dir` 与读取规则。
