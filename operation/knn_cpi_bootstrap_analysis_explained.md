# knn_cpi_bootstrap_analysis.py 说明文档

该脚本用于对 **bootstrap 训练得到的模型** 做 KNN-CPI（KNN conditional permutation importance）特征重要性分析，并对每个 bootstrap 的 Top 100 基因做单因素 Cox 回归分析，同时生成 bootstrap 层面的汇总表与相关性散点图。

- KNN-CPI：在 OOB（out-of-bag）样本上计算 2000 特征的重要性分数并输出排序。
- Cox：使用 **全体样本（preprocessed.csv）** 对 Top 100 基因做单因素 Cox（raw p 与 FDR）。
- 相关性：计算 `cindex` 与 Cox 预后因子数量（raw / FDR）的相关性，并绘制散点图。

该脚本 **不做** KNN-CPI 显著性检验、也不统计“显著 KNN-CPI 因子数”。

---

## 1. 入口参数

脚本入口：`operation/knn_cpi_bootstrap_analysis.py`

参数：

- `--csv_path`
  - Bootstrap 数据路径（由 preprocess 输出的单癌种预处理结果目录）
  - 脚本会读取：`{csv_path}/{CANCER_UPPER}_preprocessed.csv`
- `--results_dir`
  - Bootstrap 训练结果目录（每个癌种一个子目录）
  - 脚本会读取：
    - `{results_dir}/{cancer}/models/bootstrap_model_seed{seed}.pt`
    - `{results_dir}/{cancer}/models/oob_idx_seed{seed}.npy`
    - `{results_dir}/{cancer}/cindex_array.npy`（可选，若不存在则使用 OOB 上重新计算的 baseline cindex）
- `--knn_cpi_dir`
  - 输出目录根路径
- `--num_bootstrap`
  - bootstrap 模型数量（默认 100），seed 从 1..num_bootstrap
- `--knn_k`
  - KNN-CPI 近邻数 K（默认 20）
- `--num_repeats`
  - 每个特征执行条件置换的次数（默认 5）
- `--seed`
  - 随机种子（用于 KNN 邻居选择与条件置换抽样）

---

## 2. 输入数据格式约定

### 2.1 `{CANCER_UPPER}_preprocessed.csv`

必须包含：

- `survival_months`
- `censorship`（1=删失，0=事件；脚本内部会使用 `event = 1 - censorship`）
- 2000 个 RNAseq 特征列（由 `RNAseqSurvivalDataset` 自动识别）

---

## 3. 核心流程

对每个癌种：

### 3.1 遍历 bootstrap seeds



对每个 `bootstrap_seed in [1, num_bootstrap]`：

- 加载模型：`bootstrap_model_seed{seed}.pt`
- 读取 OOB 索引：`oob_idx_seed{seed}.npy`
- 构建 OOB 特征矩阵 `X_oob`
- 在 OOB 上计算 baseline C-index：
  - 模型输出 `preds_baseline = model(X_oob)`
  - `baseline_cindex_oob = concordance_index_censored(event, time, preds_baseline)[0]`

### 3.2 KNN-CPI（OOB 上计算 2000 特征重要性）

对每个特征 `f`：重复 `num_repeats` 次：

- 动态 PCA：选择累计解释方差达到 85% 的主成分数 `n_components`
- 在 PCA 降维空间中构建 KNN（`K=knn_k`）
- 条件置换（向量化）：对每个样本 `i`，从其近邻中随机抽一个邻居 `j`，用 `X[j,f]` 替换 `X[i,f]`
- 用置换后的输入推理得到 `preds_perm`
- 计算 C-index drop：
  - `drop = baseline_cindex_oob - cindex_perm`

最终该特征的 KNN-CPI 分数取 `num_repeats` 次 drop 的均值。

### 3.3 输出 2000 特征重要性排序

每个 bootstrap seed 生成一个 CSV：

- 路径：`{knn_cpi_dir}/{cancer}/knn_cpi_feature_importance/seed{seed}_knn_cpi_ranking.csv`
- 列：
  - `feature_name`
  - `importance_score`
  - `rank`（按 `importance_score` 降序生成，1 为最高）

### 3.4 Top100 单因素 Cox

- Top100 的选择：按 KNN-CPI 的 `importance_score` 从高到低取前 100。
- Cox 拟合：对每个基因单独拟合 `CoxPHFitter()`，得到 `coef/hr/p_value/CI`。
- FDR：对 Top100 的 p 值做 `fdr_bh` 校正并添加 `p_adj`、`significant_fdr`。
- 计数：同时记录 raw p 与 FDR 口径下的风险因子数、保护因子数、预后因子总数。

---

## 4. 输出文件

### 4.1 每个癌种输出目录

输出根目录：`{knn_cpi_dir}/{cancer}/`

- **(1) 每个 bootstrap 的 KNN-CPI 2000 特征排序**
  - 目录：`knn_cpi_feature_importance/`
  - 文件：`seed{seed}_knn_cpi_ranking.csv`

- **(2) Bootstrap 逐次汇总表（每 seed 一行）**
  - 文件：`{cancer}_bootstrap_detailed_results.csv`
  - 列包括：
    - `bootstrap_seed`
    - `cindex`
    - `baseline_cindex_oob`
    - `cox_prognostic_factors_fdr/raw`
    - `cox_risk_factors_fdr/raw`
    - `cox_protective_factors_fdr/raw`

- **(3) Cox 汇总表（与 (2) 取子集，便于下游读取）**
  - 文件：`{cancer}_bootstrap_cox_analysis_summary.csv`

- **(4) Cox 细节表（基因级别，跨 seed 汇总）**
  - 文件：`{cancer}_bootstrap_cox_detailed_results.csv`
  - 每行对应一个 seed 的一个基因（Top100 且拟合成功）。

- **(5) 完整结果 pkl（便于后续复用）**
  - 文件：`{cancer}_bootstrap_complete_results.pkl`

- **(6) 统计摘要（单行 CSV）**
  - 文件：`{cancer}_bootstrap_summary.csv`
  - 包含 `cindex_mean/std/min/max` 以及 Cox 统计（raw 与 FDR 的 mean/std），以及相关性字段（当 seed 数 > 2）。

- **(7) 相关性散点图**
  - 文件：`{cancer}_bootstrap_correlation_plots.png`
  - 1×3 子图（raw p 口径）：
    - `cindex` vs `cox_prognostic_factors_raw`
    - `cindex` vs `cox_risk_factors_raw`
    - `cindex` vs `cox_protective_factors_raw`

### 4.2 全癌种汇总

- 文件：`{knn_cpi_dir}/all_cancers_bootstrap_summary.csv`
- 每行一个癌种，包含每癌种的 `*_mean/std/min/max` 统计以及相关性字段（若可计算）。

---

## 5. 与 pfi_bootstrap_analysis.py 的对应关系

- **相同点**：
  - 目录结构与 bootstrap 遍历框架一致
  - 输出每个 seed 的 2000 特征排序文件
  - Top100 单因素 Cox + 汇总 CSV + 完整 PKL
  - 相关性计算与散点图输出

- **不同点**：
  - 将 PFI 的“单特征置换”替换为 KNN-CPI 的“条件置换（KNN 邻居拷贝）”。
  - 不做任何 KNN-CPI 的显著性检验/显著因子数量统计。
