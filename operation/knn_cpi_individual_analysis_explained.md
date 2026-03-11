# `operation/knn_cpi_individual_analysis.py` 说明文档

本文档解释脚本 `operation/knn_cpi_individual_analysis.py` 的用途、输入数据、对数据做的关键操作（KNN-CPI + Cox）、以及最终生成的输出文件与列名含义。

该脚本的核心目标：

- 对每个癌症类型、每个 nested CV 实验（`repeat × fold`）
  - 读取训练好的 SNN 模型
  - 在外层验证/测试集（脚本中使用 `{cancer_lower}_{fold}_val.csv`）上计算 **KNN 条件置换重要性**（KNN-CPI, KNN Conditional Permutation Importance）
  - 输出 **2000 个特征**（单基因）的重要性排序
  - 对 Top 100 重要特征做下游单因素 Cox 回归（`lifelines.CoxPHFitter` + FDR）
  - 汇总每个癌症的实验级别结果，并输出相关性散点图

说明：你已确认预处理输入数据 **已经是标准化后的表达矩阵**，因此本脚本直接使用 `RNAseqSurvivalDataset(...).features` 作为 KNN 距离计算与条件置换的输入。

---

## 1. 入口参数（命令行）

脚本底部定义参数：

- `--csv_path`：输入 CSV 目录（必填）
- `--results_dir`：nested CV 训练结果目录（必填）
- `--knn_cpi_dir`：KNN-CPI 分析输出目录（必填）
- `--seed`：随机种子（默认 `1`）
- `--knn_k`：KNN 邻居数 K（默认 `20`）
- `--num_repeats`：每个特征的条件置换重复次数（默认 `5`）

主入口：`main(args)`。

---

## 2. 输入数据与目录结构

### 2.1 Nested CV 训练结果（来自 `--results_dir`）

`main()` 会遍历：

- `cancer_list = sorted(os.listdir(args.results_dir))`

对每个癌症目录 `results_dir/{cancer}/`，在每个 `(repeat, fold)` 下读取：

#### (1) 最终测试模型

- `repeat{repeat}_s_{fold}_final_test_model.pt`

用于加载 SNN 模型参数。

#### (2) 外层测试结果（用于取 C-index）

- `repeat{repeat}_fold{fold}_results.pkl`

函数 `load_cindex_from_results()` 读取其中的：

- `results['test_cindex']`

并作为该实验的 `cindex` 写入输出汇总表。

### 2.2 表达数据 CSV（来自 `--csv_path`）

对每个癌症与 fold，读取：

- 验证/测试集（脚本中作为 CPI 评估集）：`{csv_path}/{cancer_lower}_{fold}_val.csv`
- 训练集：`{csv_path}/{cancer_lower}_{fold}_train.csv`

其中 `cancer_lower = cancer.lower()`。

### 2.3 CSV 必要列与特征列

脚本依赖 `RNAseqSurvivalDataset`（见 `operation/dataset_survival.py`）提取特征与标签，要求 CSV 至少包含：

- `case_id`
- `disc_label`
- `survival_months`
- `censorship`

特征列：所有列名包含 `"_rnaseq"` 的列会作为输入特征（通常为 2000 个）。

---

## 3. 模型输出与风险分数（`risk`）定义

脚本内部的 `compute_risk_score()` 使用 SNN 输出 logits 计算离散生存形式：

- `hazards = sigmoid(logits)`
- `survival = cumprod(1 - hazards, dim=1)`
- `risk = -sum(survival, dim=1)`

注意：这里的 `risk` 与训练/评估代码一致，用于：

- 计算 baseline C-index
- 作为条件置换后的预测输出，用于计算性能下降（drop）

---

## 4. KNN-CPI 核心算法

核心函数：`evaluate_knn_cpi(X, model, original_c_index, compute_drop_fn, ...)`。

### 4.1 输入矩阵

- `X`：标准化后的表达矩阵（脚本中传入 `X_val_np`），shape `[N_samples, 2000]`

### 4.2 动态 PCA（解释 85% 方差）

函数 `_build_knn_neighbors_indices()` 会：

1. 用 `sklearn.decomposition.PCA` 在 `X` 上拟合，计算 `explained_variance_ratio_`
2. 动态选择最小 `n_components` 使累计解释方差达到 `>= 0.85`
3. 打印：

- `[KNN-CPI] PCA 自动选择维度 n_components=... (>= 85% variance)`

4. 将数据投影到前 `n_components` 维：`X_reduced`

说明：该步骤用于为审稿提供“降维依据”，避免硬编码维度数。

### 4.3 KNN 建树与近邻索引

在 `X_reduced` 上：

- `NearestNeighbors(n_neighbors=k+1, metric="euclidean")`

查询得到近邻索引矩阵，并尽量移除自身索引后，返回：

- `neighbors_indices`：shape `[N_samples, K]`

### 4.4 向量化条件置换（无样本维度 for-loop）

函数 `vectorized_conditional_permute(X, feature_idx, neighbors_indices, rng)`：

对每个样本 i：

- 从 `neighbors_indices[i, :]` 的 K 个邻居中随机抽取一个邻居 `j`
- 用邻居的表达值替换当前样本该基因：
  - `X_perm[i, feature_idx] = X[j, feature_idx]`

实现使用 NumPy 高级索引完成，**严禁在样本维度使用 for 循环**。

### 4.5 模型推理与性能下降（drop）

对每个基因 `feature_idx`：

- 重复 `num_repeats` 次（默认 5）
  - 生成 `X_perm`
  - 用 `compute_risk_score(model, X_perm)` 得到 `risk_scores_perm`
  - 调用回调 `compute_drop_fn(risk_scores_perm, original_c_index, y_val)` 计算性能下降
- 将 `num_repeats` 次 drop 的均值作为该基因的 CPI 得分

脚本提供默认 drop：`default_compute_drop_fn()`

- `drop = original_c_index - cindex_perm`
- 其中 `cindex_perm = concordance_index(survival_months, -risk_scores_perm, censorship)`

最终返回：

- `importance`: shape `(2000,)` 的 numpy array

---

## 5. 单个实验（repeat × fold）的分析流程

核心函数：`compute_knn_cpi_and_prognostic_factors(repeat, fold, args, num_features)`。

### 5.1 baseline C-index

在原始 `X_val_np` 上预测：

- `risk_scores = compute_risk_score(model, X_val_np)`
- `baseline_cindex = concordance_index(y_val['survival_months'], -risk_scores, y_val['censorship'])`

该 baseline 仅用于 CPI drop 的计算（与 `original_c_index` 同义）。

### 5.2 2000 特征的 KNN-CPI 重要性

- 调用 `evaluate_knn_cpi(...)` 得到每个特征的 `importance_score`

并保存全量排序 CSV（见“输出文件”）。

### 5.3 Top 100 特征筛选

- `top_100_indices = np.argsort(importance)[-100:][::-1]`

---

## 6. 单因素 Cox 回归（Top 100）

函数：`perform_cox_analysis(X_val, val_df, top_100_indices, feature_names)`。

本脚本与其他 XAI 脚本一致：为了提高 Cox 的统计功效，会合并 train + val：

- `combined_df = concat([train_df, val_df])`
- `X_combined = RNAseqSurvivalDataset(combined_df).features`

Cox 分析使用：

- 时间：`survival_months`
- 删失：`censorship`
- 事件：`event = 1 - censorship`（死亡=1，删失=0）

对每个 Top 100 基因分别拟合单因素 Cox：


- `cph = CoxPHFitter(); cph.fit(cox_data, duration_col='T', event_col='E')`

输出：`coef/hr/p_value/ci_lower/ci_upper/n_samples/n_events`。

并做 FDR 校正，生成 `p_adj/significant_fdr/type_fdr`。

---

## 8. 输出文件（`--knn_cpi_dir`）

输出结构与 `pfi_individual_analysis.py` 类似：

- 每个癌症一个子目录：`{knn_cpi_dir}/{cancer}/`
- 全癌种汇总：`{knn_cpi_dir}/all_cancers_summary.csv`

### 8.1 每个实验（repeat × fold）输出

#### (1) 全量特征重要性排序（2000 特征）

- 路径：`{knn_cpi_dir}/{cancer}/knn_cpi_feature_importance/`
- 文件名：`repeat{repeat}_fold{fold}_knn_cpi_feature_importance_ranking.csv`

列名含义：

- `feature_name`：特征/基因名（RNA-seq 列名）
- `importance_score`：KNN-CPI 得分（平均 drop）
- `rank`：按 `importance_score` 降序排名（1=最重要）

### 8.2 每个癌症输出

#### (1) 实验级别汇总表

- 文件：`{knn_cpi_dir}/{cancer}/{cancer}_detailed_results.csv`

列名含义（每行对应一个 repeat×fold）：

- `repeat`
- `fold`
- `cindex`

Cox 统计（FDR 与 raw）：

- `cox_prognostic_factors_fdr`
- `cox_risk_factors_fdr`
- `cox_protective_factors_fdr`
- `cox_prognostic_factors_raw`
- `cox_risk_factors_raw`
- `cox_protective_factors_raw`

#### (2) 完整结果对象（pickle）

- 文件：`{knn_cpi_dir}/{cancer}/{cancer}_complete_results.pkl`
- 内容：`all_results` 列表（每个元素是单个实验的 dict，包含 `prognostic_factors` 与 `cox_results` 等）。

#### (3) Cox 汇总表（实验级别）

- 文件：`{knn_cpi_dir}/{cancer}/{cancer}_cox_analysis_summary.csv`

列名：

- `repeat`, `fold`, `cindex`
- `cox_prognostic_factors_fdr`, `cox_risk_factors_fdr`, `cox_protective_factors_fdr`
- `cox_prognostic_factors_raw`, `cox_risk_factors_raw`, `cox_protective_factors_raw`

#### (4) Cox 细节表（基因级别，跨实验汇总）

- 文件：`{knn_cpi_dir}/{cancer}/{cancer}_cox_detailed_results.csv`

每一行对应一个实验的一个基因（Top 100 且 Cox 拟合成功）。主要列包括：

- `repeat`, `fold`, `cindex`
- Cox 输出：`gene`, `coef`, `hr`, `p_value`, `p_adj`, `significant_fdr`, `type`, `type_fdr`, `ci_lower`, `ci_upper`, `n_samples`, `n_events`

### 8.3 全癌种汇总输出

#### (1) 所有癌症汇总表

- 文件：`{knn_cpi_dir}/all_cancers_summary.csv`

每行一个癌症，包含：

- `cindex_mean/std/min/max`
- Cox 数量统计（FDR 与 raw 的 mean/std）

---

## 9. 数据操作小结

- **输入**：
  - `{csv_path}/{cancer_lower}_{fold}_val.csv`（用于 KNN-CPI 与 baseline C-index）
  - `{csv_path}/{cancer_lower}_{fold}_train.csv`（用于与 val 合并做 Cox）
  - `{results_dir}/{cancer}/repeat{repeat}_s_{fold}_final_test_model.pt`（模型）
  - `{results_dir}/{cancer}/repeat{repeat}_fold{fold}_results.pkl`（读取 test_cindex）

- **核心操作**：
  - KNN-CPI：PCA 85% 方差 -> KNN 近邻 -> 向量化条件置换 -> 模型推理 -> C-index drop
  - Top 100：单因素 Cox + FDR
  - 汇总

- **核心输出**：
  - 每实验：`*_knn_cpi_feature_importance_ranking.csv`
  - 每癌种：`{cancer}_detailed_results.csv`、`{cancer}_complete_results.pkl`、`{cancer}_cox_analysis_summary.csv`、`{cancer}_cox_detailed_results.csv`
  - 全癌种：`all_cancers_summary.csv`
