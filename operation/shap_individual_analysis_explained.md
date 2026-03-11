# `operation/shap_individual_analysis.py` 说明文档

本文档用于解释脚本 `operation/shap_individual_analysis.py` 的用途、使用的数据、对数据做了哪些处理/统计分析，以及最终输出了哪些结果文件。

该脚本的核心目标：

- 对每个癌症类型、每个 nested CV 实验（`repeat × fold`）
  - 读取训练好的 SNN 模型
  - 在外层测试集（脚本中使用 `{cancer_lower}_{fold}_val.csv`）上计算 `risk` 的 SHAP 解释（GradientExplainer）
  - 用 Top 100 重要特征进行两类“预后因子”识别：
    - 基于 SHAP 值的统计检验（`ttest_1samp`）
    - 基于单因素 Cox 回归（`lifelines.CoxPHFitter`）
  - 输出每个实验的：全量特征重要性排序、实验级别汇总、基因级别 Cox 细节、以及 C-index 与因子数的相关性图

说明：本文档以脚本实际实现为准；描述使用中文，但技术名词、路径、列名保持英文。

---

## 1. 入口参数（命令行）

脚本底部定义的参数：

- `--csv_path`：CSV 数据路径（必填）。脚本会读取每个癌症、每个 fold 的 train/val CSV。
- `--results_dir`：Nested CV 训练结果目录（必填）。脚本会在其中查找每个癌症的模型与结果文件。
- `--shap_dir`：SHAP/Cox 分析结果输出目录（必填）。所有分析输出写入该目录。
- `--seed`：随机种子（默认 `1`）。主要用于创建 `RNAseqSurvivalDataset` 时的 seed 参数。

脚本主入口：`main(args)`。

---

## 2. 输入数据与目录结构

### 2.1 Nested CV 训练结果（来自 `--results_dir`）

`main()` 会遍历：

- `cancer_list = sorted(os.listdir(args.results_dir))`

并对每个癌症目录 `results_dir/{cancer}/` 读取：

#### (1) 最终测试模型

- `repeat{repeat}_s_{fold}_final_test_model.pt`

该文件应包含 `model_state_dict`（或直接是 state_dict）。脚本用：

- `model = SNN(omic_input_dim=num_features, model_size_omic='small', n_classes=4)`
- `checkpoint = torch.load(model_path)`
- `model.load_state_dict(checkpoint['model_state_dict'])`（若存在）

#### (2) 外层测试结果（用于取 C-index）

- `repeat{repeat}_fold{fold}_results.pkl`

函数 `load_cindex_from_results()` 会读取其中的：

- `results['test_cindex']`

并把它作为该实验的 C-index。

### 2.2 表达数据 CSV（来自 `--csv_path`）

对每个癌症与 fold，脚本读取：

- 验证/测试集（外层测试集）：`{csv_path}/{cancer_lower}_{fold}_val.csv`
- 训练集：`{csv_path}/{cancer_lower}_{fold}_train.csv`

其中 `cancer_lower = cancer.lower()`。

### 2.3 CSV 必要列与特征列筛选

脚本通过 `RNAseqSurvivalDataset` 提取特征与标签（来自 `operation/dataset_survival.py`）：

- 必须包含列：
  - `case_id`
  - `disc_label`
  - `survival_months`
  - `censorship`
- 特征列：所有列名包含 `"_rnaseq"` 的列会作为输入特征。

---

## 3. 模型输出与风险分数（`risk`）定义

脚本为了让 SHAP 能解释“风险分数”而不是 logits，定义了模型包装器：`RiskWrapper`。

### 3.1 `RiskWrapper` 的 forward

- 输入：RNA-seq 特征 `x`（形状 `[B, num_features]`）
- 调用 SNN：`logits = model(x_omic=x)`，形状 `[B, n_classes]`
- 转换为离散生存形式：
  - `hazards = sigmoid(logits)`
  - `survival = cumprod(1 - hazards, dim=1)`
- 定义风险分数（与训练/评估代码一致）：
  - `risk = -sum(survival, dim=1, keepdim=True)`，形状 `[B, 1]`

该 `risk` 是 SHAP 分析解释的目标输出。

---

## 4. 单个实验（repeat × fold）的分析流程

核心函数：`compute_shap_and_prognostic_factors(repeat, fold, args, num_features)`。

### 4.1 前置检查

对每个 `(repeat, fold)`：

- 检查模型文件是否存在：
  - `repeat{repeat}_s_{fold}_final_test_model.pt`
- 检查结果文件并读取 `test_cindex`：
  - `repeat{repeat}_fold{fold}_results.pkl`

任一缺失则跳过该实验。

### 4.2 数据读取与准备

- 读取 val（外层测试集）CSV，构建 `val_dataset`，得到：
  - `X_val`：特征矩阵（`DataFrame` 或 `ndarray`）
  - `feature_names`：特征名列表（若 `X_val` 为 `DataFrame`，则用列名；否则用 `Feature_{i}`）

- 读取 train CSV，构建 `train_dataset`，得到：
  - `X_train`：训练集特征（用于 SHAP 的 background）

- 用于 Cox 分析：脚本将 train 与 val 合并以提高样本量：
  - `combined_df = concat([train_df, val_df])`
  - `combined_dataset = RNAseqSurvivalDataset(combined_df, ...)`
  - `X_combined = combined_dataset.features`

### 4.3 SHAP 计算（GradientExplainer）

- background：训练集特征 `X_train`
- 解释对象：验证/测试集特征 `X_val`
- explainer：
  - `explainer = shap.GradientExplainer(RiskWrapper(model), background_t)`
- shap 计算：
  - `shap_exp = explainer(X_val_t)`
  - `shap_values = shap_exp.values`

随后若 `shap_values` 形状为 `[N, F, 1]` 则 squeeze 成 `[N, F]`。

### 4.4 风险分数计算（用于记录，不直接输出为文件）

脚本会在 val 上再算一次 `risk_scores`：

- `risk_scores = -sum(cumprod(1 - sigmoid(model(X_val))), dim=1)`

说明：该 `risk_scores` 在当前版本脚本中没有直接写入 CSV 输出，主要用于内部一致性。

### 4.5 特征重要性（基于 SHAP 的 abs mean）

重要性定义：

- `importance[j] = mean_i(|shap_values[i, j]|)`

并保存“全量特征（默认 2000 个）排序文件”。

### 4.6 Top 100 特征筛选

脚本固定筛选 Top 100（按 `importance` 降序）：

- `top_100_indices = argsort(importance)[-100:][::-1]`

后续的 SHAP 统计检验与 Cox 回归都仅在这 Top 100 上进行。

---

## 5. 预后因子识别：SHAP 统计检验

对 Top 100 的每个特征/基因 `i`：

- 取该特征在所有样本上的 SHAP 值向量：`shap_feature = shap_values[:, i]`
- 对 `shap_feature` 做单样本 t 检验：
  - `ttest_1samp(shap_feature, 0)`
  - 用于检验该特征 SHAP 值是否显著偏离 0
- 记录：
  - `mean_shap`, `std_shap`
  - `consistency_ratio = abs(sum(sign(shap_feature))) / n`
  - `positive_ratio = mean(shap_feature > 0)`

随后对 Top 100 的 p 值做多重检验校正：

- `multipletests(p_values, alpha=0.05, method="fdr_bh")`

并按校正结果给出类型：

- 若 FDR 显著且 `mean_shap > 0`：`driver`（增加风险）
- 若 FDR 显著且 `mean_shap < 0`：`protector`（降低风险）
- 否则：`not_significant`

脚本也会统计“原始 p 值 < 0.05”下 driver/protector 数量（raw）。

---

## 6. 预后因子识别：单因素 Cox 回归（Top 100）

函数：`perform_cox_analysis(X_val, val_df, top_100_indices, feature_names)`

注意：脚本实际传入的是合并后的数据：

- `perform_cox_analysis(X_combined, combined_df, top_100_indices, feature_names)`

### 6.1 Cox 输入与删失编码

脚本使用：

- 生存时间：`survival_months`
- 删失：`censorship`

并将其转换为 Cox 所需的 event（死亡=1）：

- `event = 1 - censorship`

### 6.2 单因素 Cox 拟合

对每个 Top 100 特征：

- 构造 DataFrame：
  - `T`：生存时间
  - `E`：事件指示（死亡=1）
  - `feature`：该基因表达值
- `dropna()`，样本少于 10 则跳过
- `cph = CoxPHFitter(); cph.fit(cox_data, duration_col='T', event_col='E')`

输出字段（每个基因）：

- `coef`
- `hr = exp(coef)`
- `p_value`
- `ci_lower`, `ci_upper`（脚本兼容不同 lifelines 版本的置信区间列名）
- `n_samples`, `n_events`

并用 raw p 值判定类型：

- `risk`：`p_value < 0.05` 且 `coef > 0`（HR>1）
- `protective`：`p_value < 0.05` 且 `coef < 0`（HR<1）
- 否则：`not_significant`

最后对 Cox 的 p 值做 FDR 校正：

- 增加字段：`p_adj`, `significant_fdr`, `type_fdr`

---

## 7. 实验级别与癌症级别统计汇总

### 7.1 单个实验的 `result` dict

`compute_shap_and_prognostic_factors()` 返回一个 dict，其中关键字段包括：

- `repeat`, `fold`
- `cindex`
- `total_factors_tested`（固定为 Top 100 的数量）
- `significant_factors`（SHAP：FDR 显著偏离 0 的特征数）

SHAP 统计（FDR 与 raw 两套）：

- `shap_prognostic_factors_fdr` / `shap_driver_factors_fdr` / `shap_protector_factors_fdr`
- `shap_prognostic_factors_raw` / `shap_driver_factors_raw` / `shap_protector_factors_raw`

Cox 统计（FDR 与 raw 两套）：

- `cox_prognostic_factors_fdr` / `cox_protective_factors_fdr` / `cox_risk_factors_fdr`
- `cox_prognostic_factors_raw` / `cox_protective_factors_raw` / `cox_risk_factors_raw`

细节列表：

- `prognostic_factors`：Top 100 每个特征的 SHAP 统计信息（含 `mean_shap/p_value/p_adj/type/...`）
- `cox_results`：Top 100 的 Cox 回归结果（含 `coef/hr/p_value/p_adj/type_fdr/...`）

### 7.2 单个癌症的汇总：`analyze_cancer_type()`

对一个癌症的所有成功实验（最多 10×5=50 个）汇总：

- C-index 均值/方差/最小/最大
- SHAP 与 Cox 的因子数量统计（FDR 与 raw 两套）

并做相关性分析（当实验数 > 2）：

- `pearsonr(cindices, shap_prognostic_counts_raw)`
- `spearmanr(...)`

以及 Cox 的对应相关性。

注意：脚本中“主要相关性分析”使用的是 **raw p 值**（未校正）的因子数量；FDR 版本作为对比。

---

## 8. 输出文件（`--shap_dir`）

输出目录结构：

- 每个癌症一个子目录：`{shap_dir}/{cancer}/`
- 另有全局汇总文件写在 `{shap_dir}/` 根目录

### 8.1 每个实验（repeat × fold）输出

#### (1) 全量特征重要性排序（默认 2000 特征）

- 路径：`{shap_dir}/{cancer}/shap_feature_importance/`
- 文件名：`repeat{repeat}_fold{fold}_shap_feature_importance_ranking.csv`

列名含义：

- `feature_name`：特征/基因名（RNA-seq 列名）
- `importance_score`：`mean(|SHAP|)`
- `rank`：按 `importance_score` 降序的排名（1=最重要）

### 8.2 每个癌症输出

#### (1) 实验级别汇总表

- 文件：`{shap_dir}/{cancer}/{cancer}_detailed_results.csv`

列名含义：

- `repeat`
- `fold`
- `cindex`
- `significant_factors`：SHAP（FDR）显著偏离 0 的 Top 100 特征数

SHAP 统计（FDR）：

- `shap_prognostic_factors_fdr`
- `shap_driver_factors_fdr`
- `shap_protector_factors_fdr`

SHAP 统计（raw）：

- `shap_prognostic_factors_raw`
- `shap_driver_factors_raw`
- `shap_protector_factors_raw`

Cox 统计（FDR）：

- `cox_prognostic_factors_fdr`
- `cox_risk_factors_fdr`
- `cox_protective_factors_fdr`

Cox 统计（raw）：

- `cox_prognostic_factors_raw`
- `cox_risk_factors_raw`
- `cox_protective_factors_raw`

#### (2) 完整结果对象（pickle）

- 文件：`{shap_dir}/{cancer}/{cancer}_complete_results.pkl`
- 内容：`all_results` 列表（每个元素是单个实验的 `result` dict，包含 SHAP 与 Cox 的基因级别细节列表）。

#### (3) Cox 汇总表（实验级别）

- 文件：`{shap_dir}/{cancer}/{cancer}_cox_analysis_summary.csv`
- 该表是从 `{cancer}_detailed_results.csv` 中抽取 Cox 相关列得到。

列名含义：

- `repeat`, `fold`, `cindex`
- `cox_prognostic_factors_fdr`, `cox_risk_factors_fdr`, `cox_protective_factors_fdr`
- `cox_prognostic_factors_raw`, `cox_risk_factors_raw`, `cox_protective_factors_raw`

#### (4) Cox 细节表（基因级别，跨实验汇总）

- 文件：`{shap_dir}/{cancer}/{cancer}_cox_detailed_results.csv`

每一行对应一个实验的一个基因（Top 100 内，且 Cox 拟合成功）。列包含两部分：

A. Cox 输出（来自 `cox_result`）：

- `repeat`, `fold`, `cindex`
- `gene`
- `coef`
- `hr`
- `p_value`
- `ci_lower`, `ci_upper`
- `n_samples`, `n_events`
- `type`（基于 raw p 值与 coef 的分类）
- `p_adj`（FDR 校正后的 p 值）
- `significant_fdr`（FDR 是否显著）
- `type_fdr`（FDR 口径的分类）

B. 同一基因对应的 SHAP 统计（来自 `prognostic_factors`）：

- `mean_shap`
- `abs_mean_shap`
- `shap_importance`
- `shap_p_value`
- `shap_p_adj`
- `shap_type`

#### (5) 相关性散点图

- 文件：`{shap_dir}/{cancer}/{cancer}_correlation_plots.png`

图中包含 2×3 个子图：

- 第一行（SHAP，raw p 值口径）：
  - `cindex` vs `shap_prognostic_factors_raw`
  - `cindex` vs `shap_driver_factors_raw`
  - `cindex` vs `shap_protector_factors_raw`
- 第二行（Cox，raw p 值口径）：
  - `cindex` vs `cox_prognostic_factors_raw`
  - `cindex` vs `cox_risk_factors_raw`
  - `cindex` vs `cox_protective_factors_raw`

每个子图都会：

- 画散点
- 用 `polyfit` 拟合直线
- 标注 Pearson `r` 与 p 值，并用 `* / ** / *** / ns` 标记显著性

### 8.3 全部癌症汇总输出

#### (1) 所有癌症汇总表

- 文件：`{shap_dir}/all_cancers_summary.csv`

每一行对应一个癌症类型，列来自 `stats_summary`（脚本汇总的 dict）。主要列包括：

- `cancer`
- `n_experiments`
- `cindex_mean`, `cindex_std`, `cindex_min`, `cindex_max`

SHAP 数量统计（FDR 与 raw 两套均值/方差）：

- `shap_prognostic_factors_fdr_mean`, `shap_prognostic_factors_fdr_std`
- `shap_driver_factors_fdr_mean`, `shap_driver_factors_fdr_std`
- `shap_protector_factors_fdr_mean`, `shap_protector_factors_fdr_std`
- `shap_prognostic_factors_raw_mean`, `shap_prognostic_factors_raw_std`
- `shap_driver_factors_raw_mean`, `shap_driver_factors_raw_std`
- `shap_protector_factors_raw_mean`, `shap_protector_factors_raw_std`

Cox 数量统计（FDR 与 raw 两套均值/方差）：

- `cox_prognostic_factors_fdr_mean`, `cox_prognostic_factors_fdr_std`
- `cox_risk_factors_fdr_mean`, `cox_risk_factors_fdr_std`
- `cox_protective_factors_fdr_mean`, `cox_protective_factors_fdr_std`
- `cox_prognostic_factors_raw_mean`, `cox_prognostic_factors_raw_std`
- `cox_risk_factors_raw_mean`, `cox_risk_factors_raw_std`
- `cox_protective_factors_raw_mean`, `cox_protective_factors_raw_std`

相关性（主要使用 raw 口径，附带 FDR 对比）：

- `corr_cindex_shap_prognostic_raw_pearson`
- `corr_cindex_shap_prognostic_raw_pearson_p`
- `corr_cindex_shap_prognostic_raw_spearman`
- `corr_cindex_shap_prognostic_raw_spearman_p`
- `corr_cindex_driver_raw_pearson`
- `corr_cindex_driver_raw_pearson_p`
- `corr_cindex_driver_raw_spearman`
- `corr_cindex_driver_raw_spearman_p`
- `corr_cindex_protector_raw_pearson`
- `corr_cindex_protector_raw_pearson_p`
- `corr_cindex_protector_raw_spearman`
- `corr_cindex_protector_raw_spearman_p`
- `corr_cindex_cox_prognostic_raw_pearson`
- `corr_cindex_cox_prognostic_raw_pearson_p`
- `corr_cindex_cox_prognostic_raw_spearman`
- `corr_cindex_cox_prognostic_raw_spearman_p`

FDR 对比相关性：

- `corr_cindex_shap_prognostic_fdr_pearson`
- `corr_cindex_shap_prognostic_fdr_pearson_p`
- `corr_cindex_cox_prognostic_fdr_pearson`
- `corr_cindex_cox_prognostic_fdr_pearson_p`

---

## 9. 数据操作小结（按“用了什么数据/做了什么操作/输出了什么”）

- **用了什么数据**：
  - `{csv_path}/{cancer_lower}_{fold}_train.csv`（作为 SHAP background、并参与 Cox 合并数据）
  - `{csv_path}/{cancer_lower}_{fold}_val.csv`（作为 SHAP 解释对象、并参与 Cox 合并数据）
  - `{results_dir}/{cancer}/repeat{repeat}_s_{fold}_final_test_model.pt`（加载模型）
  - `{results_dir}/{cancer}/repeat{repeat}_fold{fold}_results.pkl`（读取 `test_cindex`）

- **做了什么操作**：
  - 对 SNN 输出的离散生存 logits 计算 `risk`（`hazards/survival/risk`）
  - 计算 SHAP（GradientExplainer）
  - 以 `mean(|SHAP|)` 作为重要性，保存全量特征排序，并选 Top 100
  - Top 100 上：
    - SHAP 值对 0 做 t 检验 + FDR
    - 合并 train+val 后做单因素 Cox + FDR
  - 将每次实验的统计写入癌症级 CSV
  - 计算 C-index 与因子数的相关性，并输出散点图

- **输出了什么**：
  - 每实验：`repeat{repeat}_fold{fold}_shap_feature_importance_ranking.csv`
  - 每癌症：`{cancer}_detailed_results.csv`, `{cancer}_complete_results.pkl`, `{cancer}_cox_analysis_summary.csv`, `{cancer}_cox_detailed_results.csv`, `{cancer}_correlation_plots.png`
  - 全癌种：`all_cancers_summary.csv`
