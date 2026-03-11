# evaluate_faithfulness.py 使用说明

`operation/evaluate_faithfulness.py` 用于评估 XAI 方法的 **faithfulness**。

它对每个 XAI 方法给出的“特征重要性排序”做 **Deletion Curve**：逐步屏蔽 Top-K 特征并观察模型性能（C-index）下降；同时输出 **AOPC**（曲线相对 baseline 的平均下降量）。

屏蔽方式是 **Mean Masking**：将被屏蔽特征整列替换为验证集中的该列全局均值（不是置 0）。

---

## 1. 直接运行（命令行）

在项目根目录（包含 `datasets_csv/`、`results_2/`、`*_results_2/` 等目录的位置）运行：

### 1.1 单次运行（一个 cancer / repeat / fold）

```bash
python operation/evaluate_faithfulness.py \
  --cancer BLCA \
  --repeat 0 \
  --fold 0 \
  --device cuda \
  --k_steps 0,10,20,50,100,200,500,1000,1500,2000
```

### 1.2 批量运行（results_dir 下所有癌种 + repeats/folds 范围）

```bash
python operation/evaluate_faithfulness.py \
  --all_cancers \
  --repeats 0-9 \
  --folds 0-4 \
  --device cuda \
  --k_steps 0,10,20,50,100,200,500,1000,1500,2000
```

如果你不在项目根目录运行，可以加 `--project_root`：

```bash
python operation/evaluate_faithfulness.py \
  --all_cancers \
  --project_root <TCGA_PROJECT_ROOT>
```

脚本启动后会打印一次实际使用设备：

- `[Device] Using: cuda:0 (requested: cuda)`
- `[Device] Using: cpu (requested: cuda)`

---

## 2. 输入数据（脚本会读什么）

脚本在评估每个 `(cancer, repeat, fold)` 组合时，需要以下输入。

### 2.1 验证集 CSV

- **默认路径**：
  - `{project_root}/datasets_csv/preprocess_1/{cancer_lower}_{fold}_val.csv`
- **必须包含的列**：
  - `survival_months`
  - `censorship`
  - 以及全部特征列（由 `RNAseqSurvivalDataset` 提取为 `X_val`）

脚本内部读取方式：

- `val_df = pd.read_csv(val_csv)`
- `dataset = RNAseqSurvivalDataset(val_df, label_col="survival_months", seed=seed)`
- `X_val = dataset.features`
- `y_val_df = val_df[["survival_months", "censorship"]]`

### 2.2 模型权重（.pt）

- **默认路径**：
  - `{project_root}/results_2/{CANCER}/repeat{repeat}_s_{fold}_final_test_model.pt`
- **模型结构**：
  - `model_genomic.SNN(omic_input_dim=n_features, model_size_omic="small", n_classes=4)`
- **checkpoint 兼容格式**：
  - 直接是 `state_dict`
  - 或 `{"model_state_dict": state_dict}`

你也可以用 `--model_path` 明确指定某个 `.pt` 文件（仅单次模式会使用该覆盖）。

### 2.3 XAI 排名（ranking CSV）

脚本会读取 7 个 ranking CSV，并把其中的 `feature_name` 映射到验证集特征列索引，得到“从重要到不重要”的索引排序。

- **必须列**：
  - `feature_name`
- **关键约束**：
  - `feature_name` 必须全部能在验证集特征列名中找到
  - ranking 的行数必须严格等于 `n_features`
  - 文件中的行顺序就是重要性顺序（脚本不会根据 `rank` 列重新排序）

**默认路径（由脚本硬编码构造）**：

- `IG`：
  - `{project_root}/IG_results_2/{CANCER}/ig_feature_importance/repeat{repeat}_fold{fold}_ig_feature_importance_ranking.csv`
- `DeepLIFT`：
  - `{project_root}/DeepLIFT_results_2/{CANCER}/deeplift_feature_importance/repeat{repeat}_fold{fold}_deeplift_feature_importance_ranking.csv`
- `DeepSHAP`：
  - `{project_root}/deepshap_results_2/{CANCER}/deepshap_feature_importance/repeat{repeat}_fold{fold}_deepshap_feature_importance_ranking.csv`
- `GradientSHAP`：
  - `{project_root}/shap_results_2/{CANCER}/shap_feature_importance/repeat{repeat}_fold{fold}_shap_feature_importance_ranking.csv`
- `PFI`：
  - `{project_root}/PFI_results_2/{CANCER}/pfi_feature_importance/repeat{repeat}_fold{fold}_pfi_feature_importance_ranking.csv`
- `LRP`：
  - `{project_root}/LRP_results_2/{CANCER}/lrp_feature_importance/repeat{repeat}_fold{fold}_lrp_feature_importance_ranking.csv`
- `KNN_CPI`：
  - `{project_root}/KNN_CPI_results_2/{CANCER}/knn_cpi_feature_importance/repeat{repeat}_fold{fold}_knn_cpi_feature_importance_ranking.csv`

如果你的目录结构不同，可以用 `--xai_base_dir` 指向包含这些 `*_results_2` 目录的 base path。

---

## 3. 对输入数据做了什么操作（核心算法）

### 3.1 k_steps 规范化

命令行传入的 `--k_steps` 会被解析成整数列表，然后在 `evaluate_deletion_curve` 中进一步规范化：

- 去重并排序
- 丢弃 `< 0`
- `k > n_features` 的会被裁剪为 `n_features`
- 如果不包含 `k=0` 会自动补上

### 3.2 Mean Masking（删除/屏蔽 Top-K）

对每个方法的 ranking 和每个 `k`：

- `feature_means = X_val.mean(axis=0)`（在验证集上计算每列全局均值）
- `topk = ranking[:k]`
- 屏蔽：`X_masked[:, topk] = feature_means[topk]`

### 3.3 Baseline 与 AOPC

- baseline：`k=0` 时的 `baseline_cindex`
- deletion curve：记录每个 `(method, k)` 的 `cindex`
- AOPC：对每个方法，计算 `drop(k) = baseline_cindex - cindex(k)`，并对所有 `k_steps` 求均值

### 3.4 Random 对照组

如果输入 ranking 字典里没有 `Random`，脚本会自动加一个：

- `Random = permutation(0..n_features-1)`（由 `--seed` 控制随机种子）

### 3.5 C-index 计算口径（SNN）

脚本使用以下风险分数定义（与训练/评估口径一致）：

- `logits = model(x_omic=X)`
- `hazards = sigmoid(logits)`
- `survival = cumprod(1 - hazards, dim=1)`
- `risk = -sum(survival, dim=1)`
- `cindex = concordance_index_censored(event_bool, survival_months, risk)[0]`

其中 `event_bool = (1 - censorship).astype(bool)`。

---

## 4. 输出文件（写出什么，用来做什么）

输出根目录：

- 默认：`{project_root}/faithfulness_results`
- 可用 `--output_dir` 覆盖

### 4.1 单次模式输出

输出目录：

- `{output_dir}/{CANCER}/repeat{repeat}_fold{fold}/`

文件：

- `deletion_curve_results.csv`
  - 列：`cancer`, `repeat`, `fold`, `method`, `k`, `cindex`
- `aopc_summary.csv`
  - 列：`cancer`, `repeat`, `fold`, `method`, `baseline_cindex`, `aopc`
- `deletion_curve.png`

### 4.2 批量模式输出

批量模式会在输出根目录写两张汇总表：

- `{output_dir}/all_models_deletion_curve.csv`
  - 每行：`(cancer, repeat, fold, method, k)`
  - 用处：用于后续可视化 deletion curve 的“长表”。
    - 例如：按 `method` 分组对同一 `k` 求均值/标准差并画均值曲线；或按 `cancer` 分面比较不同癌种的曲线形状。
- `{output_dir}/all_models_aopc.csv`
  - 每行：`(cancer, repeat, fold, method)`
  - 用处：用于后续对 AOPC 做汇总统计与可视化。
    - 例如：按 `method` 画箱线图/小提琴图；或在每个癌种内比较不同方法的 AOPC，并进行统计检验。

注意：批量模式遇到缺文件或计算异常，会打印 `[SKIP] ...` 并继续。
