
# `operation/main.py` 说明文档

本文档用于解释脚本 `operation/main.py` 的用途、输入数据格式、主要处理流程、输出文件及其字段/列名含义。

## 1. 脚本目标与整体流程

`main.py` 用于在 RNA-seq 生存分析任务上训练一个 SNN（Self-Normalizing Network）模型，并使用 **K-fold 交叉验证**与 **多次重复实验（repeats）**进行评估。

整体流程（高层）：

- 读取预先划分好的 `train.csv` 与 `val.csv`（脚本中将原 `val` 视为测试集 `test`）
- 将每个 fold 的训练集进一步切分出内部验证集（用于 early stopping）
- 训练模型（基于离散时间生存模型的 NLL 损失）
- 在测试集上计算 C-index
- 保存每个 fold 的结果（pkl / pt）并在全部 repeats 完成后输出汇总（pkl / csv）
- 可选清理中间 checkpoint / TensorBoard 日志

## 2. 入口参数（命令行）

脚本通过 `argparse` 接收参数（常用项）：

- `--csv_path`：输入 CSV 所在目录（必填）
- `--results_dir`：输出保存目录（必填）
- `--cancer`：癌种名称（必填），用于拼接输入文件名
- `--seed`：随机种子基数（默认 `1`），每个 repeat 会派生不同 seed
- `--k`：K-fold 折数（默认 `5`）
- `--n_repeats`：重复实验次数（默认 `10`）
- `--inner_val_ratio`：从训练集中再划分内部验证集比例（默认 `0.2`）
- `--k_start`, `--k_end`：指定只跑某些 fold（`-1` 表示全跑）
- `--batch_size`：batch size（默认 `1`）
- `--max_epochs`：最大训练 epoch（默认 `50`）
- `--lr`：学习率（默认 `2e-4`）
- `--lambda_reg`：L1 正则权重（默认 `1e-5`）
- `--reg`：L2 weight decay（默认 `1e-5`）
- `--alpha_surv`：删失样本损失权重参数（默认 `0.0`，见 `loss_func.py`）
- `--gc`：梯度累积步数（默认 `32`）
- `--early_stopping`：是否启用 early stopping（默认 True）
- `--overwrite`：若存在同名结果文件是否覆盖（默认 False；不加该参数表示跳过已有结果）
- `--log_data`：是否记录 TensorBoard 日志（默认 True）
- `--cleanup_files`：实验结束后是否清理中间文件（默认 True）
- `--keep_logs`：清理时是否保留 TensorBoard 日志（默认 False）

## 3. 输入数据（CSV）

### 3.1 输入文件命名规则

对每个 fold `i`（0..k-1），脚本会读取：

- 训练集：`{csv_path}/{cancer}_{i}_train.csv`
- 测试集：`{csv_path}/{cancer}_{i}_val.csv`

注意：脚本注释中写明“原来的 val 现在是 test”。

### 3.2 输入 CSV 必要列（由 `RNAseqSurvivalDataset` 强制要求）

在 `dataset_survival.py` 的 `RNAseqSurvivalDataset` 中，输入 `DataFrame` 至少需要包含：

- `case_id`：样本/患者标识符（必须存在，否则抛错）
- `disc_label`：离散时间标签（用于离散生存建模）
- `survival_months`：生存时间（默认作为 `label_col`）
- `censorship`：删失标记（通常 `1`=删失，`0`=发生事件；脚本里 C-index 计算使用 `event_observed = 1 - censorship`）
- RNA-seq 特征列：列名中包含 `"_rnaseq"` 的所有列会被当作模型输入特征

换句话说，脚本会自动通过列名筛选：

- **特征列集合** = `{col | "_rnaseq" in col}`

## 4. 数据切分与 DataLoader

### 4.1 外层切分

外层 K-fold 的 train/test CSV 已经预先生成；脚本只负责读取。

### 4.2 内层切分（从 train 再切一个内部验证集）

脚本调用 `split_train_dataset(train_df, val_split_ratio, seed)`：

- 使用 `sklearn.model_selection.train_test_split`
- `stratify=train_df['censorship']`：按删失状态分层，尽量保持删失比例一致
- 得到：
  - `inner_train_df`
  - `inner_val_df`

该内部验证集用于 early stopping 与 checkpoint 选择。

### 4.3 DataLoader

为确保可复现：

- 每个 repeat 派生一个 `current_seed = args.seed + repeat * 1000`
- 每个 fold 使用 `seed = current_seed + fold_id` 进行内部切分
- `DataLoader` 使用 `worker_init_fn=partial(seed_worker, current_seed)`
- 并设置 `torch.Generator()` 的 seed

## 5. 模型、损失与评估指标

### 5.1 模型：`model_genomic.SNN`

- 输入维度：`args.omic_input_dim`（由特征列数决定）
- 结构：由多个 `SNN_Block` 组成的全连接网络（`Linear + SELU + AlphaDropout`），最后接一个线性分类头输出 `n_classes` 维 logits。
- `model_size_omic`：`small` 或 `big` 决定隐藏层宽度配置。

更具体的网络定义来自 `operation/model_genomic.py`：

- **输入**：`x_omic`，形状 `[B, omic_input_dim]`。
- **主干**：`fc_omic = nn.Sequential(*blocks)`，共 4 个 `SNN_Block`。
  - 每个 `SNN_Block(dim1, dim2, dropout=0.25)`：
    - `nn.Linear(dim1, dim2)`
    - `nn.SELU()`
    - `nn.AlphaDropout(p=0.25)`
- **隐藏层宽度**：
  - `small`: `omic_input_dim → 256 → 256 → 256 → 256`
  - `big`: `omic_input_dim → 1024 → 1024 → 1024 → 256`
- **输出头**：`classifier = nn.Linear(hidden[-1], n_classes)`，输出 logits `h`（`[B, n_classes]`）。
- **权重初始化**：对所有 `nn.Linear` 使用 `Normal(0, 1/sqrt(fan_in))` 初始化权重，bias 置 0。

说明：这里的 `n_classes` 对应离散生存模型的时间区间数（离散 time bins），与输入数据中的 `disc_label`（离散时间标签）配套。

### 5.2 损失：`loss_func.NLLSurvLoss`

这是离散时间生存模型的负对数似然损失：

- 网络输出 `h`（logits）
- `hazards = sigmoid(h)`
- `S = cumprod(1 - hazards)`（离散生存函数）
- 根据 `disc_label` 和 `censorship` 计算 NLL

### 5.3 正则化

- L1 正则：`utils.l1_reg_omic(model)`，只对 `model.fc_omic`（若存在）做 L1
- L2 正则：通过 `Adam(..., weight_decay=args.reg)` 实现

### 5.4 评估：C-index（Concordance Index）

- 使用 `sksurv.metrics.concordance_index_censored`
- `event_observed = (1 - censorship)`
- 风险分数（risk）：来自 `core_utils.py` 中对离散生存输出的定义：
  - `hazards = sigmoid(h)`
  - `survival = cumprod(1 - hazards, dim=1)`
  - `risk = -sum(survival)`（对时间维求和）
  该 `risk` 会写入每个外层测试集的 `test_results`（即 `repeat{repeat}_fold{fold}_results.pkl` 内每个病人的 `risk` 字段），用于后续汇总和评估。

## 6. early stopping 与 checkpoint

在 `core_utils.py` 中实现 `EarlyStoppingWithCIndex`：

- `warmup=5`：前 5 个 epoch 不保存/不触发早停
- `patience=7`：验证 loss 不下降的容忍次数
- `stop_epoch=20`：epoch >= 20 才允许早停
- 同时维护：
  - `best_loss` 对应模型（保存 `*_best_loss_checkpoint.pt`）
  - `best_cindex` 对应模型（保存 `*_best_cindex_checkpoint.pt`）

训练结束后：优先加载 best C-index checkpoint；若不存在则退化为 best loss checkpoint。

## 7. 输出文件（results_dir）

以下所有输出均写入 `--results_dir`。

### 7.1 每个 repeat × fold 的输出

#### (1) 数据切分信息

- 文件：`repeat{repeat}_fold{fold}_split_info.pkl`
- 写入位置：`{results_dir}/repeat{repeat}_fold{fold}_split_info.pkl`

该 pkl 是一个 dict（字段含义）：

- `repeat`：第几次重复（从 0 开始）
- `fold`：fold 编号
- `seed`：用于该 fold 内部切分的 seed（`current_seed + fold`）
- `original_train_indices`：原 train_df 的 index 列表
- `inner_train_indices`：内部训练集 index 列表
- `inner_val_indices`：内部验证集 index 列表
- `test_indices`：测试集 test_df 的 index 列表
- `inner_val_ratio`：内部验证集比例

#### (2) fold 结果

- 文件：`repeat{repeat}_fold{fold}_results.pkl`

该 pkl 是一个 dict（字段含义）：

- `repeat`
- `fold`
- `seed`：repeat 的 seed（`current_seed`）
- `data_split_seed`：该 fold 内部切分 seed（`current_seed + fold`）
- `test_results`：测试集逐样本预测结果（来自 `summary_survival_omic`，见下）
- `test_cindex`：该 fold 测试集 C-index
- `inner_train_size` / `inner_val_size` / `test_size`
- `model_path`：最终测试模型 pt 文件路径（见下面）
- `best_cindex_model_path`：best_cindex checkpoint 路径
- `best_loss_model_path`：best_loss checkpoint 路径
- `split_info_path`：对应 split_info 的路径

#### (3) checkpoint / 最终模型

- best C-index checkpoint：`repeat{repeat}_s_{fold}_best_cindex_checkpoint.pt`
- best loss checkpoint：`repeat{repeat}_s_{fold}_best_loss_checkpoint.pt`
- 最终测试模型（用于复现）：`repeat{repeat}_s_{fold}_final_test_model.pt`

其中最终测试模型文件保存的是一个 dict（torch.save）：

- `model_state_dict`
- `test_cindex`
- `epoch_stopped`（若启用 early stopping，则记录最佳 cindex epoch；否则为 `max_epochs`）
- `args`（训练参数字典）
- `seed`
- `repeat`
- `fold`

#### (4) TensorBoard 日志（可选）

若 `--log_data=True`，每个 fold 会创建目录：

- `{results_dir}/{fold}/events.out.tfevents.*`

记录的 scalar（见 `train_loop_survival_omic` / `validate_survival_omic` / `train`）：

- `train/loss_surv`, `train/loss`, `train/c_index`
- `val/loss_surv`, `val/loss`, `val/c-index`
- `final/test_cindex`

### 7.2 全部 repeats 完成后的汇总输出

#### (1) 总汇总 pkl

- 文件：`final_summary.pkl`

字段含义：

- `total_experiments`：总实验次数（实际跑到的 repeat×fold 数）
- `n_repeats`
- `n_folds`
- `final_mean_cindex`
- `final_std_cindex`
- `all_test_cindexes`：所有实验的测试 cindex 列表
- `repeat_results`：每个 repeat 的统计（mean/std 与列表）
- `args`：运行参数字典

#### (2) 每次实验的明细 CSV

- 文件：`detailed_results.csv`
- 列名：
  - `repeat`：重复编号（int）
  - `fold`：fold 编号（int；注意脚本这里用 enumerate，因此是 0..len-1）
  - `test_cindex`：该实验的测试 C-index（float）

#### (3) 每个 repeat 的汇总 CSV

- 文件：`repeat_summary.csv`
- 列名：
  - `repeat`：重复编号（int）或字符串 `Overall`
  - `mean_test_cindex`
  - `std_test_cindex`

## 8. `test_results`（逐样本预测结果）字段说明

`core_utils.summary_survival_omic()` 返回：

- `patient_results`：dict，以 `case_id`（若数据集提供）为 key
- `c_index`：测试集 C-index

`patient_results[case_id]` 的字段：

- `risk`：模型输出导出的风险分数（float）
- `disc_label`：该样本的离散时间标签（来自输入 `disc_label`）
- `survival_months`：生存时间（来自输入 `survival_months`）
- `censorship`：删失标记（来自输入 `censorship`）

这些内容会被写入每个 fold 的 `repeat{repeat}_fold{fold}_results.pkl` 的 `test_results` 字段中。

## 9. 清理中间文件（可选）

若 `--cleanup_files=True`，脚本会调用 `cleanup_intermediate_files(results_dir, keep_logs)`：

- 默认会删除：
  - `repeat*_s_*_best_cindex_checkpoint.pt`
  - `repeat*_s_*_best_loss_checkpoint.pt`
- 若 `--keep_logs` 未开启，还会删除：
  - `{results_dir}/*/events.out.tfevents.*`
  - `{results_dir}/[0-9]/`（fold 的 TensorBoard 目录）

并保留核心文件：

- `repeat*_s_*_final_test_model.pt`
- `repeat*_fold*_results.pkl`

## 10. 你可能需要确认的点（为了文档完全准确）

由于本仓库未在此脚本中显式生成 `{csv_path}/{cancer}_{i}_train.csv` / `{cancer}_{i}_val.csv`，它们来自上游数据准备流程。若你希望我把“这些 CSV 是如何生成的、包含哪些额外列”的说明也补上，需要你指出对应的上游脚本/路径。
