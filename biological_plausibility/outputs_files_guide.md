# 生物学合理性验证脚本输出文件说明

本文档说明 `scripts/02_calculate_gene_scores.py` 在运行后生成的输出目录结构和各个 CSV 文件的主要内容，方便后续分析和画图。

---

## 一、整体流程概览

### 1. 前置输入

脚本主要依赖两类输入：

- **XAI 结果（模型输出）**
  - 在 `config.py` 中通过 `XAI_RESULT_DIRS` 指定，例如：
    - `XAI_RESULT_DIRS['DeepLIFT'] = TCGA_DIR / 'DeepLIFT_results_2'`
  - 每个 XAI 方法、每个癌种都有对应的 `pkl`：
    - 结构：`{XAI_RESULT_DIRS[xai]}/{cancer}/{cancer}_complete_results.pkl`
    - 示例：`DeepLIFT_results_2/BRCA/BRCA_complete_results.pkl`
  - `pkl` 内部是一个 **列表**，每个元素对应一个模型的结果字典，包含字段：
    - `prognostic_factors`: 列表，每个元素形如 `{"gene": "TP53_rnaseq", ...}`。脚本从这里提取 Top100 基因。

- **数据库文件（生物学验证）**
  - 由 `scripts/01_download_databases.py` 下载或准备，位于 `DATABASE_DIR`（在 `config.py` 定义）下，典型包括：
    - `oncokb_genes.tsv`
    - `dgidb_interactions.tsv`
    - `civic_genes.tsv`
    - `opentargets_associations.tsv`（Open Targets 关联数据）
    - `cancermine.tsv`（CancerMine 文献挖掘数据）
  - `scripts/database_loader.py` 负责加载这些文件，并构建 “基因→癌症类型集合” 的映射和四分类逻辑。

### 2. 主脚本调用方式

```bash
# 仅跑一个癌种 + 一个 XAI 方法
python scripts/02_calculate_gene_scores.py --cancer BRCA --xai DeepLIFT

# 跑所有癌种 + 所有 XAI 方法
python scripts/02_calculate_gene_scores.py --all
```

运行后，所有输出写入 `OUTPUT_DIR`（在 `config.py` 中定义，一般为 `BASE_DIR / 'outputs'`）。

---

## 二、按“癌种 × XAI 方法”的输出目录

对于每个 `(cancer_type, xai_method)` 组合，脚本会创建一个子目录：

```text
{OUTPUT_DIR}/{cancer_type}/{xai_method}/
例如：outputs/BRCA/DeepLIFT/
```


在该目录下会生成两个核心文件：

- `gene_validation.csv`
- `model_statistics.csv`

下面分别说明。

---

## 三、gene_validation.csv（基因级验证结果）

**路径示例**：

```text
outputs/BRCA/DeepLIFT/gene_validation.csv
```

**粒度**：

- **每一行 = 某个模型的一个 Top100 基因在五个数据库中的验证结果**。
- 一个癌种 × XAI 的所有模型、所有 Top100 基因全部堆在一起。

### 1. 基础列

- **`gene`**
  - 基因名，统一为大写 HUGO symbol。
  - 例如：`TP53`, `EGFR`。

- **`cancer_type`**
  - 当前验证的 TCGA 癌种代码。
  - 例如：`BRCA`, `LUAD`, `STAD` 等。

- **`model_idx`**
  - 该基因来自哪一个模型，0 起始索引。
  - 如果某个 XAI 方法在某癌种上有 50 个模型，则 `model_idx` ∈ [0, 49]。

### 2. 各数据库的分类结果

对于每个数据库（OncoKB / DGIdb / CIViC / Open Targets / CancerMine），都有分类列：

- `{db}_class`

其中 `{db}` 可以是：`oncokb`, `dgidb`, `civic`, `opentargets`, `cancermine`。

#### `{db}_class`

当前版本是**四分类**：

- `same_only`
  - 该数据库中，这个基因 **仅在当前癌症类型** 有关联记录，没有其他癌种记录。
- `same_and_other`
  - 该数据库中，这个基因 **在当前癌症类型有记录，同时也在其他癌种有记录**。
- `other_only`
  - 该数据库中，这个基因 **仅在其他癌症类型** 有关联，当前癌种无记录。
- `not_supported`
  - 该数据库中 **没有任何癌症相关记录**（或解析不到）。

分类逻辑由 `database_loader.py` 实现，核心思想：

- 对强结构化的数据（如 Open Targets）：
  - 使用 `tcga_code` → `efo/mondo` 映射，判断基因是否出现在当前癌种的关联列表中。
  - 同时检查是否在其他癌种列表中，从而区分 `same_only` 和 `same_and_other`。
- 对文本型癌症字段（OncoKB / CIViC / CancerMine）：
  - 使用 `TCGA_CANCER_MAPPING` 给定关键词列表，通过模糊匹配判断是否属于当前癌种或其他癌种。
  - 根据匹配结果分为四类。
- 对 DGIdb：
  - 主要关心是否有"药物-基因"记录，有记录统一归为 `same_only`，无记录为 `not_supported`。
  - 因此 `dgidb_class` 在当前实现中只会出现 `same_only` / `not_supported`（不会出现 `same_and_other` / `other_only`）。

### 3. 四分类汇总列（跨数据库）

除了逐库的 `class`，脚本还为每个基因计算了跨库的统计量：

- **`n_same_any`**
  - 该基因在多少个数据库中对当前癌种有支持（`same_only` 或 `same_and_other`）。
  - 这是"广义当前癌种支持"的核心指标。

- **`n_other_any`**
  - 该基因在多少个数据库中对其他癌种有记录（`same_and_other` 或 `other_only`）。
  - 表示该基因在其他癌症类型中的支持度。

> 说明：如果你看到某些历史版本的 `gene_validation.csv` 缺少 `n_other_any` 列，这是由于旧输出文件生成于更早版本脚本。当前版本 `scripts/02_calculate_gene_scores.py` 的 `validate_gene()` 会计算并输出 `n_other_any`；建议重新运行脚本以获得完整列。

- **`n_hit_any_cancer`**
  - 该基因在多少个数据库中有任意癌种记录（`same_only` / `same_and_other` / `other_only`）。
  - 表示该基因在肿瘤学领域的总体支持度。

- **`n_same_only`**
  - 被标为 `same_only` 的库的数量。

- **`n_same_and_other`**
  - 被标为 `same_and_other` 的库的数量。

- **`n_other_only`**
  - 被标为 `other_only` 的库的数量。

- **`n_not_supported`**
  - 被标为 `not_supported` 的库的数量。

> 这些指标为后续在模型级别和汇总级别上，构造多源支持度、比例等统计提供基础。其中 `n_same_any`、`n_other_any` 和 `n_hit_any_cancer` 是最常用的三个汇总指标。

---

## 四、model_statistics.csv（模型级汇总结果）

**路径示例**：

```text
outputs/BRCA/DeepLIFT/model_statistics.csv
```

**粒度**：

- **每一行 = 单个模型的 Top100 基因在五个数据库上的整体统计结果**。

脚本通过 `TriClassValidator.calculate_hit_statistics(validation_df)` 计算这些指标。

### 1. 基本信息

- `model_idx`
  - 模型索引，与 `gene_validation.csv` 中的 `model_idx` 对应。

- `cancer_type`
  - 当前癌种，例如 `BRCA`。

- `xai_method`
  - 当前 XAI 方法，例如 `DeepLIFT`, `shap`, `IG` 等。

- `n_genes`
  - 该模型参与统计的基因数，一般为 100（Top100），如果有缺失则可能略少。

### 2. 各数据库的四分类计数与三 hit 指标

对于每个数据库 `db ∈ {oncokb, dgidb, civic, opentargets, cancermine}`，若该列存在，包含：

- `{db}_same_only`
  - 在该库中被标记为 `same_only` 的基因数量。

- `{db}_same_and_other`
  - 在该库中被标记为 `same_and_other` 的基因数量。

- `{db}_other_only`
  - 在该库中被标记为 `other_only` 的基因数量。

- `{db}_not_supported`
  - 在该库中被标记为 `not_supported` 的基因数量。

> 说明：对 DGIdb，由于其分类函数仅返回 `same_only` / `not_supported`，因此在 `model_statistics.csv` 中 `dgidb_same_and_other` 和 `dgidb_other_only` 通常恒为 0。

- `{db}_hit_same_any`
  - 定义为：`{db}_same_only + {db}_same_and_other`。
  - 即在该库中 **对当前癌种有支持** 的基因数。

- `{db}_hit_other_any`
  - 定义为：`{db}_same_and_other + {db}_other_only`。
  - 即在该库中 **对其他癌种有记录** 的基因数。

- `{db}_hit_any_cancer`
  - 定义为：`{db}_same_only + {db}_same_and_other + {db}_other_only`。
  - 即在该库中 **有任意癌种记录** 的基因数。

### 3. 四分类分布的均值统计（跨 5 库）

基于 `n_same_any / n_other_any / n_hit_any_cancer / n_same_only / n_same_and_other / n_other_only / n_not_supported` 这些 per-gene 指标，脚本为每个模型计算：

- **`mean_same_any`**
  - 所有 Top100 基因的 `n_same_any` 的平均值。
  - 表示平均每个基因在多少个数据库中"对当前癌种有支持"。

- **`mean_other_any`**
  - 所有 Top100 基因的 `n_other_any` 的平均值。
  - 表示平均每个基因在多少个数据库中"对其他癌种有记录"。

- **`mean_hit_any_cancer`**
  - `n_hit_any_cancer` 的平均值。
  - 表示平均每个基因在多少个数据库中"有任意癌种记录"。

- **`mean_same_only`**
  - `n_same_only` 的平均值。

- **`mean_same_and_other`**
  - `n_same_and_other` 的平均值。

- **`mean_other_only`**
  - `n_other_only` 的平均值。

- **`mean_not_supported`**
  - `n_not_supported` 的平均值。

### 4. 高置信度基因比例

脚本还统计了一些阈值比例，用于衡量"有较多数据库支持"的基因占比：

- `pct_same_any_ge1`, `pct_same_any_ge2`, `pct_same_any_ge3`
  - Top100 基因中，`n_same_any ≥ 1/2/3` 的比例。
  - 即至少在 1/2/3 个数据库中"对当前癌种有支持"的基因占比。

- `pct_hit_any_ge2`, `pct_hit_any_ge3`
  - 基于 `n_hit_any_cancer` 的比例：
    - 至少在 2/3 个数据库中"有任意癌种记录"的基因比例。

### 5. 总命中数（三 hit）

- `total_hit_same_any`
  - 定义为：

    ```python
    total_hit_same_any = sum(stats.get(f'{db}_hit_same_any', 0) for db in ['oncokb', 'dgidb', 'civic', 'opentargets', 'cancermine'])
    ```

  - 表示将 5 个数据库的"对当前癌种有支持"的命中数加总。

- `total_hit_other_any`
  - 定义为：

    ```python
    total_hit_other_any = sum(stats.get(f'{db}_hit_other_any', 0) for db in ['oncokb', 'dgidb', 'civic', 'opentargets', 'cancermine'])
    ```

  - 表示将 5 个数据库的"对其他癌种有记录"的命中数加总。

- `total_hit_any_cancer`
  - 定义为：

    ```python
    total_hit_any_cancer = sum(stats.get(f'{db}_hit_any_cancer', 0) for db in ['oncokb', 'dgidb', 'civic', 'opentargets', 'cancermine'])
    ```

  - 表示将 5 个数据库的"有任意癌种记录"的命中数加总。

- `total_same_only`, `total_same_and_other`, `total_other_only`
  - 分别统计各分类的总数，用于更细粒度的分析。

---

## 五、汇总级输出（跨癌种 × XAI × 模型）

上述两个文件是“局部”（单个癌种 × XAI）的结果。脚本还会在 `OUTPUT_DIR` 根目录下生成全局汇总，方便画图和比较。

### 1. all_model_statistics.csv

**路径**：

```text
outputs/all_model_statistics.csv
```

**内容**：

- 将所有 `(cancer_type, xai_method)` 组合下的 `model_statistics.csv` 纵向合并。
- 每一行仍然是一个模型，但增加了：
  - `cancer_type`
  - `xai_method`
- 包含前面介绍的所有模型级指标：
  - 各库的 `{db}_same_only`, `{db}_same_and_other`, `{db}_other_only`, `{db}_not_supported`
  - 各库的 `{db}_hit_same_any`, `{db}_hit_other_any`, `{db}_hit_any_cancer`
  - `mean_same_any`, `mean_other_any`, `mean_hit_any_cancer`, `mean_same_only`, `mean_same_and_other`, `mean_other_only`, `mean_not_supported`
  - 各种 `pct_*`、`total_hit_same_any`、`total_hit_other_any`、`total_hit_any_cancer` 等。

**用途**：

- 适合做 **全局的箱线图 / 小提琴图**：
  - 比较不同 XAI、不同癌种的模型表现分布。

---

### 2. xai_summary.csv

**路径**：

```text
outputs/xai_summary.csv
```

**构造方式**：

- 对 `all_model_statistics.csv` 按 `xai_method` 分组：

  ```python
  xai_summary = summary_df.groupby('xai_method').agg(agg_dict)
  ```

- `agg_dict` 包含：
  - `mean_same_any`, `mean_other_any`, `mean_hit_any_cancer`, `mean_same_only`, `mean_same_and_other`, `mean_other_only`, `mean_not_supported`
  - `total_hit_same_any`, `total_hit_other_any`, `total_hit_any_cancer`
  - `total_same_only`, `total_same_and_other`, `total_other_only`
  - `pct_same_any_ge1`, `pct_same_any_ge2`, `pct_same_any_ge3`, `pct_hit_any_ge2`, `pct_hit_any_ge3`
  - 以及各库的 `{db}_hit_same_any`, `{db}_hit_other_any`, `{db}_hit_any_cancer`, `{db}_same_only` 等。
- 对每个指标都计算 `mean` 和 `std`，列名形如：
  - `mean_same_any_mean`, `mean_same_any_std`
  - `mean_other_any_mean`, `mean_other_any_std`
  - `total_hit_same_any_mean`, `total_hit_same_any_std` 等。

**用途**：

- 用于比较 **不同 XAI 方法的整体表现**：
  - 例如每个 XAI 的 `mean_same_any_mean`、`total_hit_same_any_mean` 的柱状图 + 误差棒。

---

### 3. cancer_xai_summary.csv

**路径**：

```text
outputs/cancer_xai_summary.csv
```

**构造方式**：

- 对 `all_model_statistics.csv` 按 `['cancer_type', 'xai_method']` 分组：

  ```python
  cancer_xai_summary = summary_df.groupby(['cancer_type', 'xai_method']).agg(agg_dict_simple).reset_index()
  ```

- `agg_dict_simple` 典型包含：
  - `mean_same_any`, `mean_other_any`, `mean_hit_any_cancer`, `mean_same_only`, `mean_same_and_other`, `mean_other_only`, `mean_not_supported`
  - `total_hit_same_any`, `total_hit_other_any`, `total_hit_any_cancer`
  - `total_same_only`, `total_same_and_other`, `total_other_only`
  - 以及各库的 `{db}_hit_same_any`, `{db}_hit_other_any`, `{db}_hit_any_cancer`, `{db}_same_only` 等。

**用途**：

- 每一行对应一个“癌种 × XAI 方法”的组合：
  - 适合做 **热力图（heatmap）**：
    - 行：`cancer_type`
    - 列：`xai_method`
    - 值：如 `total_hit_same_any`, `mean_same_any`, `oncokb_hit_same_any` 等。

---

### 4. pivot_*.csv（透视表）

脚本还会根据若干关键指标生成透视表文件，便于直接画图：

```python
pivot_metrics = [
    'total_hit_same_any', 'total_hit_other_any', 'total_hit_any_cancer',
    'total_same_only', 'total_same_and_other', 'total_other_only',
    'mean_same_any', 'mean_other_any', 'mean_hit_any_cancer',
    'mean_same_only', 'mean_same_and_other', 'mean_other_only'
]
for db in ['oncokb', 'dgidb', 'civic', 'opentargets', 'cancermine']:
    pivot_metrics.append(f'{db}_hit_same_any')
    pivot_metrics.append(f'{db}_hit_other_any')
    pivot_metrics.append(f'{db}_hit_any_cancer')
    pivot_metrics.append(f'{db}_same_only')
    pivot_metrics.append(f'{db}_same_and_other')
    pivot_metrics.append(f'{db}_other_only')

for metric in pivot_metrics:
    pivot = cancer_xai_summary.pivot(index='cancer_type', columns='xai_method', values=metric)
    pivot.to_csv(output_dir / f"pivot_{metric}.csv")
```

**典型文件**：

- `pivot_total_hit_same_any.csv` - 总 hit_same_any（对当前癌种有支持）
- `pivot_total_hit_other_any.csv` - 总 hit_other_any（对其他癌种有记录）
- `pivot_total_hit_any_cancer.csv` - 总 hit_any_cancer（有任意癌种记录）
- `pivot_total_same_only.csv` - 总 same_only
- `pivot_total_same_and_other.csv` - 总 same_and_other
- `pivot_total_other_only.csv` - 总 other_only
- `pivot_mean_same_any.csv` - 平均 same_any
- `pivot_mean_other_any.csv` - 平均 other_any
- `pivot_mean_hit_any_cancer.csv` - 平均 hit_any_cancer
- `pivot_mean_same_only.csv` - 平均 same_only
- `pivot_mean_same_and_other.csv` - 平均 same_and_other
- `pivot_mean_other_only.csv` - 平均 other_only
- `pivot_oncokb_hit_same_any.csv` - OncoKB 的 hit_same_any
- `pivot_oncokb_hit_other_any.csv` - OncoKB 的 hit_other_any
- `pivot_oncokb_hit_any_cancer.csv` - OncoKB 的 hit_any_cancer
- `pivot_oncokb_same_only.csv` - OncoKB 的 same_only
- ... (其他数据库类似: dgidb, civic, opentargets, cancermine)

**结构**：

- 行：`cancer_type`
- 列：`xai_method`
- 单元格：对应 `metric` 在该“癌种 × XAI”上的平均值。

**用途**：

- 非常适合直接用于 `seaborn.heatmap` 等函数绘制二维热力图：

  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # 示例：绘制 hit_same_any 热力图
  df = pd.read_csv('outputs/pivot_total_hit_same_any.csv', index_col=0)
  plt.figure(figsize=(8, 6))
  sns.heatmap(df, annot=True, cmap='viridis', fmt='.1f')
  plt.title('Total hit_same_any across cancers and XAI methods')
  plt.show()
  ```

---

## 六、小结

- `gene_validation.csv`：**基因级**，每行一个模型的一个 Top100 基因，记录其在五个数据库中的四分类结果（`same_only` / `same_and_other` / `other_only` / `not_supported`）和多源支持度（`n_same_any` / `n_other_any` / `n_hit_any_cancer` / `n_same_only` / `n_same_and_other` / `n_other_only` / `n_not_supported`）。
- `model_statistics.csv`：**模型级**，每行一个模型，对 Top100 基因在五库的四分类分布进行统计，输出三 hit 指标（`hit_same_any` / `hit_other_any` / `hit_any_cancer`）、各分类计数和比例。
- `all_model_statistics.csv`：拼接所有癌种和 XAI 的模型统计，用于做全局分布图（箱线图、小提琴图等）。
- `xai_summary.csv`：按 XAI 方法聚合，计算均值和标准差，用于比较不同 XAI 的整体生物学合理性。包含四分类和三 hit 的完整统计。
- `cancer_xai_summary.csv`：按 癌种×XAI 聚合，用于热力图等二维比较。包含 `mean_same_any`, `mean_other_any`, `total_hit_same_any`, `total_hit_other_any`, `total_hit_any_cancer` 等关键指标。
- `pivot_*.csv`：由 `cancer_xai_summary.csv` 派生出来的透视表，结构为 癌种×XAI，用于直接画图。包含 `pivot_total_hit_same_any.csv`, `pivot_total_hit_other_any.csv`, `pivot_total_hit_any_cancer.csv`, `pivot_mean_same_any.csv`, `pivot_mean_other_any.csv` 等多个指标的透视表。

**四分类系统说明**：
- `same_only`：基因仅在当前癌症类型有记录（无其他癌种）
- `same_and_other`：基因在当前癌症类型有记录，同时也在其他癌种有记录
- `other_only`：基因仅在其他癌症类型有记录（当前癌种无记录）
- `not_supported`：基因在数据库中没有任何癌症关联记录

**三 hit 指标说明**：
- `hit_same_any`：对当前癌种有支持（`same_only` + `same_and_other`）
- `hit_other_any`：对其他癌种有记录（`same_and_other` + `other_only`）
- `hit_any_cancer`：对任意癌种有记录（`same_only` + `same_and_other` + `other_only`）
