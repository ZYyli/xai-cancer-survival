# XAI 生物学合理性评估 Pipeline 文档

## 📁 目录结构

```
biological_plausibility/
├── config.py                      # 全局配置文件
├── run_full_pipeline.py           # 一键运行全流程
├── scripts/
│   ├── __init__.py
│   ├── database_loader.py         # 数据库加载器
│   ├── 01_download_databases.py   # Step 1: 下载数据库
│   ├── 02_calculate_gene_scores.py # Step 2: 分类验证
│   ├── 03_statistical_analysis.py # Step 3: 统计分析
│   └── 04_visualization.py        # Step 4: 可视化
├── databases/                     # 数据库文件目录
│   ├── oncokb_genes.tsv
│   ├── cgc_genes.csv
│   ├── dgidb_interactions.tsv
│   └── civic_genes.tsv
└── outputs/                       # 输出目录
    ├── categorical_validation_summary.csv  # 分类统计汇总
    ├── {cancer}/{xai}/            # 各癌种×XAI结果
    ├── statistics/                # 统计分析结果
    └── figures/                   # 可视化图表
```

---

## 脚本逻辑关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         run_full_pipeline.py                            │
│                        (一键运行全流程控制器)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            config.py                                    │
│                         (全局配置中心)                                   │
│  ┌─────────────┬──────────────┬──────────────┬─────────────────────┐   │
│  │ 路径配置     │ 癌种/XAI列表  │ 评分规则     │ API配置/统计参数      │   │
│  └─────────────┴──────────────┴──────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Step 1           │  │ Step 2           │  │ Step 3 & 4       │
│ 01_download_     │  │ 02_calculate_    │  │ 03_visualizae_   │
│ databases.py     │  │ gene_scores.py   │  │ and_test.py      │
└────────┬─────────┘  └────────┬─────────┘  └──────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐           │
│ databases/       │  │ database_loader  │◄──────────┘
│ *.tsv, *.csv     │  │ .py              │
└──────────────────┘  └──────────────────┘
```

### 数据流向

```
XAI结果 (.pkl)                    数据库文件
     │                                │
     ▼                                ▼
┌────────────────────────────────────────────────────┐
│        02_calculate_gene_scores.py                 │
│  ┌─────────────┐    ┌──────────────────────────┐  │
│  │ 加载50个    │───▶│ CategoricalValidator     │  │
│  │ CV模型结果  │    │ 统计每个数据库的类别   │  │
│  └─────────────┘    └──────────────────────────┘  │
└────────────────────────────────────────────────────┘
                         │
                         ▼
              outputs/categorical_validation_summary.csv
              outputs/{cancer}/{xai}/*.csv
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│ 03_statistical_  │          │ 04_visualization │
│ analysis.py      │          │ .py              │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         ▼                             ▼
  statistics/*.csv              figures/*.png
```

---

## 各脚本详细说明

---

### 1. `config.py` - 全局配置文件

**功能**: 集中管理所有配置参数，供其他脚本导入使用。

#### 配置模块

| 模块 | 变量 | 说明 |
|------|------|------|
| **路径配置** | `BASE_DIR` | 项目根目录 (`biological_plausibility/`) |
| | `TCGA_DIR` | TCGA 数据目录 (`BASE_DIR.parent`) |
| | `DATABASE_DIR` | 数据库文件目录 |
| | `OUTPUT_DIR` | 输出目录 |
| **癌种/XAI** | `CANCER_TYPES` | 15 种癌症类型列表 |
| | `XAI_METHODS` | 6 种 XAI 方法列表 |
| | `TOP_K` | 分析 top-K 基因 (默认 100) |
| **新配置** | `XAI_RESULT_DIRS` | 各 XAI 方法结果目录 |
| | `TCGA_TO_EFO` | 癌种 → Open Targets EFO ID |
| | `OPENTARGETS_GRAPHQL_URL` | Open Targets GraphQL URL |
| **统计参数** | `BOOTSTRAP_N` | Bootstrap 重采样次数 (10000) |
| | `ALPHA` | 显著性水平 (0.05) |
| | `RANDOM_SEED` | 随机种子 (42) |
| **API 配置** | `ONCOKB_API_TOKEN` | OncoKB API Token (环境变量) |
| | `NCBI_API_KEY` | NCBI API Key (环境变量) |
| | `NCBI_EMAIL` | NCBI Email (环境变量) |

---

### 2. `scripts/database_loader.py` - 数据库加载器

**功能**: 加载并解析 4 个生物学数据库，提供基因查询接口。

#### 类: `DatabaseLoader`

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `_load_oncokb()` | 加载 OncoKB 数据 | - |
| `_load_cgc()` | 加载 CGC 数据 | - |
| `_load_dgidb()` | 加载 DGIdb 数据 | - |
| `_load_civic()` | 加载 CIViC 数据 | - |
| `get_oncokb_level(gene)` | 查询 OncoKB 等级 | `str` or `None` |
| `get_cgc_tier(gene)` | 查询 CGC Tier | `"Tier1"/"Tier2"` or `None` |
| `get_dgidb_status(gene)` | 查询药物状态 | `"approved_on_label"` 等 |
| `get_civic_level(gene)` | 查询 CIViC 等级 | `"A"~"E"` or `None` |

#### 输入文件

| 文件 | 格式 | 必需列 |
|------|------|--------|
| `databases/oncokb_genes.tsv` | TSV | `hugoSymbol`, `highestSensitiveLevel` |
| `databases/cgc_genes.csv` | CSV | `Gene Symbol`, `Tier` |
| `databases/dgidb_interactions.tsv` | TSV | `gene_name`, `approved` |
| `databases/civic_genes.tsv` | TSV | `molecular_profile`, `evidence_level` |

#### 依赖关系

```python
from config import DATABASE_DIR, ONCOKB_SCORES, CGC_SCORES, DGIDB_SCORES, CIVIC_SCORES
```

---

### 3. `scripts/01_download_databases.py` - 数据库下载

**功能**: 下载/创建 4 个生物学数据库文件。

#### 运行方式

```bash
python scripts/01_download_databases.py
python scripts/01_download_databases.py --skip_existing
```

#### 函数模块

| 函数 | 说明 | 需要认证 |
|------|------|----------|
| `download_oncokb()` | 下载 OncoKB 数据 | ✅ 需要 API Token |
| `download_cgc()` | 创建 CGC 模板 | ✅ 需要 COSMIC 账号手动下载 |
| `download_dgidb()` | 创建 DGIdb 模板 | ✅ 需要手动下载 |
| `download_civic()` | 下载 CIViC 数据 | ❌ 公开下载 |

#### 输入/输出

| 类型 | 路径 |
|------|------|
| **输入** | 环境变量 `ONCOKB_API_TOKEN` |
| **输出** | `databases/oncokb_genes.tsv` |
| | `databases/cgc_genes.csv` |
| | `databases/dgidb_interactions.tsv` |
| | `databases/civic_genes.tsv` |

#### 依赖关系

```python
from config import DATABASE_DIR, ONCOKB_API_TOKEN
```

---

### 4. `scripts/02_calculate_gene_scores.py` - 基因评分计算

**功能**: 对 Nested CV 模型 (5 folds × 10 repeats = 50 个模型) 的 top100 基因进行生物学合理性评分。

#### 运行方式

```bash
# 单个癌种×XAI
python scripts/02_calculate_gene_scores.py --cancer BRCA --xai DeepLIFT

# 单个癌种，所有 XAI
python scripts/02_calculate_gene_scores.py --cancer BRCA

# 所有组合
python scripts/02_calculate_gene_scores.py --all

# 启用 PubMed 查询
python scripts/02_calculate_gene_scores.py --cancer BRCA --use_pubmed
```

#### 类: `GeneScorer`

| 方法 | 说明 |
|------|------|
| `score_gene(gene, cancer_type)` | 计算单个基因的总评分 (使用 config.py 规则) |
| `_query_pubmed(gene, cancer_type)` | 查询 PubMed 文献数量 (NCBI API) |
| `_calculate_pubmed_score(count)` | 调用 `config.get_pubmed_score()` |
| `score_gene_list(genes, cancer_type)` | 批量评分 |

#### 核心函数

| 函数 | 说明 |
|------|------|
| `load_nested_cv_results()` | 加载 50 个 CV 模型结果 (.pkl) |
| `extract_top100_genes()` | 从模型中提取 top100 基因 |
| `calculate_xai_bio_score()` | 计算 XAI 综合分数 (60% 平均分 + 40% 覆盖率) |
| `process_cancer_xai_nested_cv()` | 处理单个癌种×XAI 组合 |

#### 输入/输出

| 类型 | 路径 | 说明 |
|------|------|------|
| **输入** | `{XAI}_results_2/{cancer}/{cancer}_complete_results.pkl` | Nested CV 模型结果 |
| | `databases/*.tsv, *.csv` | 数据库文件 |
| **输出** | `outputs/nested_cv_summary.csv` | 所有癌种×XAI 的汇总统计 |
| | `outputs/{cancer}/{xai}/cv_model_scores.csv` | 50 个 CV 模型的分数 |
| | `outputs/{cancer}/{xai}/all_gene_scores.csv` | 所有基因的详细评分 |

#### 输出文件结构

**nested_cv_summary.csv**:
```
cancer_type, xai_method, n_models, xai_bio_score_mean, xai_bio_score_std, 
xai_bio_score_median, xai_bio_score_ci_lower, xai_bio_score_ci_upper, 
mean_coverage, mean_gene_score
```

**cv_model_scores.csv**:
```
mean_score, coverage, xai_bio_score, repeat, fold, model_idx, n_genes
```

**all_gene_scores.csv**:
```
gene, oncokb_score, cgc_score, dgidb_score, civic_score, pubmed_score, 
pubmed_count, total_score, repeat, fold, model_idx
```

#### 依赖关系

```python
from database_loader import DatabaseLoader
from config import (
    BASE_DIR, TCGA_DIR, DATABASE_DIR, OUTPUT_DIR,
    CANCER_TYPES, XAI_METHODS,
    ONCOKB_SCORES, ONCOKB_MAX,
    CGC_SCORES, CGC_MAX,
    DGIDB_SCORES, DGIDB_MAX,
    CIVIC_SCORES, CIVIC_MAX,
    get_pubmed_score, PUBMED_MAX,
    TOTAL_MAX,
    # PubMed 查询配置
    NCBI_API_KEY, NCBI_EMAIL,
    PUBMED_API_URL, PUBMED_TIMEOUT, PUBMED_REQUEST_INTERVAL,
    build_pubmed_query
)
```

---

### 5. `scripts/03_statistical_analysis.py` - 统计分析

**功能**: 对 XAI 方法进行统计比较，包括 Kruskal-Wallis 检验、Wilcoxon 成对比较、Cliff's Delta 效应量。

#### 运行方式

```bash
python scripts/03_statistical_analysis.py
```

#### 核心函数

| 函数 | 说明 |
|------|------|
| `bootstrap_ci(data, statistic)` | 计算 Bootstrap 95% 置信区间 |
| `cliff_delta(x, y)` | 计算 Cliff's Delta 效应量 |
| `kruskal_wallis_test(groups)` | Kruskal-Wallis 非参数检验 |
| `pairwise_wilcoxon_bh(groups)` | 成对 Wilcoxon + BH 校正 |
| `analyze_cross_cancer(summary_df)` | 跨癌种 XAI 比较 |

#### Cliff's Delta 效应量解释

| |d| 范围 | 效应大小 |
|----------|----------|
| < 0.147 | 可忽略 (negligible) |
| 0.147 - 0.33 | 小效应 (small) |
| 0.33 - 0.474 | 中效应 (medium) |
| ≥ 0.474 | 大效应 (large) |

#### 输入/输出

| 类型 | 路径 | 说明 |
|------|------|------|
| **输入** | `outputs/nested_cv_summary.csv` | 汇总表 |
| | `outputs/{cancer}/{xai}/cv_model_scores.csv` | 50 个模型分数 |
| **输出** | `outputs/statistics/kruskal_wallis_results.csv` | K-W 检验结果 |
| | `outputs/statistics/pairwise_wilcoxon_results.csv` | 成对比较结果 |
| | `outputs/statistics/cross_cancer_pairwise.csv` | 跨癌种成对比较 |

#### 依赖关系

```python
from config import OUTPUT_DIR, CANCER_TYPES, XAI_METHODS, BOOTSTRAP_N, ALPHA, RANDOM_SEED
```

---

### 6. `scripts/04_visualization.py` - 可视化

**功能**: 生成 XAI 生物学合理性评估的可视化图表。

#### 运行方式

```bash
python scripts/04_visualization.py
python scripts/04_visualization.py --cancer BRCA
python scripts/04_visualization.py --all_cancers
```

#### 可视化函数

| 函数 | 说明 | 输出文件 |
|------|------|----------|
| `plot_xai_heatmap()` | XAI Bio Score 热图 (癌种×XAI) | `heatmap_xai_bio_score.png` |
| `plot_xai_bio_score_ranking()` | XAI Bio Score 排名热图 | `heatmap_xai_bio_scores.png` |
| `plot_xai_ranking_bump()` | XAI 排名变化图 (Bump Chart) | `bump_chart_xai_ranking.png` |
| `plot_xai_boxplot()` | XAI Bio Score 箱线图 | `boxplot_xai_bio_score_cv.png` |
| `plot_gene_score_violin()` | Gene Score 小提琴图 | `violin_gene_score_*.png` |
| `plot_overlap_matrix()` | 基因重叠 Jaccard 矩阵 | `overlap_matrix_*.png` |
| `plot_heatmap()` | 基因×XAI 热图 | `heatmap_genes_*.png` |

#### 输入/输出

| 类型 | 路径 |
|------|------|
| **输入** | `outputs/nested_cv_summary.csv` |
| | `outputs/{cancer}/{xai}/cv_model_scores.csv` |
| **输出** | `outputs/figures/*.png` |

#### 依赖关系

```python
from config import OUTPUT_DIR, CANCER_TYPES, XAI_METHODS
```

---

### 7. `run_full_pipeline.py` - 全流程控制器

**功能**: 一键运行完整的评估流程 (Step 1 → 4)。

#### 运行方式

```bash
# 单个癌种×单个XAI
python run_full_pipeline.py --cancer BLCA --xai DeepLIFT

# 单个癌种所有XAI
python run_full_pipeline.py --cancer BLCA

# 所有癌种×所有XAI
python run_full_pipeline.py --all

# 启用 PubMed 查询
python run_full_pipeline.py --cancer BLCA --use_pubmed

# 只运行可视化
python run_full_pipeline.py --viz_only

# 只运行统计分析
python run_full_pipeline.py --stats_only

# 跳过数据库下载
python run_full_pipeline.py --all --skip_download
```

#### 命令行参数

| 参数 | 说明 |
|------|------|
| `--cancer` | 指定癌种 |
| `--xai` | 指定 XAI 方法 |
| `--all` | 处理所有癌种×所有XAI |
| `--use_pubmed` | 启用 PubMed 查询 |
| `--skip_download` | 跳过数据库下载 |
| `--viz_only` | 只运行可视化 |
| `--stats_only` | 只运行统计分析 |

#### 执行流程

```
1. check_databases()    → 检查数据库文件是否存在
2. 01_download_databases.py  → 下载/创建数据库 (如缺失)
3. 02_calculate_gene_scores.py → 计算基因评分
4. 03_statistical_analysis.py → 统计分析
5. 04_visualization.py → 生成可视化
```

#### 依赖关系

```python
from config import CANCER_TYPES, XAI_METHODS, OUTPUT_DIR
```

---

## 🔧 环境变量配置

```bash
# OncoKB API Token (可选，用于下载完整数据)
export ONCOKB_API_TOKEN="your_token_here"

# NCBI API Key (可选，用于 PubMed 查询)
export NCBI_API_KEY="your_api_key_here"
export NCBI_EMAIL="your_email@example.com"
```

### PubMed 检索式说明

使用 MeSH 词汇提高检索精确度，检索式格式：

```
"{gene}"[Title/Abstract] AND "{mesh_term}"[MeSH Terms]
```

例如：
- EGFR + BRCA → `"EGFR"[Title/Abstract] AND "breast neoplasms"[MeSH Terms]`
- TP53 + LUAD → `"TP53"[Title/Abstract] AND "lung adenocarcinoma"[MeSH Terms]`

---

## 📊 完整输入/输出汇总

### 输入文件

| 来源 | 路径 | 说明 |
|------|------|------|
| XAI 结果 | `{XAI}_results_2/{cancer}/{cancer}_complete_results.pkl` | Nested CV 模型 (50个) |
| OncoKB | `databases/oncokb_genes.tsv` | 癌症基因临床分级 |
| CGC | `databases/cgc_genes.csv` | Cancer Gene Census |
| DGIdb | `databases/dgidb_interactions.tsv` | 药物-基因互作 |
| CIViC | `databases/civic_genes.tsv` | 临床证据汇总 (A-E 等级) |

### 输出文件

| 步骤 | 路径 | 说明 |
|------|------|------|
| Step 2 | `outputs/nested_cv_summary.csv` | 所有癌种×XAI 汇总 |
| | `outputs/{cancer}/{xai}/cv_model_scores.csv` | 50 个模型分数 |
| | `outputs/{cancer}/{xai}/all_gene_scores.csv` | 基因详细评分 |
| Step 3 | `outputs/statistics/kruskal_wallis_results.csv` | K-W 检验 |
| | `outputs/statistics/pairwise_wilcoxon_results.csv` | 成对比较 |
| | `outputs/statistics/cross_cancer_pairwise.csv` | 跨癌种比较 |
| Step 4 | `outputs/figures/*.png` | 可视化图表 |

---

## 🚀 快速开始

```bash
cd /home/zuoyiyi/SNN/TCGA/biological_plausibility

# 1. 下载数据库 (首次运行)
python scripts/01_download_databases.py

# 2. 运行单个癌种测试
python scripts/02_calculate_gene_scores.py --cancer BRCA --xai DeepLIFT

# 3. 运行完整流程
python run_full_pipeline.py --all

# 4. 启用 PubMed 查询的完整流程
export NCBI_API_KEY="your_api_key"
python run_full_pipeline.py --all --use_pubmed
```
