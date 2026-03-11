# SHAP Individual Analysis 使用文档

## 📋 概述

`shap_individual_analysis.py` 是一个综合分析工具，用于对生存预测模型进行可解释性分析。该脚本结合了**SHAP值分析**和**Cox比例风险回归分析**，从机器学习模型的预测机制和传统统计学的生存分析两个角度识别预后因子。

### 🎯 核心功能

1. **SHAP值分析**：识别模型如何利用基因特征做出预测
2. **Cox回归分析**：识别哪些基因与患者生存结果显著相关
3. **特征重要性排序**：对所有特征按贡献度排序
4. **相关性分析**：探索模型性能与预后因子数量的关系
5. **可视化输出**：生成散点图展示各类相关性

---

## 🔬 分析方法

### 1. SHAP分析（模型可解释性角度）

SHAP (SHapley Additive exPlanations) 分析解释模型如何做出预测决策。

#### 分析逻辑

- **Driver因子**：SHAP均值显著 > 0（对风险预测的正向贡献）
  - 特征值增加 → 模型预测风险增加
  - 表示模型将该基因视为"增加死亡风险"的信号

- **Protector因子**：SHAP均值显著 < 0（对风险预测的负向贡献）
  - 特征值增加 → 模型预测风险降低
  - 表示模型将该基因视为"降低死亡风险"的信号

#### 统计检验

- 使用**单样本t检验**测试SHAP值是否显著偏离0
- FDR多重检验校正控制假阳性率

### 2. Cox回归分析（传统统计学角度）

Cox比例风险模型评估基因表达与生存时间的关系。

#### 分析逻辑

- **Risk因子**：HR (风险比) > 1 且显著
  - 基因表达增加 → 死亡风险增加
  - 预后不良的标志物

- **Protective因子**：HR < 1 且显著
  - 基因表达增加 → 死亡风险降低
  - 预后良好的标志物

#### 统计检验

- 单因素Cox回归分析（每个基因单独分析）
- FDR多重检验校正
- 计算95%置信区间

### 3. 数据使用策略

- **SHAP分析**：使用验证集样本，训练集作为背景数据
- **Cox分析**：合并训练集和验证集，提高样本量和统计功效

---

## 🏗️ 代码结构

### 主要函数

#### `RiskWrapper`
模型包装器，将SNN模型的logits输出转换为风险分数，用于SHAP分析。

```python
logits → hazards (sigmoid) → survival (cumprod) → risk score
```

#### `perform_cox_analysis()`
对Top 100特征进行批量单因素Cox回归分析。

**输入**：
- `X_val`: 特征矩阵（合并数据集）
- `val_df`: 包含生存时间和事件信息的数据框
- `top_100_indices`: 要分析的特征索引
- `feature_names`: 特征名称列表

**输出**：
- `cox_results`: 每个基因的Cox分析结果列表
- 各类因子数量（FDR校正前后）

#### `compute_shap_and_prognostic_factors()`
计算单个实验（repeat-fold组合）的SHAP值和预后因子。

**核心步骤**：
1. 加载模型和数据
2. 计算SHAP值
3. 筛选Top 100重要特征
4. SHAP统计检验
5. Cox回归分析
6. 保存特征重要性排序
7. 返回汇总结果

#### `analyze_cancer_type()`
分析单个癌症类型的所有实验（10 repeats × 5 folds = 50个实验）。

**核心步骤**：
1. 循环处理所有repeat-fold组合
2. 汇总统计信息
3. 相关性分析（C-index vs 预后因子数量）
4. 生成可视化
5. 保存结果文件

#### `create_correlation_plots()`
创建相关性散点图，包含拟合直线和统计信息。

**图表内容**：
- 2×3子图布局
- 第一行：SHAP分析（预后因子总数、Driver、Protector vs C-index）
- 第二行：Cox分析（预后因子总数、Risk、Protective vs C-index）
- 显示Pearson相关系数、p值和显著性标记

---

## 📁 输出文件详解

### 文件结构树

```
{shap_dir}/
├── all_cancers_summary.csv                    # 所有癌症汇总
└── {cancer}/                                   # 每个癌症类型
    ├── feature_importance/                     # 特征重要性目录
    │   ├── repeat0_fold0_feature_importance_ranking.csv
    │   ├── repeat0_fold1_feature_importance_ranking.csv
    │   └── ... (共50个文件)
    ├── {cancer}_detailed_results.csv           # 详细结果汇总
    ├── {cancer}_cox_analysis_summary.csv       # Cox分析汇总
    ├── {cancer}_cox_detailed_results.csv       # Cox详细结果（每个基因）
    ├── {cancer}_complete_results.pkl           # 完整结果（pickle格式）
    └── {cancer}_correlation_plots.png          # 相关性图
```

### 文件1：特征重要性排序

**文件名**: `repeat{X}_fold{Y}_feature_importance_ranking.csv`

**数量**: 每个癌症类型50个文件

**列说明**：
| 列名 | 说明 |
|------|------|
| `feature_name` | 基因/特征名称 |
| `importance_score` | 重要性得分（基于SHAP值绝对均值） |
| `rank` | 排名（1-2000） |

**用途**：
- 查看每个实验中所有2000个特征的完整排序
- 识别不同实验中一致重要的特征
- 为后续分析选择重要特征

**示例**：
```csv
feature_name,importance_score,rank
TP53,0.0234,1
BRCA1,0.0198,2
EGFR,0.0156,3
...
```

### 文件2：详细结果汇总

**文件名**: `{cancer}_detailed_results.csv`

**数量**: 每个癌症类型1个文件

**列说明**：
| 列名 | 说明 |
|------|------|
| `repeat` | 重复次数编号（0-9） |
| `fold` | 折数编号（0-4） |
| `cindex` | 模型的C-index性能指标 |
| `significant_factors` | 显著因子总数 |
| `shap_prognostic_factors_fdr` | SHAP预后因子数（FDR校正后） |
| `shap_driver_factors_fdr` | SHAP Driver因子数（FDR校正后） |
| `shap_protector_factors_fdr` | SHAP Protector因子数（FDR校正后） |
| `shap_prognostic_factors_raw` | SHAP预后因子数（原始p值） |
| `shap_driver_factors_raw` | SHAP Driver因子数（原始p值） |
| `shap_protector_factors_raw` | SHAP Protector因子数（原始p值） |
| `cox_prognostic_factors_fdr` | Cox预后因子数（FDR校正后） |
| `cox_risk_factors_fdr` | Cox Risk因子数（FDR校正后） |
| `cox_protective_factors_fdr` | Cox Protective因子数（FDR校正后） |
| `cox_prognostic_factors_raw` | Cox预后因子数（原始p值） |
| `cox_risk_factors_raw` | Cox Risk因子数（原始p值） |
| `cox_protective_factors_raw` | Cox Protective因子数（原始p值） |

**用途**：
- 快速浏览所有50个实验的汇总统计
- 比较不同实验之间的差异
- 用于相关性分析和可视化

### 文件3：Cox分析汇总

**文件名**: `{cancer}_cox_analysis_summary.csv`

**列说明**：与详细结果汇总类似，但仅包含Cox分析相关的列。

### 文件4：Cox详细结果（每个基因）

**文件名**: `{cancer}_cox_detailed_results.csv`

**数量**: 每个癌症类型1个文件

**列说明**：
| 列名 | 说明 |
|------|------|
| `repeat` | 重复次数编号 |
| `fold` | 折数编号 |
| `cindex` | 模型C-index |
| `gene` | 基因名称 |
| `coef` | Cox回归系数 |
| `hr` | 风险比 (Hazard Ratio) |
| `p_value` | 原始p值 |
| `p_adj` | FDR校正后的p值 |
| `ci_lower` | 95%置信区间下限 |
| `ci_upper` | 95%置信区间上限 |
| `n_samples` | 样本数量 |
| `n_events` | 事件数量（死亡人数） |
| `type` | 因子类型（基于原始p值） |
| `type_fdr` | 因子类型（基于FDR校正p值） |
| `significant_fdr` | 是否FDR显著 |
| `mean_shap` | 该基因的SHAP均值 |
| `abs_mean_shap` | SHAP绝对均值 |
| `shap_importance` | SHAP重要性得分 |
| `shap_p_value` | SHAP的p值 |
| `shap_p_adj` | SHAP的FDR校正p值 |
| `shap_type` | SHAP因子类型 |

**用途**：
- 深入分析每个基因在每个实验中的详细统计信息
- 比较SHAP分析和Cox分析的一致性
- 识别在多个实验中一致显著的基因

**因子类型说明**：
- `risk` / `driver`: 增加风险的因子
- `protective` / `protector`: 降低风险的因子
- `not_significant`: 不显著
- `neutral`: 显著但效应为0（罕见）

### 文件5：完整结果（Pickle格式）

**文件名**: `{cancer}_complete_results.pkl`

**内容**: Python列表，包含所有实验的完整结果字典。

**字典结构**：
```python
{
    'repeat': int,
    'fold': int,
    'cindex': float,
    'total_factors_tested': 100,
    'significant_factors': int,
    
    # SHAP分析结果
    'shap_prognostic_factors_fdr': int,
    'shap_driver_factors_fdr': int,
    'shap_protector_factors_fdr': int,
    'shap_prognostic_factors_raw': int,
    'shap_driver_factors_raw': int,
    'shap_protector_factors_raw': int,
    
    # Cox分析结果
    'cox_prognostic_factors_fdr': int,
    'cox_protective_factors_fdr': int,
    'cox_risk_factors_fdr': int,
    'cox_prognostic_factors_raw': int,
    'cox_protective_factors_raw': int,
    'cox_risk_factors_raw': int,
    
    # 详细数据
    'prognostic_factors': [  # SHAP分析的详细信息
        {
            'gene': str,
            'importance': float,
            'mean_shap': float,
            'abs_mean_shap': float,
            'std_shap': float,
            'ttest_stat': float,
            'p_value': float,
            'p_adj': float,
            'consistency_ratio': float,
            'positive_ratio': float,
            'type': str
        },
        ...
    ],
    'cox_results': [  # Cox分析的详细信息
        {
            'gene': str,
            'coef': float,
            'hr': float,
            'p_value': float,
            'p_adj': float,
            'ci_lower': float,
            'ci_upper': float,
            'n_samples': int,
            'n_events': int,
            'type': str,
            'type_fdr': str,
            'significant_fdr': bool
        },
        ...
    ],
    'feature_names': list,
    'importance_scores': array,
    'analysis_method': 'SHAP_plus_Cox_Analysis'
}
```

**用途**：
- 用于后续Python程序的深入分析
- 保留所有原始数据，避免信息丢失
- 支持自定义分析和可视化

### 文件6：相关性可视化图

**文件名**: `{cancer}_correlation_plots.png`

**图表布局**: 2行×3列 = 6个子图

**第一行（SHAP分析）**：
1. C-index vs SHAP预后因子总数
2. C-index vs SHAP Driver因子数
3. C-index vs SHAP Protector因子数

**第二行（Cox分析）**：
1. C-index vs Cox预后因子总数
2. C-index vs Cox Risk因子数
3. C-index vs Cox Protective因子数

**图表元素**：
- 散点图：每个点代表一个实验
- 红色虚线：线性拟合直线
- 文本框：显示相关系数、p值和显著性标记
  - `***`: p < 0.001
  - `**`: p < 0.01
  - `*`: p < 0.05
  - `ns`: 不显著

**用途**：
- 直观展示模型性能与预后因子数量的关系
- 评估SHAP分析和Cox分析的有效性
- 支持论文撰写和演示

### 文件7：所有癌症汇总

**文件名**: `all_cancers_summary.csv`

**数量**: 1个文件（位于shap_dir根目录）

**列说明**：
| 列名 | 说明 |
|------|------|
| `cancer` | 癌症类型名称 |
| `n_experiments` | 成功分析的实验数量 |
| `cindex_mean` | C-index均值 |
| `cindex_std` | C-index标准差 |
| `cindex_min` | C-index最小值 |
| `cindex_max` | C-index最大值 |
| `shap_prognostic_factors_fdr_mean` | SHAP预后因子均值（FDR） |
| `shap_prognostic_factors_fdr_std` | SHAP预后因子标准差（FDR） |
| `shap_driver_factors_fdr_mean` | SHAP Driver因子均值（FDR） |
| `shap_protector_factors_fdr_mean` | SHAP Protector因子均值（FDR） |
| `shap_prognostic_factors_raw_mean` | SHAP预后因子均值（原始p值） |
| ... | （其他统计量） |
| `cox_prognostic_factors_fdr_mean` | Cox预后因子均值（FDR） |
| `cox_risk_factors_fdr_mean` | Cox Risk因子均值（FDR） |
| `cox_protective_factors_fdr_mean` | Cox Protective因子均值（FDR） |
| `cox_prognostic_factors_raw_mean` | Cox预后因子均值（原始p值） |
| ... | （其他统计量） |
| `corr_cindex_shap_prognostic_raw_pearson` | C-index与SHAP预后因子的Pearson相关系数 |
| `corr_cindex_shap_prognostic_raw_pearson_p` | 相关系数的p值 |
| `corr_cindex_shap_prognostic_raw_spearman` | Spearman相关系数 |
| `corr_cindex_cox_prognostic_raw_pearson` | C-index与Cox预后因子的Pearson相关系数 |
| `corr_cindex_cox_prognostic_raw_pearson_p` | 相关系数的p值 |
| ... | （其他相关性指标） |

**用途**：
- 跨癌症类型的比较分析
- 识别不同癌症的共性和特性
- 评估方法在不同癌症中的表现

---

## 🚀 使用方法

### 命令行参数

```bash
python shap_individual_analysis.py \
    --csv_path /path/to/csv/files \
    --results_dir /path/to/nested_cv_results \
    --shap_dir /path/to/output/shap_results \
    --seed 1
```

**参数说明**：
- `--csv_path`: CSV数据文件路径（包含各fold的训练/验证集）
  - 需要包含文件：`{cancer_lower}_{fold}_train.csv`, `{cancer_lower}_{fold}_val.csv`
  
- `--results_dir`: Nested CV结果目录（包含训练好的模型）
  - 需要包含文件：`repeat{X}_s_{Y}_final_test_model.pt`, `repeat{X}_fold{Y}_results.pkl`
  
- `--shap_dir`: SHAP分析结果保存目录
  
- `--seed`: 随机种子（默认为1）

### 输入文件要求

#### CSV数据文件
必须包含以下列：
- `survival_months`: 生存时间（月）
- `censorship`: 删失状态（0=死亡，1=删失）
- 基因表达特征列（2000个）

#### 模型文件
- 格式：PyTorch模型检查点（.pt文件）
- 包含：`model_state_dict` 或直接的state_dict

#### 结果文件
- 格式：Pickle文件（.pkl）
- 必须包含：`test_cindex` 字段

### 运行示例

```bash
# 完整运行
python shap_individual_analysis.py \
    --csv_path /home/user/TCGA/data/csv \
    --results_dir /home/user/TCGA/results/nested_cv \
    --shap_dir /home/user/TCGA/results/shap_analysis \
    --seed 42

# 输出目录结构
# /home/user/TCGA/results/shap_analysis/
# ├── all_cancers_summary.csv
# ├── BLCA/
# │   ├── feature_importance/
# │   ├── BLCA_detailed_results.csv
# │   ├── BLCA_cox_analysis_summary.csv
# │   ├── BLCA_cox_detailed_results.csv
# │   ├── BLCA_complete_results.pkl
# │   └── BLCA_correlation_plots.png
# ├── BRCA/
# │   └── ...
# └── ...
```

---

## 📊 结果解读

### 1. 模型性能评估

查看 `all_cancers_summary.csv` 中的 `cindex_mean` 列：
- C-index > 0.7：良好的预测性能
- C-index > 0.6：中等预测性能
- C-index < 0.6：较弱的预测性能

### 2. 预后因子识别

#### SHAP分析视角
- **Driver因子**：模型认为表达增加会增加风险
- **Protector因子**：模型认为表达增加会降低风险

查看文件：`{cancer}_cox_detailed_results.csv`
筛选条件：`shap_type == 'driver'` 或 `shap_type == 'protector'`

#### Cox分析视角
- **Risk因子**：统计上与死亡风险增加相关（HR > 1）
- **Protective因子**：统计上与死亡风险降低相关（HR < 1）

查看文件：`{cancer}_cox_detailed_results.csv`
筛选条件：`type_fdr == 'risk'` 或 `type_fdr == 'protective'`

### 3. 一致性分析

比较SHAP和Cox分析的结果：
```python
import pandas as pd

df = pd.read_csv('BLCA/BLCA_cox_detailed_results.csv')

# 筛选两种方法都显著的基因
consistent = df[
    (df['shap_type'] != 'not_significant') & 
    (df['type_fdr'] != 'not_significant')
]

# 检查方向是否一致
# driver + risk: 模型和统计都认为是风险因子
# protector + protective: 模型和统计都认为是保护因子
consistent['direction_match'] = (
    ((df['shap_type'] == 'driver') & (df['type_fdr'] == 'risk')) |
    ((df['shap_type'] == 'protector') & (df['type_fdr'] == 'protective'))
)

print(f"一致的基因数量: {consistent['direction_match'].sum()}")
```

### 4. 相关性解读

查看相关性图或 `all_cancers_summary.csv` 中的相关性列：

- **正相关**（r > 0，p < 0.05）：
  - 预后因子数量多 → C-index高
  - 说明模型能识别更多有效的预后信息
  
- **负相关**（r < 0，p < 0.05）：
  - 预后因子数量多 → C-index低
  - 可能存在过拟合或噪声

- **无相关**（p > 0.05）：
  - 预后因子数量与模型性能无明显关系

### 5. 特征重要性分析

识别跨实验一致重要的特征：
```python
import pandas as pd
import os

# 读取所有重要性排序文件
importance_dir = 'BLCA/feature_importance/'
all_rankings = []

for file in os.listdir(importance_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(importance_dir, file))
        df['experiment'] = file
        all_rankings.append(df)

combined = pd.concat(all_rankings)

# 计算每个基因的平均排名
avg_ranking = combined.groupby('feature_name')['rank'].agg(['mean', 'std'])
avg_ranking = avg_ranking.sort_values('mean')

# Top 10 一致重要的基因
print("跨实验一致重要的Top 10基因:")
print(avg_ranking.head(10))
```

### 6. 统计显著性解读

#### 原始p值 vs FDR校正p值

- **原始p值** (`*_raw` 列)：
  - 适合探索性分析
  - 灵敏度高，但假阳性率较高
  - 用于初步筛选候选基因

- **FDR校正p值** (`*_fdr` 列)：
  - 控制假发现率，更保守
  - 适合最终结论和发表
  - 推荐用于重要发现的报告

**建议**：
- 分析时查看两者，了解显著性程度
- 发表时主要报告FDR校正后的结果
- 可补充说明"在未校正的情况下额外发现X个候选基因"

---

## 📈 高级应用

### 1. 跨癌症比较

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取汇总文件
summary = pd.read_csv('all_cancers_summary.csv')

# 比较不同癌症的C-index
plt.figure(figsize=(12, 6))
plt.bar(summary['cancer'], summary['cindex_mean'], 
        yerr=summary['cindex_std'], capsize=5)
plt.xlabel('Cancer Type')
plt.ylabel('C-index')
plt.title('Model Performance Across Cancer Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cross_cancer_performance.png')

# 比较预后因子数量
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(summary['cancer'], summary['shap_prognostic_factors_raw_mean'])
axes[0].set_title('SHAP Prognostic Factors')
axes[0].set_xlabel('Cancer Type')
axes[0].set_ylabel('Number of Factors')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(summary['cancer'], summary['cox_prognostic_factors_raw_mean'])
axes[1].set_title('Cox Prognostic Factors')
axes[1].set_xlabel('Cancer Type')
axes[1].set_ylabel('Number of Factors')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('cross_cancer_factors.png')
```

### 2. 识别泛癌基因

识别在多个癌症中都显著的基因（泛癌标志物）：

```python
import pandas as pd
import os

cancer_types = ['BLCA', 'BRCA', 'COAD', ...]  # 所有癌症类型
all_significant_genes = {}

for cancer in cancer_types:
    file = f'{cancer}/{cancer}_cox_detailed_results.csv'
    if os.path.exists(file):
        df = pd.read_csv(file)
        
        # 筛选FDR显著的基因
        sig = df[df['significant_fdr'] == True]
        genes = sig['gene'].unique()
        
        for gene in genes:
            if gene not in all_significant_genes:
                all_significant_genes[gene] = []
            all_significant_genes[gene].append(cancer)

# 在多个癌症中都显著的基因
pan_cancer_genes = {
    gene: cancers 
    for gene, cancers in all_significant_genes.items() 
    if len(cancers) >= 3  # 至少在3个癌症中显著
}

print(f"泛癌基因数量: {len(pan_cancer_genes)}")
for gene, cancers in sorted(pan_cancer_genes.items(), 
                            key=lambda x: len(x[1]), reverse=True)[:20]:
    print(f"{gene}: {len(cancers)} cancers - {', '.join(cancers)}")
```

### 3. 生物学通路富集分析

将识别的预后基因导出用于通路分析：

```python
import pandas as pd

# 读取某个癌症的Cox详细结果
df = pd.read_csv('BLCA/BLCA_cox_detailed_results.csv')

# 筛选FDR显著的风险基因
risk_genes = df[df['type_fdr'] == 'risk']['gene'].unique()

# 筛选FDR显著的保护基因
protective_genes = df[df['type_fdr'] == 'protective']['gene'].unique()

# 保存用于富集分析
pd.DataFrame({'gene': risk_genes}).to_csv('BLCA_risk_genes.txt', 
                                           index=False, header=False)
pd.DataFrame({'gene': protective_genes}).to_csv('BLCA_protective_genes.txt', 
                                                 index=False, header=False)

print(f"Risk genes: {len(risk_genes)}")
print(f"Protective genes: {len(protective_genes)}")
```

然后使用这些基因列表在以下工具中进行富集分析：
- [DAVID](https://david.ncifcrf.gov/)
- [Enrichr](https://maayanlab.cloud/Enrichr/)
- [Metascape](https://metascape.org/)
- [g:Profiler](https://biit.cs.ut.ee/gprofiler/)

---

## ⚠️ 注意事项

### 1. 数据要求
- 确保CSV文件包含正确的生存时间和删失状态列
- 特征数量必须与模型训练时一致（默认2000）
- 检查数据中是否有缺失值

### 2. 计算资源
- SHAP计算需要GPU加速（推荐）
- 每个实验需要几分钟到十几分钟
- 预计总运行时间：数小时到数天（取决于癌症数量）

### 3. 统计考虑
- FDR校正控制假发现率，推荐用于最终报告
- 原始p值适合探索性分析
- 相关性分析需要足够的样本量（至少3个数据点）

### 4. 结果解读
- Driver/Risk因子表示"高表达 → 预后差"
- Protector/Protective因子表示"高表达 → 预后好"
- SHAP和Cox分析从不同角度评估，结果可能不完全一致
- 两种方法都显著的基因更可靠

### 5. 已知限制
- 仅分析Top 100特征（基于SHAP重要性）
- 单因素Cox分析，未考虑基因间交互
- SHAP值依赖于背景数据集的选择

---

## 🔧 故障排除

### 常见错误

#### 1. 模型文件未找到
```
错误: 模型文件不存在: repeat0_s_0_final_test_model.pt
```
**解决方案**：检查`--results_dir`路径和文件命名格式

#### 2. C-index加载失败
```
错误: 无法获取cindex，跳过 repeat0_fold0
```
**解决方案**：确保results.pkl文件存在且包含`test_cindex`字段

#### 3. Cox分析失败
```
错误: Cox分析失败: Convergence halted
```
**解决方案**：
- 可能是数据问题（常数特征、共线性）
- 检查该特征的分布
- 代码会自动跳过失败的特征

#### 4. 置信区间列名错误
```
警告: 未知的置信区间列名: ['0.95_lower', '0.95_upper']
```
**解决方案**：代码已兼容不同版本的lifelines，会自动处理

#### 5. 内存不足
```
错误: CUDA out of memory
```
**解决方案**：
- 减少背景数据集大小
- 使用CPU模式（移除`.to(device)`）
- 增加GPU显存

---

## 📚 参考文献

### SHAP方法
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS*.

### Cox回归
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society*.

### FDR校正
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society*.

---

## 📝 更新日志

### 版本历史
- **v1.0**: 初始版本，包含SHAP和Cox分析
- **v1.1**: 添加特征重要性排序保存
- **v1.2**: 改进相关性图，添加拟合直线和统计信息
- **v1.3**: 合并训练集和验证集用于Cox分析，提高统计功效
- **v1.4**: 兼容不同版本的lifelines库

---

## 💡 最佳实践

1. **首次运行**：先在单个癌症上测试，确认结果符合预期
2. **批量处理**：使用脚本循环处理多个癌症类型
3. **结果验证**：检查相关性图，确保数据质量
4. **深入分析**：使用pickle文件进行自定义分析
5. **生物学验证**：将发现的基因与文献对比验证

---

## 📧 联系与支持

如有问题或建议，请联系开发团队或提交Issue。

---

**文档版本**: 1.0  
**最后更新**: 2025-10-14  
**适用代码版本**: shap_individual_analysis.py v1.4

