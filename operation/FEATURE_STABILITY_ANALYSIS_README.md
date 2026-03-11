# 特征稳定性分析脚本说明文档

## 📚 目录
- [概述](#概述)
- [核心功能](#核心功能)
- [稳定性指标](#稳定性指标)
- [代码结构](#代码结构)
- [使用方法](#使用方法)
- [配置参数](#配置参数)
- [输出文件](#输出文件)
- [关键方法详解](#关键方法详解)
- [结果解读指南](#结果解读指南)
- [可视化建议](#可视化建议)
- [注意事项](#注意事项)

---

## 📖 概述

这是一个用于分析生存预测模型中**特征重要性稳定性**的数据生成工具。主要目标是评估在同一癌症类型、同一可解释AI（XAI）方法下，跨不同模型训练（不同repeat和fold）时，Top-k重要特征的一致性和稳定性。

### 🎯 主要目的
1. **评估特征选择的稳定性** - 不同模型是否选择相似的重要特征
2. **识别稳定核心特征** - 找出在多数模型中都被认为重要的特征
3. **跨XAI方法一致性分析** - 不同XAI方法是否识别出相同的重要特征
4. **性能关联分析** - 特征稳定性与模型性能（C-index）的关系
5. **生成可视化数据** - 保存所有原始数据供后续自定义可视化

### 🔬 分析场景
- **10次重复 × 5折交叉验证** = 50个模型
- **6种XAI方法**: SHAP, IG, LRP, PFI, DeepSHAP, DeepLIFT
- **15种癌症类型**: BLCA, BRCA, COADREAD, GBMLGG, HNSC, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, PAAD, SKCM, STAD, UCEC
- **多个Top-k设置**: 20, 50, 100, 200, 300
- **多个稳定性阈值**: 70%, 75%, 80%, 85%, 90%

### ⚠️ 重要说明
**本脚本专注于数据生成，不包含绘图功能。**所有可视化需要使用生成的CSV数据文件自行完成。

---

## 🚀 核心功能

### 1. **成对稳定性分析**
对50个模型的特征排名进行两两比较（1225对），计算多种稳定性指标并保存原始数据。

### 2. **特征频次统计**
统计每个特征在Top-k中出现的频次，识别稳定核心特征。

### 3. **分层分析**
- **同fold内稳定性**: 不同repeat、相同fold的模型间稳定性
- **跨fold稳定性**: 不同fold间的稳定性

### 4. **随机基线对比**
通过随机模拟计算期望的稳定性水平，评估实际稳定性是否显著高于随机。

### 5. **跨XAI方法一致性**
比较不同XAI方法识别的稳定特征的重叠程度，识别共识特征。

### 6. **性能关联分析**
分析特征稳定性与模型预测性能（C-index）的相关性。

### 7. **多阈值分析**
同时分析5个稳定性阈值（70%, 75%, 80%, 85%, 90%），提供不同严格程度的稳定特征集。

---

## 📊 稳定性指标

### 1. **Jaccard相似度**
```
Jaccard = |A ∩ B| / |A ∪ B|
```
- **范围**: [0, 1]
- **含义**: 两个Top-k特征集的重叠程度
- **解读**: 值越高表示稳定性越好

| 范围 | 稳定性等级 | 含义说明 |
|------|-----------|---------|
| **0.8-1.0** | 🟢 极高稳定性 | 重叠度≥80%，特征选择非常稳定 |
| **0.6-0.8** | 🟢 高稳定性 | 重叠度60-80%，可接受的稳定性 |
| **0.4-0.6** | 🟡 中等稳定性 | 重叠度40-60%，需要注意可靠性 |
| **0.2-0.4** | 🟠 低稳定性 | 重叠度20-40%，一致性较差 |
| **0-0.2** | 🔴 极低稳定性 | 重叠度<20%，接近随机水平 |

---

### 2. **Kuncheva指数** ⭐ 推荐
```
Kuncheva = (r - k²/m) / (k - k²/m)
```
- **范围**: [-1, 1]
- **含义**: 经过机会校正的Jaccard相似度
- **优势**: 消除随机选择的影响

| 范围 | 稳定性等级 | 含义说明 |
|------|-----------|---------|
| **0.5-1.0** | 🟢 显著超越随机 | 特征选择一致性远高于随机期望 |
| **0.2-0.5** | 🟢 适度超越随机 | 中等水平的稳定性 |
| **-0.1-0.2** | 🟡 接近随机期望 | 稳定性不佳，需要改进 |
| **< -0.1** | 🔴 低于随机期望 | 存在系统性问题 |

---

### 3. **RBO (Rank-Biased Overlap)**
```
RBO = (1-p) × Σ(p^(d-1) × overlap_d)
```
- **范围**: [0, 1]
- **含义**: 考虑排名顺序的相似度
- **参数**: p=0.9，重视前面的排名
- **优势**: 同时考虑集合重叠和排名顺序

---

### 4. **Spearman秩相关**
- **范围**: [-1, 1]
- **含义**: 两个排名的单调关系强度
- **优势**: 对异常值稳健

---

### 5. **特征频次**
- **范围**: [0, 1]
- **含义**: 特征在所有模型的Top-k中出现的比例
- **阈值**: 支持多个阈值（70%, 75%, 80%, 85%, 90%）

---

### 6. **跨XAI一致性**
- **范围**: [0, 1]
- **含义**: 不同XAI方法识别的稳定特征的重叠程度
- **共识特征**: 所有XAI方法都认为重要的特征

---

## 🏗️ 代码结构

### 主类: `FeatureStabilityAnalyzer`

```
FeatureStabilityAnalyzer
│
├── 初始化与配置
│   ├── __init__()
│   ├── _get_default_threshold()
│   └── _get_default_thresholds()
│
├── 路径管理
│   ├── _ensure_dir()
│   ├── _build_xai_results_path()
│   └── _build_feature_importance_path()
│
├── 数据加载
│   ├── load_feature_rankings()
│   └── load_model_performance_data()
│
├── 稳定性计算
│   ├── calculate_jaccard_similarity()
│   ├── calculate_kuncheva_index()
│   ├── calculate_rbo()
│   ├── calculate_rank_correlation()
│   ├── analyze_pairwise_stability()
│   └── calculate_feature_frequency()
│
├── 统计分析
│   ├── _calculate_stats_summary()
│   ├── create_stability_summary()
│   ├── create_stratified_analysis()
│   └── calculate_random_baseline()
│
├── 高级分析
│   ├── analyze_stability_performance_correlation()
│   ├── analyze_cross_xai_consistency()
│   ├── calculate_overlap_curves()  # 只计算数据，不绘图
│   └── _analyze_single_topk_stability()
│
├── 数据保存
│   ├── _save_pairwise_raw_data()
│   ├── save_cross_xai_results()
│   └── create_summary_report()
│
├── 报告生成
│   ├── create_cross_xai_summary_report()  # TXT报告（保留）
│   └── create_cross_cancer_summary()
│
└── 主流程
    ├── analyze_cancer_xai_combination()
    └── run_full_analysis()
```

**已删除的功能**:
- ❌ 所有绘图函数（plot_*）
- ❌ 大部分文本报告生成函数
- ✅ 保留跨XAI一致性TXT报告

---

## 🔧 使用方法

### 基本使用

```python
from feature_stability_analysis import FeatureStabilityAnalyzer

# 1. 初始化分析器
analyzer = FeatureStabilityAnalyzer(
    results_dir='/home/zuoyiyi/SNN/TCGA',
    output_dir='/home/zuoyiyi/SNN/TCGA/stability_analysis_enhanced',
    xai_methods=['shap', 'ig', 'lrp', 'pfi', 'deepshap', 'deeplift']
)

# 2. 运行完整分析
cancer_types = ['BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC',
                'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC']
analyzer.run_full_analysis(cancer_types=cancer_types)
```

### 命令行使用

```bash
cd /home/zuoyiyi/SNN/TCGA/operation
python feature_stability_analysis.py \
    --results_dir /home/zuoyiyi/SNN/TCGA \
    --output_dir /home/zuoyiyi/SNN/TCGA/stability_analysis_enhanced
```

**注意**: 脚本中已硬编码XAI方法和癌症类型列表，命令行参数只需指定目录。

---

## ⚙️ 配置参数

### 可配置参数列表

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `top_k_values` | [20, 50, 100, 200, 300] | 要分析的Top-k值列表 |
| `total_features` | 2000 | 总特征数 |
| `stability_thresholds` | [0.7, 0.75, 0.8, 0.85, 0.9] | 稳定特征阈值列表 |
| `stability_threshold` | [0.75, 0.7] | 默认稳定特征阈值 |
| `random_simulations` | 1000 | 随机基线模拟次数 |
| `min_rankings_required` | 3 | 最少需要的排名文件数 |
| `n_repeats` | 10 | 重复实验次数 |
| `n_folds` | 5 | 交叉验证折数 |

---

## 📁 输出文件

### 目录结构

```
stability_analysis_enhanced/
│
├── cross_cancer_stability_summary.csv          # 跨癌症汇总表
├── aggregated_by_xai_method.csv                # 按XAI方法聚合
├── cross_xai_consistency_report.txt            # 跨XAI一致性报告（TXT）
│
├── [CANCER_TYPE]/
│   │
│   ├── [XAI_METHOD]/
│   │   ├── stability_summary_report.csv              # 稳定性汇总（含多阈值）
│   │   ├── pairwise_stability_raw_top20.csv          # 成对比较原始数据
│   │   ├── pairwise_stability_raw_top50.csv
│   │   ├── pairwise_stability_raw_top100.csv
│   │   ├── pairwise_stability_raw_top200.csv
│   │   ├── pairwise_stability_raw_top300.csv
│   │   ├── jaccard_curve_data.csv                    # Jaccard曲线数据
│   │   ├── kuncheva_curve_data.csv                   # Kuncheva曲线数据
│   │   ├── rbo_curve_data.csv                        # RBO曲线数据
│   │   ├── performance_correlations.csv              # 性能相关性统计
│   │   ├── stable_features_top20.csv                 # 所有稳定特征列表
│   │   ├── stable_features_top50.csv
│   │   ├── stable_features_top100.csv
│   │   ├── stable_features_top200.csv
│   │   ├── stable_features_top300.csv
│   │   ├── stable_core_top100_threshold70.csv        # 阈值≥70%
│   │   ├── stable_core_top100_threshold75.csv        # 阈值≥75%
│   │   ├── stable_core_top100_threshold80.csv        # 阈值≥80%
│   │   ├── stable_core_top100_threshold85.csv        # 阈值≥85%
│   │   └── stable_core_top100_threshold90.csv        # 阈值≥90%
│   │
│   └── cross_xai_analysis/
│       ├── cross_xai_consistency_summary.csv         # 一致性汇总
│       ├── consensus_features_top100_thresh70.csv    # 共识特征（阈值70%）
│       ├── consensus_features_top100_thresh75.csv    # 共识特征（阈值75%）
│       ├── consensus_features_top100_thresh80.csv    # 共识特征（阈值80%）
│       ├── features_by_consensus_top100_thresh70.csv # 按一致性排序（阈值70%）
│       ├── features_by_consensus_top100_thresh75.csv # 按一致性排序（阈值75%）
│       └── features_by_consensus_top100.csv          # 按一致性排序（兼容旧版，默认阈值）
```

### 主要输出文件说明

#### 1. **pairwise_stability_raw_top{k}.csv** - 成对比较原始数据

**用途**: 包含所有1225对模型比较的原始稳定性数据

**列名及含义**:
- `repeat1`, `fold1`: 第一个模型的repeat和fold
- `repeat2`, `fold2`: 第二个模型的repeat和fold
- `pair_type`: 比较类型（`same_fold`或`cross_fold`）
- `jaccard`: Jaccard相似度
- `kuncheva`: Kuncheva指数
- `rbo_09`: RBO (p=0.9)
- `spearman_r`: Spearman秩相关系数
- `spearman_p`: Spearman p值

**可视化用途**:
- 绘制稳定性分布直方图
- 比较同fold vs 跨fold稳定性
- 分析稳定性的变异性

---

#### 2. **曲线数据文件** (jaccard/kuncheva/rbo_curve_data.csv)

**用途**: Overlap@k曲线数据，展示稳定性如何随Top-k值变化

**列名及含义**:
- `k`: Top-k值（20-300，步长20）
- `mean`: 平均稳定性
- `std`: 标准差
- `ci_lower`: 95%置信区间下界
- `ci_upper`: 95%置信区间上界

**可视化用途**:
- 绘制稳定性随k值变化的曲线图
- 添加置信区间阴影
- 比较不同XAI方法的曲线

---

#### 3. **stable_core_top{k}_threshold{XX}.csv** - 多阈值稳定核心特征

**用途**: 不同阈值下的稳定核心特征列表

**列名及含义**:
- `rank`: 频次排名
- `feature_name`: 特征名称
- `frequency`: 出现频次

**可视化用途**:
- **补充图1**: 不同阈值15种癌症的稳定特征数目变化
- **补充图2**: 选定阈值展示不同XAI对15种癌症的稳定特征数目

---

#### 4. **consensus_features_top{k}_thresh{threshold}.csv** - 共识特征

**用途**: 所有6种XAI方法都认为重要的共识特征

**列名及含义**:
- `consensus_feature`: 共识特征名称

**应用场景**:
- **数据库检索**: 提取特征名用于5个数据库检索
- **生物标志物候选**: 最高置信度的候选特征
- **实验验证**: 优先选择这些特征进行验证

---

#### 5. **features_by_consensus_top{k}_thresh{threshold}.csv** - 按一致性排序的特征

**用途**: 列出所有被至少两个XAI方法选中的特征，并详细说明是哪些方法。现在为每个阈值生成独立文件。

**列名及含义**:
- `feature`: 特征名称
- `n_methods`: 出现的XAI方法数（1-6）
- `selected_by_methods`: 选中该特征的具体方法列表（如 "IG, LRP, shap"）

**可视化用途**:
- 绘制Upset图展示XAI方法间的重叠
- 绘制条形图展示特征的一致性程度

---

#### 6. **cross_cancer_stability_summary.csv** - 跨癌症汇总表

**用途**: 汇总所有癌症×XAI×Top-k的稳定性指标

**主要列**:
- `cancer_type`, `xai_method`, `top_k`
- `jaccard_mean`, `jaccard_std`
- `kuncheva_mean`, `kuncheva_std`
- `rbo_09_mean`, `rbo_09_std`
- `n_stable_core_70pct`, `n_stable_core_75pct`, `n_stable_core_80pct`, `n_stable_core_85pct`, `n_stable_core_90pct`

**可视化用途**:
- **主图1**: 总体评估不同XAI的Kuncheva
- **主图2**: 总体评估三类XAI的Kuncheva（需手动添加XAI分类）
- **主图6**: 分癌症比较XAI稳定性
- **补充图**: 不同阈值/XAI的稳定特征数目

---

#### 7. **performance_correlations.csv** - 性能相关性

**列名及含义**:
- `top_k`: Top-k值
- `n_experiments`: 实验数量
- `pearson_r`, `pearson_p`: Pearson相关系数和p值
- `spearman_r`, `spearman_p`: Spearman相关系数和p值

**可视化用途**:
- **主图7**: 稳定性与性能相关性散点图

---

## 🔍 关键方法详解

### 数据加载

#### `load_feature_rankings(cancer_type, xai_method)`
加载指定癌症和XAI方法的所有特征排名文件（50个）。

### 稳定性计算

#### `analyze_pairwise_stability(rankings, k)`
计算所有排名对之间的稳定性指标（1225对）。

#### `calculate_overlap_curves(cancer_type, xai_method, all_rankings)`
计算Overlap@k曲线数据（不绘图），返回Jaccard、Kuncheva、RBO的DataFrame。

### 高级分析

#### `analyze_cross_xai_consistency(cancer_type, all_xai_results, k, threshold)`
分析跨XAI方法的一致性，识别共识特征。

**核心操作**:
1. 提取各XAI方法的稳定特征（频次≥阈值）
2. 计算成对Jaccard相似度（15对）
3. 找出所有XAI都选择的共识特征（交集）
4. 统计特征被选择的频次

---

## 📊 结果解读指南

### 快速判断流程

1. **查看Kuncheva均值**
   - \> 0.3: 显著稳定 ✅
   - 0-0.3: 中等稳定 ⚠️
   - < 0: 存在问题 ❌

2. **查看Jaccard均值**
   - \> 0.6: 高稳定性 ✅
   - 0.4-0.6: 中等稳定性 ⚠️
   - < 0.4: 低稳定性 ❌

3. **查看稳定核心特征数**
   - 阈值70%: 相对稳定的特征集
   - 阈值80%: 高度稳定的核心特征
   - 阈值90%: 极其稳定的特征（可能很少）

4. **查看跨XAI一致性**
   - 共识特征数 > 10: 高一致性 ✅
   - 共识特征数 5-10: 中等一致性 ⚠️
   - 共识特征数 < 5: 低一致性 ❌

---

## 🎨 可视化建议

### 主要可视化（7个）

#### 1. 总体评估不同XAI的Kuncheva
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cross_cancer_stability_summary.csv')
df_top100 = df[df['top_k'] == 100]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_top100, x='xai_method', y='kuncheva_mean')
plt.ylabel('Kuncheva Index')
plt.title('XAI Method Stability Comparison (Top-100)')
plt.savefig('xai_kuncheva_comparison.png', dpi=300)
```

#### 2. 总体评估三类XAI的Kuncheva
```python
# 手动添加XAI分类
xai_categories = {
    'shap': 'Propagation', 'deepshap': 'Propagation', 'lrp': 'Propagation',
    'ig': 'Gradient', 'deeplift': 'Gradient',
    'pfi': 'Perturbation'
}
df_top100['xai_category'] = df_top100['xai_method'].map(xai_categories)

sns.boxplot(data=df_top100, x='xai_category', y='kuncheva_mean')
plt.title('XAI Category Stability Comparison')
```

#### 3. 稳定性随k值变化
```python
jaccard_df = pd.read_csv('LGG/shap/jaccard_curve_data.csv')

plt.figure(figsize=(8, 6))
plt.plot(jaccard_df['k'], jaccard_df['mean'], marker='o')
plt.fill_between(jaccard_df['k'], jaccard_df['ci_lower'], jaccard_df['ci_upper'], alpha=0.3)
plt.xlabel('Top-k')
plt.ylabel('Jaccard Similarity')
plt.title('Stability vs Top-k (LGG, SHAP)')
```

#### 4. 排名稳定性随k值变化
```python
rbo_df = pd.read_csv('LGG/shap/rbo_curve_data.csv')

plt.plot(rbo_df['k'], rbo_df['mean'], marker='o')
plt.fill_between(rbo_df['k'], rbo_df['ci_lower'], rbo_df['ci_upper'], alpha=0.3)
plt.xlabel('Top-k')
plt.ylabel('RBO (p=0.9)')
plt.title('Ranking Stability vs Top-k')
```

#### 5. 折内外稳定性比较
```python
raw_df = pd.read_csv('LGG/shap/pairwise_stability_raw_top100.csv')

sns.boxplot(data=raw_df, x='pair_type', y='jaccard')
plt.ylabel('Jaccard Similarity')
plt.title('Same-Fold vs Cross-Fold Stability')
```

#### 6. 分癌症比较XAI稳定性
```python
df_top100 = df[df['top_k'] == 100]
pivot = df_top100.pivot(index='cancer_type', columns='xai_method', values='kuncheva_mean')

sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('XAI Stability by Cancer Type')
```

#### 7. 稳定性与性能相关性
```python
perf_df = pd.read_csv('LGG/shap/performance_correlations.csv')

# 需要读取原始数据绘制散点图
# 或使用相关系数绘制条形图
plt.bar(perf_df['top_k'], perf_df['pearson_r'])
plt.xlabel('Top-k')
plt.ylabel('Pearson r')
plt.title('Stability-Performance Correlation')
```

### 补充图

#### 补充图1: 不同阈值15种癌症的稳定特征数目
```python
df = pd.read_csv('cross_cancer_stability_summary.csv')
df_shap = df[(df['xai_method'] == 'shap') & (df['top_k'] == 100)]

thresholds = [70, 75, 80, 85, 90]
for cancer in df_shap['cancer_type'].unique():
    cancer_data = df_shap[df_shap['cancer_type'] == cancer]
    values = [cancer_data[f'n_stable_core_{t}pct'].values[0] for t in thresholds]
    plt.plot(thresholds, values, marker='o', label=cancer)

plt.xlabel('Stability Threshold (%)')
plt.ylabel('Number of Stable Features')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Stable Features Across Thresholds (SHAP, Top-100)')
```

#### 补充图2: 选定阈值展示不同XAI对15种癌症的稳定特征数目
```python
df_top100 = df[df['top_k'] == 100]
pivot = df_top100.pivot(index='cancer_type', columns='xai_method', values='n_stable_core_70pct')

sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Stable Features by XAI Method (Threshold ≥70%, Top-100)')
```

#### 补充图3: 6种XAI的共同稳定特征（代表性癌症）
```python
# 读取共识特征 (使用70%阈值文件)
consensus_df = pd.read_csv('LGG/cross_xai_analysis/consensus_features_top100_thresh70.csv')
features_by_consensus = pd.read_csv('LGG/cross_xai_analysis/features_by_consensus_top100_thresh70.csv')

# 绘制条形图
top20 = features_by_consensus.head(20)
plt.figure(figsize=(12, 6))
sns.barplot(data=top20, x='feature', y='n_methods')
plt.xticks(rotation=90)
plt.ylabel('Number of XAI Methods')
plt.title('Top 20 Features by XAI Consensus (LGG, Top-100, Threshold ≥70%)')
```

---

## 📝 数据库检索流程

### 提取共识特征用于检索

```python
import pandas as pd

# 选择代表性癌症（如LGG）
consensus_df = pd.read_csv('LGG/cross_xai_analysis/consensus_features_top100_thresh70.csv')

# 提取特征名（去掉_rnaseq后缀）
features = consensus_df['consensus_feature'].str.replace('_rnaseq', '').tolist()

print(f"✅ 6种XAI共同选择的稳定特征: {len(features)}个")
print(f"特征列表: {features}")

# 在5个数据库中检索
databases = ['KEGG', 'GO', 'Reactome', 'WikiPathways', 'MSigDB']
for db in databases:
    print(f"\n🔍 在{db}数据库中检索...")
    for feature in features:
        # 调用数据库API或手动检索
        search_in_database(db, feature)
```

---

## ⚠️ 注意事项

### 1. 数据要求
- 需要至少3个排名文件才能进行分析
- 标准配置: 10 repeats × 5 folds = 50个文件
- 文件命名格式: `repeat{i}_fold{j}_feature_importance.csv`

### 2. 内存使用
- 大规模分析（15癌症×6XAI）可能需要较大内存
- 建议至少16GB RAM
- 可以分批次运行（指定cancer_types子集）

### 3. 运行时间
- 单个癌症-XAI组合: 约2-5分钟
- 完整分析（15癌症×6XAI）: 约3-6小时
- 取决于CPU性能和数据大小

### 4. 文件路径
- 所有路径必须使用绝对路径
- 确保results_dir包含正确的子目录结构
- 输出目录会自动创建

### 5. XAI方法名称
- 脚本中硬编码: `['shap', 'IG', 'LRP', 'PFI', 'deepshap', 'DeepLIFT']`
- 注意大小写（IG, LRP, DeepLIFT大写）
- 如需修改，编辑main()函数中的xai_methods列表

### 6. 可视化
- **本脚本不生成任何图片**
- 所有可视化需要使用生成的CSV文件自行完成
- 参考"可视化建议"章节的示例代码

### 7. 数据完整性检查
运行后检查以下文件是否生成:
```bash
# 检查主要输出
ls stability_analysis_enhanced/cross_cancer_stability_summary.csv
ls stability_analysis_enhanced/aggregated_by_xai_method.csv

# 检查单个癌症-XAI的输出
ls stability_analysis_enhanced/LGG/shap/*.csv

# 检查跨XAI分析输出
ls stability_analysis_enhanced/LGG/cross_xai_analysis/*.csv
```

---

## 📞 技术支持

如有问题，请检查:
1. 文件路径是否正确
2. 数据文件格式是否符合要求
3. Python环境是否安装所需包（pandas, numpy, scipy）
4. 内存是否充足

---

**版本**: 2.0 (数据生成专用版)  
**最后更新**: 2024-12-09  
**作者**: AI助手
