# Bootstrap特征稳定性分析脚本使用说明

## 📋 概述

`feature_stability_analysis_bootstrap.py` 是针对Bootstrap数据的特征重要性稳定性分析脚本，与原有的`feature_stability_analysis.py`（针对10×5折交叉验证）保持相同的分析流程，但适配了Bootstrap数据格式。

## 🔄 与原脚本的主要区别

### 1. **数据来源**
- **原脚本**: 10次重复 × 5折交叉验证 = 50个模型
  - 数据路径: `{xai}_results_2/[cancer]/{xai}_feature_importance/`
  - 文件格式: `repeat{X}_fold{Y}_{xai}_feature_importance_ranking.csv`

- **Bootstrap脚本**: 100次或500次Bootstrap迭代
  - 数据路径: `{xai}_bootstrap_results/[cancer]/{xai}_feature_importance/`
  - 文件格式: `seed{X}_{xai}_ranking.csv`

### 2. **配置参数**
- **原脚本**: `n_repeats=10`, `n_folds=5`
- **Bootstrap脚本**: `num_bootstraps=100` (可配置为100或500)

### 3. **性能数据加载**
- **原脚本**: 从 `results_2/[cancer]/detailed_results.csv` 读取C-index
- **Bootstrap脚本**: 从 `results_bootstrap/[cancer]/cindex_array.npy` 读取C-index数组

### 4. **分层分析**
- **原脚本**: 分析同fold内 vs 跨fold的稳定性差异
- **Bootstrap脚本**: 不进行分层分析（所有bootstrap种子平等对待）

## 🚀 使用方法

### 基本用法

```bash
python operation/feature_stability_analysis_bootstrap.py \
    --results_dir /home/zuoyiyi/SNN/TCGA \
    --output_dir /home/zuoyiyi/SNN/TCGA/stability_analysis_bootstrap \
    --num_bootstraps 100
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--results_dir` | 结果根目录（包含`{xai}_bootstrap_results/`子目录） | `/home/zuoyiyi/SNN/TCGA` |
| `--output_dir` | 稳定性分析结果输出目录 | `/home/zuoyiyi/SNN/TCGA/stability_analysis_bootstrap` |
| `--num_bootstraps` | Bootstrap迭代次数（100或500） | `100` |

### 示例

#### 分析100次Bootstrap数据
```bash
python operation/feature_stability_analysis_bootstrap.py \
    --num_bootstraps 100 \
    --output_dir /home/zuoyiyi/SNN/TCGA/stability_bootstrap_100
```

#### 分析500次Bootstrap数据
```bash
python operation/feature_stability_analysis_bootstrap.py \
    --num_bootstraps 500 \
    --output_dir /home/zuoyiyi/SNN/TCGA/stability_bootstrap_500
```

## 📊 分析内容

### 1. **稳定性指标**（与原脚本相同）
- **Jaccard相似度**: 衡量Top-k特征集合的重合度
- **Kuncheva指数**: 经过机会校正的Jaccard相似度
- **RBO (Rank-Biased Overlap)**: 考虑排名顺序的相似度
- **Spearman相关系数**: 排名相关性

### 2. **Top-k值设置**
- 20, 50, 100, 200, 300

### 3. **稳定性阈值**
- 70%: 在≥70%的bootstrap模型中出现的特征
- 80%: 在≥80%的bootstrap模型中出现的特征

### 4. **随机基线**
- 通过1000次随机模拟计算期望稳定性
- 用于评估实际稳定性是否超越随机水平

### 5. **跨XAI方法一致性分析**
- 比较不同XAI方法（SHAP, IG, LRP, PFI）选择特征的一致性
- 识别共识特征（所有方法都认为重要的特征）

### 6. **稳定性与性能相关性**
- 分析特征选择稳定性与模型C-index的关系
- 判断稳定的特征选择是否对应更好的预测性能

## 📁 输出文件结构

```
stability_analysis_bootstrap/
├── cross_cancer_summary_report.txt          # 跨癌症汇总文本报告
├── cross_cancer_stability_summary.csv       # 跨癌症汇总数据表
├── cross_cancer_stability_comparison_top100.png  # Top-100跨癌症对比图
├── cross_cancer_stability_comparison_top200.png  # Top-200跨癌症对比图
├── cross_cancer_stability_comparison_top300.png  # Top-300跨癌症对比图
├── cross_xai_consistency_report.txt         # 跨XAI一致性报告
│
└── [CANCER]/                                # 每个癌症的详细结果
    ├── [XAI_METHOD]/                        # 每个XAI方法的详细结果
    │   ├── stability_summary_report.csv     # 数值统计汇总（含多阈值）
    │   ├── stability_analysis_report.txt    # 文本格式详细报告
    │   ├── performance_correlations.csv     # 稳定性与C-index相关性统计
    │   ├── stability_performance_correlation.png  # 稳定性与C-index散点图
    │   ├── overlap_curves.png               # Jaccard和Kuncheva曲线
    │   ├── jaccard_curve_data.csv           # Jaccard曲线数据
    │   ├── kuncheva_curve_data.csv          # Kuncheva曲线数据
    │   ├── stability_distribution_top*.png  # 各Top-k的稳定性分布图
    │   ├── feature_frequency_top*.png       # 各Top-k的特征频次图
    │   ├── stable_features_top*.csv         # 所有稳定特征列表（按频次排序）
    │   ├── stable_core_top*_threshold70.csv # 阈值≥70%的稳定核心特征
    │   └── stable_core_top*_threshold80.csv # 阈值≥80%的稳定核心特征
    │
    └── cross_xai_analysis/                  # 跨XAI一致性分析
        ├── cross_xai_consistency_summary.csv
        ├── consensus_features_top*.csv      # 共识特征（所有XAI方法都选中）
        └── features_by_consensus_top*.csv   # 按一致性排序的特征（含选中方法）
```

## 🔍 输出文件说明

### 核心CSV文件

1. **`stability_summary_report.csv`**: 每个Top-k值的稳定性统计
   - 包含Jaccard、Kuncheva、RBO等指标的均值、标准差、中位数、四分位数
   - 包含多个阈值（70%、80%）的稳定核心特征数量
   - 与随机基线的对比

2. **`stable_features_top*.csv`**: 所有稳定特征列表
   - `feature_name`: 特征名称
   - `frequency`: 出现频次（0-1之间）
   - `rank`: 按频次的排名

3. **`stable_core_top*_threshold70.csv`**: 高稳定性特征（≥70%）
   - 仅包含在至少70%的bootstrap模型中都被选为Top-k的特征
   - 这些特征具有更高的稳健性

4. **`stable_core_top*_threshold80.csv`**: 极高稳定性特征（≥80%）
   - 仅包含在至少80%的bootstrap模型中都被选为Top-k的特征
   - 这些特征最稳健，最不受样本变化影响

5. **`performance_correlations.csv`**: 稳定性与性能相关性
   - `pearson_r/p`: Pearson相关系数及p值
   - `spearman_r/p`: Spearman相关系数及p值
   - 评估高稳定性是否对应高C-index

6. **`consensus_features_top*.csv`**: 跨XAI共识特征
   - 所有XAI方法都识别为稳定的特征
   - 这些特征最可靠

7. **`features_by_consensus_top*.csv`**: 按一致性排序的特征
   - `n_methods`: 有多少种XAI方法选中该特征
   - `selected_by_methods`: 具体是哪些方法选中（如："ig, lrp, shap"）

### 可视化图表

1. **`overlap_curves.png`**: 
   - Jaccard相似度和Kuncheva指数随Top-k变化的曲线
   - 带95%置信区间
   - 评估不同Top-k值下的稳定性趋势

2. **`stability_distribution_top*.png`**:
   - 4个子图：Jaccard、Kuncheva、RBO、Spearman的分布直方图
   - 展示稳定性指标的分布特征

3. **`feature_frequency_top*.png`**:
   - 左图：特征出现频次的分布
   - 右图：Top-20高频特征的条形图

4. **`stability_performance_correlation.png`**:
   - 散点图：特征选择稳定性（X轴）vs C-index（Y轴）
   - 包含线性拟合线和相关性统计信息
   - 评估稳定性与预测性能的关系

5. **`cross_cancer_stability_comparison_top*.png`**:
   - 跨癌症对比条形图
   - 3个子图：Jaccard、Kuncheva、稳定核心特征数
   - 便于比较不同癌症的特征稳定性

## 🎯 预期发现

### Bootstrap特点
1. **重抽样机制**:
   - 每个bootstrap样本通过有放回抽样生成
   - 平均包含约63.2%的原始样本
   - Out-of-Bag (OOB)样本用于评估

2. **与交叉验证的区别**:
   - Bootstrap更关注对样本扰动的稳健性
   - 交叉验证更关注对数据分割方式的稳健性
   - Bootstrap可以进行更多次迭代（100-500次）

3. **稳定性评估**:
   - 如果Jaccard > 0.5，说明特征选择非常稳定
   - 如果Jaccard > 0.3，说明特征选择较为稳定
   - Kuncheva > 0，说明超越了随机期望

## ⚙️ 代码内部配置

如需修改配置，编辑脚本中的以下部分：

```python
class BootstrapFeatureStabilityAnalyzer:
    def __init__(self, ...):
        # 可修改的配置参数
        self.top_k_values = [20, 50, 100, 200, 300]  # Top-k值
        self.total_features = 2000                    # 总特征数
        self.stability_thresholds = [0.7, 0.8]        # 稳定性阈值
        self.stability_threshold = 0.7                # 默认阈值
        self.random_simulations = 1000                # 随机基线模拟次数
        self.min_rankings_required = 10               # 最少需要的文件数
```

## 📌 注意事项

1. **数据路径要求**:
   - 确保 `{results_dir}/{xai}_bootstrap_results/{cancer}/{xai}_feature_importance/` 存在
   - 确保文件命名为 `seed{X}_{xai}_ranking.csv`，其中X从1到num_bootstraps

2. **性能数据**（可选）:
   - 如果存在 `{results_dir}/results_bootstrap/{cancer}/cindex_array.npy`
   - 将自动进行稳定性与性能相关性分析

3. **最少文件数**:
   - 至少需要10个有效的ranking文件才会进行分析
   - 这是为了确保统计结果的可靠性

4. **XAI方法**:
   - 默认分析: SHAP, IG, LRP, PFI
   - DeepSHAP可能失败率较高，建议先检查数据完整性

5. **内存使用**:
   - Bootstrap 500次的数据量约为100次的5倍
   - 确保有足够的内存进行成对比较（500次需要约12.5万次比较）

## 🔗 相关脚本

- `feature_stability_analysis.py`: 原始的10×5折交叉验证稳定性分析
- `ig_bootstrap_analysis.py`: IG方法的Bootstrap分析（生成输入数据）
- `lrp_bootstrap_analysis.py`: LRP方法的Bootstrap分析（生成输入数据）
- `pfi_bootstrap_analysis.py`: PFI方法的Bootstrap分析（生成输入数据）
- `shap_bootstrap_analysis.py`: SHAP方法的Bootstrap分析（生成输入数据）
- `deepshap_bootstrap_analysis.py`: DeepSHAP方法的Bootstrap分析（生成输入数据）

## 📧 支持

如有问题，请检查：
1. 数据路径是否正确
2. 文件命名是否符合格式
3. 必要的CSV列（`feature_name`, `importance_score`, `rank`）是否存在
4. 是否有足够的有效文件数




