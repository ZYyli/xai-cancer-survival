# 特征稳定性分析 - 工作流程图

## 🔄 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    特征稳定性分析系统                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. 初始化 FeatureStabilityAnalyzer   │
        │     - 设置路径                         │
        │     - 配置参数                         │
        │     - XAI方法列表                      │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  2. 运行完整分析 run_full_analysis()  │
        │     - 15个癌症类型                     │
        │     - 4个XAI方法                       │
        │     - 共60个分析任务                    │
        └───────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  单癌症-XAI分析  │       │  跨方法一致性    │
    │  (60次循环)      │       │  分析            │
    └─────────────────┘       └─────────────────┘
                │                       │
                ▼                       ▼
        ┌───────────┐           ┌───────────┐
        │  结果汇总  │           │  共识特征  │
        └───────────┘           └───────────┘
```

---

## 📊 单癌症-XAI组合分析详细流程

```
analyze_cancer_xai_combination(cancer_type, xai_method)
│
├─► 步骤1: 加载数据
│   ├── load_feature_rankings()
│   │   ├── 读取50个排名文件 (10 repeat × 5 fold)
│   │   ├── 验证必要列存在
│   │   └── 返回: {(repeat, fold): DataFrame}
│   │
│   └── load_model_performance_data()
│       ├── 读取 detailed_results.csv
│       └── 返回: {(repeat, fold): c_index}
│
├─► 步骤2: 数据验证
│   └── 检查是否有足够的排名文件 (≥3个)
│
├─► 步骤3: 多Top-k稳定性分析 (循环4次: k=20,50,100,200)
│   │
│   └── _analyze_single_topk_stability(k)
│       │
│       ├── 3.1 成对稳定性分析
│       │   └── analyze_pairwise_stability()
│       │       ├── 生成1,225个排名对
│       │       ├── 计算Jaccard相似度
│       │       ├── 计算Kuncheva指数
│       │       ├── 计算RBO (p=0.9, 0.98)
│       │       └── 计算秩相关 (Spearman, Kendall)
│       │
│       ├── 3.2 稳定性统计摘要
│       │   └── create_stability_summary()
│       │       ├── 计算均值、标准差
│       │       ├── 计算中位数、四分位数
│       │       └── 返回统计指标字典
│       │
│       ├── 3.3 分层分析
│       │   └── create_stratified_analysis()
│       │       ├── 同fold内稳定性
│       │       └── 跨fold稳定性
│       │
│       ├── 3.4 特征频次统计
│       │   └── calculate_feature_frequency()
│       │       ├── 统计每个特征出现次数
│       │       └── 计算频次比例
│       │
│       ├── 3.5 随机基线计算
│       │   └── calculate_random_baseline()
│       │       ├── 随机模拟1000次
│       │       ├── 计算期望Jaccard
│       │       └── 计算期望Kuncheva
│       │
│       ├── 3.6 识别稳定核心特征
│       │   └── 筛选 frequency >= 0.7 的特征
│       │
│       ├── 3.7 生成可视化
│       │   ├── plot_stability_distribution()
│       │   │   ├── Jaccard分布直方图
│       │   │   ├── Kuncheva分布直方图
│       │   │   ├── RBO分布直方图
│       │   │   └── 分层箱线图
│       │   │
│       │   └── plot_feature_frequency()
│       │       ├── 频次分布直方图
│       │       └── Top-20稳定特征柱状图
│       │
│       └── 3.8 保存结果
│           └── stable_features_top{k}.csv
│
├─► 步骤4: 绘制Overlap@k曲线
│   └── plot_overlap_curves()
│       ├── k从10到200，步长10
│       ├── 绘制Jaccard曲线
│       ├── 绘制Kuncheva曲线
│       └── 保存曲线数据CSV
│
├─► 步骤5: 性能相关性分析 🆕
│   └── analyze_stability_performance_correlation()
│       ├── 计算每个实验的平均稳定性
│       ├── 与C-index计算Pearson相关
│       ├── 与C-index计算Spearman相关
│       └── 保存 performance_correlations.csv
│
└─► 步骤6: 生成报告
    ├── create_summary_report()
    │   └── stability_summary_report.csv
    │
    └── create_text_report()
        └── stability_analysis_report.txt
```

---

## 🔗 跨XAI方法一致性分析流程

```
analyze_cross_xai_consistency(cancer_type, all_xai_results, k)
│
├─► 步骤1: 提取各XAI方法的稳定特征
│   ├── SHAP稳定特征 (frequency >= 0.7)
│   ├── IG稳定特征
│   ├── LRP稳定特征
│   └── PFI稳定特征
│
├─► 步骤2: 成对一致性分析
│   ├── SHAP vs IG    → Jaccard
│   ├── SHAP vs LRP   → Jaccard
│   ├── SHAP vs PFI   → Jaccard
│   ├── IG vs LRP     → Jaccard
│   ├── IG vs PFI     → Jaccard
│   └── LRP vs PFI    → Jaccard
│
├─► 步骤3: 计算共识特征
│   └── 所有XAI方法的交集
│       └── consensus_features
│
├─► 步骤4: 特征方法计数
│   └── 统计每个特征被几个方法选中
│       ├── 4个方法: 最强共识
│       ├── 3个方法: 强共识
│       ├── 2个方法: 中等共识
│       └── 1个方法: 方法特异性
│
└─► 步骤5: 保存结果
    ├── consensus_features_top{k}.csv
    ├── features_by_consensus_top{k}.csv
    └── cross_xai_consistency_summary.csv
```

---

## 🎯 关键数据结构

### 1. Rankings 字典
```python
rankings = {
    (0, 0): DataFrame,  # repeat0_fold0
    (0, 1): DataFrame,  # repeat0_fold1
    ...
    (9, 4): DataFrame   # repeat9_fold4
}

# DataFrame 结构:
#   feature_name | importance_score | rank
#   GENE_A       | 0.85            | 1
#   GENE_B       | 0.73            | 2
#   ...
```

### 2. Pairwise Results 字典
```python
pairwise_results = {
    'jaccard': [0.45, 0.52, ...],      # 1,225个值
    'kuncheva': [0.32, 0.41, ...],
    'rbo_09': [0.48, 0.55, ...],
    'rbo_098': [0.51, 0.58, ...],
    'spearman_r': [0.61, 0.68, ...],
    'kendall_tau': [0.43, 0.51, ...],
    'pair_info': [
        {'key1': (0,0), 'key2': (0,1), 'type': 'cross_fold'},
        ...
    ]
}
```

### 3. Stability Summary 字典
```python
stability_summary = {
    'jaccard_mean': 0.45,
    'jaccard_std': 0.12,
    'jaccard_median': 0.46,
    'jaccard_q25': 0.38,
    'jaccard_q75': 0.53,
    'kuncheva_mean': 0.32,
    ...
}
```

### 4. Stability Results 字典
```python
stability_results = {
    'rankings': {...},  # 原始排名
    20: {  # Top-20结果
        'pairwise_results': {...},
        'stability_summary': {...},
        'stratified_analysis': {...},
        'feature_frequency': {...},
        'stable_core': {...},
        'random_baseline': {...}
    },
    50: {...},   # Top-50结果
    100: {...},  # Top-100结果
    200: {...}   # Top-200结果
}
```

---

## 🧮 计算复杂度分析

```
单个癌症-XAI组合:
├── 加载数据: O(50) - 50个文件
├── 成对比较: O(C(50,2)) = O(1,225)
│   └── 每对计算6个指标
├── 特征频次: O(50 × 2000) = O(100,000)
└── 可视化: O(20 + 50 + 100 + 200) = O(370)

全部分析:
├── 60个癌症-XAI组合
├── 4个Top-k值
└── 总计: 60 × 4 × 1,225 = 294,000 次成对比较

跨XAI分析:
├── 15个癌症
├── 每个癌症: C(4,2) = 6 个XAI方法对
└── 总计: 15 × 6 = 90 个一致性计算
```

---

## 📈 稳定性指标计算示例

### Jaccard 计算流程
```
Set1 = {Gene_A, Gene_B, Gene_C, Gene_D, Gene_E}  # Model 1 Top-5
Set2 = {Gene_A, Gene_C, Gene_F, Gene_G, Gene_H}  # Model 2 Top-5

交集 = {Gene_A, Gene_C}                           # 2个
并集 = {Gene_A, Gene_B, Gene_C, Gene_D, Gene_E,  # 8个
        Gene_F, Gene_G, Gene_H}

Jaccard = 2/8 = 0.25
```

### Kuncheva 计算流程
```
k = 5           # Top-k大小
m = 2000        # 总特征数
r = 2           # 交集大小

expected_overlap = k²/m = 25/2000 = 0.0125

Kuncheva = (r - expected_overlap) / (k - expected_overlap)
         = (2 - 0.0125) / (5 - 0.0125)
         = 1.9875 / 4.9875
         = 0.398
```

### Feature Frequency 计算流程
```
Gene_A在50个模型中:
- Top-100出现次数: 42次
- Frequency = 42/50 = 0.84

Gene_B在50个模型中:
- Top-100出现次数: 15次
- Frequency = 15/50 = 0.30

Gene_A是稳定核心特征 (0.84 >= 0.7) ✅
Gene_B不是稳定核心特征 (0.30 < 0.7) ❌
```

---

## 🔍 性能相关性分析流程 🆕

```
analyze_stability_performance_correlation()
│
├─► 对每个实验 (repeat, fold):
│   │
│   ├── 1. 获取该实验的Top-k特征
│   │   └── top_k_current
│   │
│   ├── 2. 计算与其他49个实验的Jaccard
│   │   └── [0.45, 0.52, 0.48, ...]  # 49个值
│   │
│   ├── 3. 计算平均Jaccard (平均稳定性)
│   │   └── avg_jaccard = 0.47
│   │
│   └── 4. 获取该实验的C-index
│       └── cindex = 0.68
│
├─► 收集所有实验数据:
│   experiment_data = [
│       {'repeat': 0, 'fold': 0, 'cindex': 0.68, 'avg_jaccard': 0.47},
│       {'repeat': 0, 'fold': 1, 'cindex': 0.71, 'avg_jaccard': 0.51},
│       ...
│   ]
│
└─► 计算相关性:
    ├── Pearson相关 (线性关系)
    │   └── r = 0.23, p = 0.04 (显著)
    │
    └── Spearman相关 (单调关系)
        └── ρ = 0.28, p = 0.02 (显著)

解读: 特征稳定性与模型性能正相关
```

---

## 🎨 可视化输出示例

### 1. Overlap Curves
```
    Jaccard
      ^
  0.6 |                    ___-----
      |               ___---
  0.4 |          ___--
      |      _--
  0.2 |   _-
      |_-___________________>
        10  50  100  150  200  Top-k

解读: 随着k增大，稳定性先上升后趋于平稳
```

### 2. Stability Distribution
```
频次
  ^
  |     █
  |    ███
  |   █████
  |  ███████
  | █████████
  |___________> Jaccard
    0.3   0.5

解读: Jaccard呈正态分布，均值约0.45
```

### 3. Feature Frequency
```
    频次
      ^
  1.0 |█        特征稳定性排名
      |█
  0.8 |█ █
      |█ █
  0.6 |█ █ █
      |█ █ █ █
  0.4 |█ █ █ █ █
      |___________>
       1 2 3 4 5...  特征排名

解读: 前几个特征频次接近1.0，非常稳定
```

---

## 💾 文件I/O流程

```
输入文件读取:
├── {xai_method}_results_2/{cancer}/
│   └── {xai}_feature_importance/
│       ├── repeat0_fold0_{xai}_feature_importance_ranking.csv
│       ├── repeat0_fold1_{xai}_feature_importance_ranking.csv
│       └── ... (共50个文件)
│
└── results_2/{cancer}/
    └── detailed_results.csv  # 性能数据

                    ↓

            数据处理与分析

                    ↓

输出文件生成:
├── stability_analysis/
│   ├── {cancer}/{xai}/
│   │   ├── *.csv  # 数值结果
│   │   └── *.png  # 可视化图表
│   │
│   ├── {cancer}/cross_xai_analysis/
│   │   └── *.csv  # 跨XAI结果
│   │
│   └── 全局汇总文件
```

---

**提示**: 此文档展示了代码的执行流程和内部工作原理，帮助理解代码逻辑。
