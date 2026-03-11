# 代码详细注释说明（中文版）

## 📦 类定义

### `FeatureStabilityAnalyzer` - 特征稳定性分析器

这是主分析类，封装了所有稳定性分析的功能。

---

## 🔧 初始化方法

### `__init__(results_dir, output_dir, xai_methods=None)`

**功能**: 初始化分析器，设置路径和配置参数

**参数**:
- `results_dir` (str): 结果文件的根目录路径
- `output_dir` (str): 分析结果的输出目录
- `xai_methods` (list): XAI方法名称列表，如 `['shap', 'IG', 'LRP', 'PFI']`

**配置参数**（类属性）:
```python
self.top_k_values = [20, 50, 100, 200]     # 要分析的Top-k值列表
self.total_features = 2000                  # 数据集的总特征数
self.stability_threshold = 0.7              # 稳定特征的频次阈值
self.random_simulations = 1000              # 随机基线的模拟次数
self.min_rankings_required = 3              # 进行分析所需的最少排名文件数
self.results_subdir = 'results_2'           # 结果子目录名称
```

**示例**:
```python
analyzer = FeatureStabilityAnalyzer(
    results_dir='/home/user/data',
    output_dir='/home/user/output',
    xai_methods=['shap', 'IG']
)
```

---

## 🛠️ 私有辅助方法

### `_build_xai_results_path(xai_method, cancer_type, subpath='')`

**功能**: 统一构建XAI结果文件的路径

**参数**:
- `xai_method` (str): XAI方法名，如 'shap'
- `cancer_type` (str): 癌症类型，如 'BRCA'
- `subpath` (str): 可选的子路径

**返回**: str - 完整的文件路径

**实现逻辑**:
```python
# 构建XAI方法目录: shap_results_2, IG_results_2等
xai_dir = f'{xai_method}_results_2'
# 组合: results_dir/xai_dir/cancer_type/[subpath]
base_path = os.path.join(self.results_dir, xai_dir, cancer_type)
if subpath:
    return os.path.join(base_path, subpath)
return base_path
```

### `_build_feature_importance_path(xai_method, cancer_type)`

**功能**: 构建特征重要性文件的目录路径和文件名模式

**返回**: tuple (目录路径, 文件名模式)

**逻辑**:
- 对于 SHAP: `shap_feature_importance` 目录
- 对于其他: `{method}_feature_importance` 目录（小写）

### `_calculate_stats_summary(values)`

**功能**: 统一计算统计指标（避免重复代码）

**参数**:
- `values` (list): 数值列表

**返回**: dict - 包含以下键的统计指标字典
```python
{
    'mean': 均值,
    'std': 标准差,
    'median': 中位数,
    'q25': 25%分位数,
    'q75': 75%分位数,
    'min': 最小值,
    'max': 最大值
}
```

**处理**: 自动过滤NaN值，如果全是NaN则返回全NaN字典

---

## 📂 数据加载方法

### `load_feature_rankings(cancer_type, xai_method)`

**功能**: 加载指定癌症和XAI方法的所有特征排名文件

**参数**:
- `cancer_type` (str): 癌症类型
- `xai_method` (str): XAI方法

**返回**: dict - `{(repeat, fold): DataFrame}` 格式

**DataFrame结构**:
| 列名 | 类型 | 说明 |
|------|------|------|
| feature_name | str | 特征名称 |
| importance_score | float | 重要性得分 |
| rank | int | 排名（1开始） |

**加载过程**:
1. 构建文件路径
2. 遍历10个repeat × 5个fold = 50个文件
3. 验证必要列存在
4. 按重要性得分排序
5. 重新计算rank列
6. 存入字典

**示例输出**:
```python
{
    (0, 0): DataFrame([['GENE_A', 0.85, 1], ...]),
    (0, 1): DataFrame([['GENE_B', 0.78, 1], ...]),
    ...
}
```

### `load_model_performance_data(cancer_type)`

**功能**: 加载模型性能数据（C-index）

**返回**: dict - `{(repeat, fold): cindex}`

**文件位置**: `results_2/{cancer_type}/detailed_results.csv`

**必要列**: `repeat`, `fold`, `test_cindex`

---

## 📊 稳定性指标计算方法

### `calculate_jaccard_similarity(set1, set2)`

**功能**: 计算两个集合的Jaccard相似度

**公式**: `J(A,B) = |A ∩ B| / |A ∪ B|`

**参数**:
- `set1`, `set2`: 两个特征集合（Python set对象）

**返回**: float [0, 1]

**代码**:
```python
intersection = len(set1.intersection(set2))
union = len(set1.union(set2))
return intersection / union if union > 0 else 0.0
```

### `calculate_kuncheva_index(set1, set2, k, m)`

**功能**: 计算Kuncheva指数（机会校正的Jaccard）

**公式**: `KI = (r - k²/m) / (k - k²/m)`

**参数**:
- `set1`, `set2`: 两个特征集合
- `k`: Top-k大小
- `m`: 总特征数

**返回**: float [-1, 1]

**特殊处理**: 当分母接近0时（`abs(denominator) < 1e-10`），根据交集是否等于k返回1.0或0.0

**意义**:
- `> 0`: 超越随机期望
- `= 0`: 与随机相当
- `< 0`: 低于随机期望

### `calculate_rbo(rank1, rank2, p=0.9, depth=None)`

**功能**: 计算Rank-Biased Overlap（考虑排名顺序的相似度）

**公式**: `RBO = (1-p) × Σ(p^(d-1) × overlap_d)`

**参数**:
- `rank1`, `rank2`: 两个排名列表（特征名称的有序列表）
- `p`: 耐心参数，默认0.9（越大越重视前面的排名）
- `depth`: 计算深度，默认为两个列表的最小长度

**返回**: float [0, 1]

**实现逻辑**:
```python
for d in range(1, depth + 1):
    # 取前d个特征的集合
    set1 = set(rank1[:d])
    set2 = set(rank2[:d])
    # 计算overlap
    overlap_d = len(set1.intersection(set2)) / d
    # 加权累加
    overlap += (p ** (d-1)) * overlap_d
return (1 - p) * overlap
```

### `calculate_rank_correlation(rank1, rank2)`

**功能**: 计算两个排名的秩相关（Spearman和Kendall）

**返回**: tuple (spearman_r, spearman_p, kendall_tau, kendall_p)

**步骤**:
1. 找到两个排名的共同特征
2. 提取这些特征在两个排名中的位置
3. 使用scipy.stats计算相关性

**注意**: 如果共同特征少于2个，返回全NaN

---

## 🔬 高级分析方法

### `analyze_pairwise_stability(rankings, k)`

**功能**: 对所有排名对进行两两比较，计算稳定性指标

**参数**:
- `rankings`: 排名字典 `{(repeat, fold): DataFrame}`
- `k`: Top-k值

**返回**: dict - 包含所有指标列表
```python
{
    'jaccard': [0.45, 0.52, ...],      # 1,225个值
    'kuncheva': [0.32, 0.41, ...],
    'rbo_09': [0.48, 0.55, ...],
    'rbo_098': [0.51, 0.58, ...],
    'spearman_r': [0.61, 0.68, ...],
    'spearman_p': [...],
    'kendall_tau': [0.43, 0.51, ...],
    'kendall_p': [...],
    'pair_info': [
        {'key1': (0,0), 'key2': (0,1), 'type': 'cross_fold'},
        ...
    ]
}
```

**计算量**: C(50,2) = 1,225个排名对

**pair_info说明**:
- `type='same_fold'`: 同fold不同repeat
- `type='cross_fold'`: 不同fold

### `calculate_feature_frequency(rankings, k)`

**功能**: 统计每个特征在Top-k中出现的频次

**返回**: dict - `{feature_name: frequency}`

**频次计算**:
```python
frequency = 出现次数 / 总排名数
# 例如: 某特征在50个模型中42次是Top-100
# frequency = 42/50 = 0.84
```

### `calculate_random_baseline(k, m, n_simulations=None)`

**功能**: 通过随机模拟计算稳定性的随机基线

**参数**:
- `k`: Top-k值
- `m`: 总特征数
- `n_simulations`: 模拟次数（默认使用配置值）

**返回**: dict
```python
{
    'jaccard_mean': 随机Jaccard的期望值,
    'jaccard_std': 随机Jaccard的标准差,
    'kuncheva_mean': 随机Kuncheva的期望值,
    'kuncheva_std': 随机Kuncheva的标准差
}
```

**模拟过程**:
1. 随机选择两个Top-k集合
2. 计算Jaccard和Kuncheva
3. 重复n次
4. 计算均值和标准差

**用途**: 判断实际稳定性是否显著高于随机期望

---

## 📈 统计分析方法

### `create_stability_summary(pairwise_results)`

**功能**: 创建稳定性指标的统计摘要

**使用**: 调用 `_calculate_stats_summary()` 统一计算

**对每个指标计算**: mean, std, median, q25, q75, min, max

### `create_stratified_analysis(pairwise_results)`

**功能**: 分层分析稳定性（同fold vs 跨fold）

**返回**: dict
```python
{
    'jaccard_same_fold': {'mean': 0.52, 'std': 0.08, 'n': 200},
    'jaccard_cross_fold': {'mean': 0.45, 'std': 0.12, 'n': 1025},
    'kuncheva_same_fold': {...},
    'kuncheva_cross_fold': {...},
    'rbo_09_same_fold': {...},
    'rbo_09_cross_fold': {...}
}
```

**意义**: 同fold内稳定性通常高于跨fold（因为训练数据有重叠）

---

## 🆕 新增功能方法

### `analyze_stability_performance_correlation(cancer_type, xai_method, stability_results, performance_data)`

**功能**: 分析特征稳定性与模型性能的相关性

**思路**:
1. 对每个实验，计算其Top-k特征与其他实验的平均Jaccard相似度（代表该实验的特征稳定性）
2. 获取该实验的C-index（模型性能）
3. 计算稳定性与性能的相关性（Pearson和Spearman）

**返回**: dict - 每个Top-k的相关性结果
```python
{
    100: {
        'n_experiments': 50,
        'pearson_r': 0.23,
        'pearson_p': 0.04,
        'spearman_r': 0.28,
        'spearman_p': 0.02,
        'experiment_data': [...]
    }
}
```

### `analyze_cross_xai_consistency(cancer_type, all_xai_results, k)`

**功能**: 分析不同XAI方法识别的稳定特征的一致性

**步骤**:
1. 提取各XAI方法的稳定特征（频次≥阈值）
2. 计算成对Jaccard相似度
3. 计算所有方法的交集（共识特征）
4. 统计每个特征被多少个方法选中

**返回**: dict
```python
{
    'methods_analyzed': ['shap', 'IG', 'LRP', 'PFI'],
    'n_methods': 4,
    'pairwise_jaccard': {
        'shap_vs_IG': 0.35,
        'shap_vs_LRP': 0.28,
        ...
    },
    'consensus_features': ['GENE_A', 'GENE_C', ...],
    'n_consensus': 15,
    'overall_jaccard': 0.22,
    'feature_method_counts': {
        'GENE_A': 4,  # 4个方法都选中
        'GENE_B': 3,  # 3个方法选中
        ...
    }
}
```

---

## 🎨 可视化方法

### `plot_overlap_curves(cancer_type, xai_method, all_rankings)`

**功能**: 绘制Overlap@k曲线（稳定性随Top-k变化）

**过程**:
1. k从10到200，步长10
2. 对每个k计算Jaccard和Kuncheva
3. 绘制曲线图，添加置信区间

**输出**: 
- `overlap_curves.png` - 两个子图（Jaccard和Kuncheva）
- `jaccard_curve_data.csv` - 曲线数据
- `kuncheva_curve_data.csv` - 曲线数据

### `plot_stability_distribution(cancer_type, xai_method, pairwise_results, k)`

**功能**: 绘制稳定性指标的分布图

**4个子图**:
1. Jaccard分布直方图
2. Kuncheva分布直方图（含随机基线）
3. RBO分布直方图
4. 分层箱线图（同fold vs 跨fold）

**输出**: `stability_distribution_top{k}.png`

### `plot_feature_frequency(cancer_type, xai_method, feature_freq, k)`

**功能**: 绘制特征频次分析图

**2个子图**:
1. 频次分布直方图（所有特征的频次分布）
2. Top-20稳定特征柱状图（最稳定的20个特征）

**输出**: `feature_frequency_top{k}.png`

### `create_cross_cancer_plots(summary_df)`

**功能**: 创建跨癌症的稳定性对比图

**子图**: 每个XAI方法一行，3列
1. Jaccard对比（各癌症）
2. Kuncheva对比（各癌症）
3. 稳定核心特征数量对比

**输出**: `cross_cancer_stability_comparison.png`

---

## 📝 报告生成方法

### `create_summary_report(cancer_type, xai_method, analysis_result)`

**功能**: 创建CSV格式的稳定性摘要报告

**输出**: `stability_summary_report.csv`

**包含列**: 
- 基础信息（癌症、方法、Top-k）
- 稳定性指标（mean, std, median, q25, q75）
- 随机基线对比
- 分层分析结果

### `create_text_report(cancer_type, xai_method, analysis_result, output_dir)`

**功能**: 创建文本格式的分析报告

**输出**: `stability_analysis_report.txt`

**内容结构**:
1. 基本信息
2. 主要发现摘要（各Top-k的稳定性）
3. 详细统计信息
4. 稳定核心特征列表
5. 解读说明
6. 文件说明

### `create_cross_xai_summary_report(cross_xai_results)`

**功能**: 创建跨XAI方法的一致性汇总报告

**输出**: `cross_xai_consistency_report.txt`

**内容**:
- 各癌症的XAI一致性分析
- 成对一致性
- 共识特征
- Top高一致性特征
- 整体统计

---

## 🔄 主流程方法

### `_analyze_single_topk_stability(cancer_type, xai_method, rankings, k)`

**功能**: 分析单个Top-k值的完整稳定性（私有辅助方法）

**流程**:
1. 成对稳定性 → `analyze_pairwise_stability()`
2. 稳定性摘要 → `create_stability_summary()`
3. 分层分析 → `create_stratified_analysis()`
4. 特征频次 → `calculate_feature_frequency()`
5. 随机基线 → `calculate_random_baseline()`
6. 识别稳定核心 → 筛选频次≥阈值的特征
7. 可视化 → `plot_stability_distribution()` 和 `plot_feature_frequency()`
8. 保存结果 → CSV文件

**返回**: dict - 该Top-k的所有结果

### `_analyze_all_topk_stability(cancer_type, xai_method, rankings)`

**功能**: 分析所有Top-k值的稳定性（私有辅助方法）

**循环**: 对 `self.top_k_values` 中的每个k调用 `_analyze_single_topk_stability()`

**返回**: dict `{k: result, ...}` 加上 `'rankings'` 键

### `_analyze_and_save_performance_correlations(cancer_type, xai_method, stability_results, performance_data, output_dir)`

**功能**: 分析并保存性能相关性（私有辅助方法）

**步骤**:
1. 调用 `analyze_stability_performance_correlation()`
2. 转换结果为DataFrame
3. 保存为 `performance_correlations.csv`

### `analyze_cancer_xai_combination(cancer_type, xai_method)`

**功能**: 分析单个癌症-XAI组合的完整稳定性（主入口）

**流程**:
1. 加载特征排名
2. 加载性能数据
3. 验证数据充分性
4. 分析所有Top-k稳定性
5. 绘制Overlap@k曲线
6. 性能相关性分析
7. 创建报告

**返回**: dict - 包含所有分析结果

### `run_full_analysis(cancer_types=None)`

**功能**: 运行完整的稳定性分析（主流程入口）

**流程**:
```
1. 获取癌症类型列表（如果未提供则自动发现）
2. 嵌套循环:
   for 每个癌症:
       for 每个XAI方法:
           调用 analyze_cancer_xai_combination()
           创建单个组合的报告
3. 跨XAI一致性分析:
   for 每个癌症:
       调用 analyze_cross_xai_consistency()
       保存跨XAI结果
4. 创建跨癌症汇总
5. 创建跨XAI汇总报告
```

---

## 🎯 关键设计模式

### 1. 统一路径管理
```python
# 不推荐（分散的路径构建）
path1 = os.path.join(dir1, dir2, file)
path2 = os.path.join(dir1, dir3, file)

# 推荐（统一方法）
path1 = self._build_xai_results_path(method, cancer, subdir)
```

### 2. 配置参数化
```python
# 不推荐（硬编码）
if freq >= 0.7:  # 魔术数字

# 推荐（配置化）
if freq >= self.stability_threshold:
```

### 3. 方法拆分
```python
# 不推荐（100+行的大方法）
def analyze_all():
    # 很多代码...

# 推荐（拆分为小方法）
def analyze_all():
    result1 = self._step1()
    result2 = self._step2(result1)
    return self._step3(result2)
```

### 4. 统一统计计算
```python
# 不推荐（重复代码）
mean1 = np.mean(values1)
std1 = np.std(values1)
...

# 推荐（统一方法）
stats = self._calculate_stats_summary(values)
```

---

## 💡 使用建议

### 1. 自定义分析
```python
# 修改配置
analyzer.top_k_values = [10, 20, 50]
analyzer.stability_threshold = 0.8

# 只分析特定癌症
analyzer.run_full_analysis(cancer_types=['BRCA', 'LUAD'])
```

### 2. 单步调试
```python
# 逐步调用
rankings = analyzer.load_feature_rankings('BRCA', 'shap')
pairwise = analyzer.analyze_pairwise_stability(rankings, 100)
summary = analyzer.create_stability_summary(pairwise)
```

### 3. 结果提取
```python
result = analyzer.analyze_cancer_xai_combination('BRCA', 'shap')

# 获取Top-100稳定性
top100 = result['stability_results'][100]
jaccard_mean = top100['stability_summary']['jaccard_mean']

# 获取稳定核心特征
stable_core = top100['stable_core']
```

---

**最后更新**: 2025-10-14  
**文档版本**: 1.0


























