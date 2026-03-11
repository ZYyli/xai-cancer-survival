# 特征稳定性分析运行指南

## 📖 快速开始

### 1. 测试运行（推荐先运行）
测试单个癌症，快速验证代码是否正常：
```bash
cd /home/zuoyiyi/SNN/TCGA/operation
python run_stability_analysis_fast.py --mode test --cancer BLCA
```

### 2. 完整分析
```bash
python run_stability_analysis_fast.py --mode full
```

### 3. 原始方式运行
```bash
python feature_stability_analysis.py
```

## ⏱️ 运行时间估计

### 计算量分析
- **总组合数**: 4 XAI方法 × 15 癌症 = 60 个组合
- **文件加载**: 60 × 50 = 3,000 个CSV文件
- **成对比较**: 60 × 1,225 = 73,500 对
- **Overlap曲线**: 60 × 20个k值 × 1,225对 ≈ 1,470,000 次计算

### 预计耗时
- **测试单个癌症（1个XAI）**: 约 2-5 分钟
- **单个癌症（4个XAI）**: 约 8-20 分钟
- **完整分析（60个组合）**: 约 **2-8 小时**（取决于硬件）

## 🐌 运行慢的主要原因

### 1. Overlap曲线计算（最耗时）
**位置**: `plot_overlap_curves()` 函数（第717-778行）

```python
for k in range(10, 201, 10):  # 20个k值
    pairwise_results = self.analyze_pairwise_stability(rankings, k)
    # 每次都重新计算1225对！
```

**优化方案**: 
- 减少k值采样点：`range(10, 201, 20)` → 从20个减少到10个
- 或者注释掉这部分（如果不需要曲线图）

### 2. 高分辨率图片生成
**位置**: 多处 `plt.savefig(..., dpi=300)`

**优化方案**: 
- 降低DPI：`dpi=150` 或 `dpi=200`
- 只保存PNG，不保存PDF

### 3. 成对稳定性计算
每个组合计算 C(50,2) = 1,225 对，这是必需的。

## 🚀 性能优化方案

### 方案1: 减少Overlap曲线采样（推荐）

修改第722行：
```python
# 原代码
for k in range(10, 201, 10):  # 20个点

# 优化后
for k in range(20, 201, 20):  # 10个点，速度提升约50%
```

### 方案2: 降低图片质量

修改所有 `savefig` 的dpi参数：
```python
# 原代码
plt.savefig(..., dpi=300)

# 优化后
plt.savefig(..., dpi=150)  # 速度提升约75%，图片质量仍可接受
```

### 方案3: 跳过某些分析

在 `analyze_cancer_xai_combination()` 中注释掉不需要的部分：

```python
# 如果不需要overlap曲线，注释掉第1028-1035行
# print(f"   📈 绘制Overlap@k曲线...")
# jaccard_curve, kuncheva_curve = self.plot_overlap_curves(cancer_type, xai_method, rankings)
# ...
```

### 方案4: 并行处理（高级）

使用多进程处理不同的癌症：

```python
from multiprocessing import Pool

def analyze_one_combination(args):
    cancer_type, xai_method, analyzer = args
    return analyzer.analyze_cancer_xai_combination(cancer_type, xai_method)

# 使用4个进程并行
with Pool(4) as pool:
    results = pool.map(analyze_one_combination, task_list)
```

## 📊 运行监控

使用提供的 `run_stability_analysis_fast.py`，可以看到：
- 实时进度 `[15/60]`
- 每个组合的耗时
- 预计完成时间（ETA）

示例输出：
```
[15/60] BLCA-shap ✅ 完成 (3.2s) - ETA: 14:35:20
[16/60] BLCA-IG ✅ 完成 (2.8s) - ETA: 14:35:18
```

## 🎯 建议的运行策略

### 第一次运行
1. **测试运行**（5分钟）
   ```bash
   python run_stability_analysis_fast.py --mode test --cancer BLCA --xai shap
   ```

2. **检查结果**
   ```bash
   ls -lh stability_analysis_test/BLCA/shap/
   ```

3. **如果满意，运行完整分析**（后台运行，防止断开）
   ```bash
   nohup python run_stability_analysis_fast.py --mode full > stability_log.txt 2>&1 &
   ```

4. **查看进度**
   ```bash
   tail -f stability_log.txt
   ```

### 快速版本（如果时间紧张）

创建快速配置版本，修改 `feature_stability_analysis.py` 第45-48行：

```python
# 快速版本配置
self.top_k_values = [100]  # 只分析Top-100，不分析20,50,200
self.random_simulations = 100  # 从1000减少到100
```

然后注释掉 overlap 曲线生成（第1028-1035行）。

这样可以将运行时间从 **2-8小时** 降低到 **30-60分钟**。

## 📁 输出结果说明

```
stability_analysis/
├── cross_cancer_stability_summary.csv      # 跨癌症汇总
├── cross_cancer_summary_report.txt         # 文本报告
├── cross_xai_consistency_report.txt        # 跨XAI一致性报告
└── [CANCER]/
    ├── [XAI_METHOD]/
    │   ├── stability_summary_report.csv
    │   ├── performance_correlations.csv
    │   ├── overlap_curves.png              # ⚠️ 最耗时的部分
    │   ├── stability_distribution_top*.png
    │   └── feature_frequency_top*.png
    └── cross_xai_analysis/
        └── ...
```

## ❓ 常见问题

### Q: 运行到一半卡住了？
A: 可能在生成overlap曲线，这部分最耗时，请耐心等待。

### Q: 内存不足？
A: 减少同时处理的癌症数量，或增加系统swap。

### Q: 想中断后继续运行？
A: 目前不支持断点续传，建议使用测试模式逐个运行。

## 💡 最佳实践

1. **先测试**：用1个癌症测试，确保代码正常
2. **后台运行**：使用 `nohup` 防止SSH断开
3. **监控进度**：使用 `tail -f` 查看日志
4. **优化配置**：根据需求调整 top_k_values 和图片质量
5. **分批运行**：如果时间充裕，可以每次运行5个癌症

## 🔧 故障排除

如果遇到错误，检查：
1. 数据文件是否存在
2. 路径配置是否正确
3. 是否有足够的磁盘空间（每个癌症约100-200MB输出）
4. Python环境是否安装了所有依赖包
























