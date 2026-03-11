# 特征稳定性分析 - 快速参考

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 2. 运行分析
```bash
python feature_stability_analysis.py
```

---

## 📊 核心指标速查

| 指标 | 公式/范围 | 最佳值 | 含义 |
|------|-----------|--------|------|
| **Jaccard** | [0, 1] | >0.5 | Top-k特征重叠度 |
| **Kuncheva** | [-1, 1] | >0.3 | 机会校正的稳定性 |
| **RBO** | [0, 1] | >0.5 | 考虑排名的相似度 |
| **Spearman** | [-1, 1] | >0.7 | 排名一致性 |

---

## 🔧 配置参数

```python
# 在代码中修改这些参数
analyzer.top_k_values = [20, 50, 100, 200]  # Top-k列表
analyzer.stability_threshold = 0.7          # 稳定性阈值
analyzer.random_simulations = 1000          # 随机模拟次数
analyzer.min_rankings_required = 3          # 最少文件数
```

---

## 📁 关键输出文件

### 各癌症-XAI组合
```
[CANCER]/[XAI_METHOD]/
├── stability_summary_report.csv      # 📊 数值汇总
├── stable_features_top{k}.csv        # ⭐ 稳定特征列表
├── performance_correlations.csv      # 📈 性能相关性
└── overlap_curves.png                # 📉 稳定性曲线
```

### 跨XAI分析
```
[CANCER]/cross_xai_analysis/
├── cross_xai_consistency_summary.csv # 🔗 一致性汇总
├── consensus_features_top{k}.csv     # 🎯 共识特征
└── features_by_consensus_top{k}.csv  # 📋 按一致性排序
```

### 全局汇总
```
stability_analysis/
├── cross_cancer_stability_summary.csv # 📊 跨癌症汇总
├── cross_xai_consistency_report.txt   # 📄 跨XAI报告
└── cross_cancer_summary_report.txt    # 📄 总结报告
```

---

## 🎯 常用代码片段

### 分析单个癌症
```python
result = analyzer.analyze_cancer_xai_combination('BRCA', 'shap')
```

### 查看Top-100稳定性
```python
summary = result['stability_results'][100]['stability_summary']
print(f"Jaccard: {summary['jaccard_mean']:.3f}")
```

### 获取稳定核心特征
```python
stable_core = result['stability_results'][100]['stable_core']
print(f"稳定核心: {len(stable_core)} 个特征")
```

### 查看跨XAI共识特征
```python
consensus = pd.read_csv('BRCA/cross_xai_analysis/consensus_features_top100.csv')
```

---

## 📈 稳定性判断标准

### Jaccard相似度
- ✅ **>0.5**: 稳定性良好
- ⚠️ **0.3-0.5**: 中等稳定性  
- ❌ **<0.3**: 不稳定

### Kuncheva指数
- ✅ **>0.3**: 显著超越随机
- ⚠️ **0-0.3**: 略超随机
- ❌ **<0**: 低于随机

### 稳定核心特征
- **定义**: 在≥70%模型中都是Top-k的特征
- **应用**: 用于后续生物标志物研究

---

## 🐛 常见问题

### Q1: "数据不足，跳过"
**原因**: 排名文件少于3个  
**解决**: 检查数据文件是否完整

### Q2: "目录不存在"
**原因**: 路径配置错误  
**解决**: 检查 `results_dir` 路径

### Q3: Jaccard值很低
**原因**: 特征选择不稳定是正常现象  
**说明**: 关注稳定核心特征即可

---

## 📚 方法速查

| 方法 | 用途 | 返回值 |
|------|------|--------|
| `load_feature_rankings()` | 加载排名文件 | 字典 {(r,f): DataFrame} |
| `analyze_pairwise_stability()` | 成对稳定性 | 指标字典 |
| `calculate_feature_frequency()` | 特征频次 | {feature: freq} |
| `analyze_cross_xai_consistency()` | 跨XAI一致性 | 一致性结果 |
| `run_full_analysis()` | 完整分析 | 无（保存结果） |

---

## 🔍 结果解读

### 1. 查看稳定性摘要
```bash
cat BRCA/shap/stability_analysis_report.txt
```

### 2. 加载稳定特征
```python
df = pd.read_csv('BRCA/shap/stable_features_top100.csv')
top_10 = df.head(10)  # 前10个最稳定特征
```

### 3. 对比不同Top-k
```python
for k in [20, 50, 100, 200]:
    df = pd.read_csv(f'BRCA/shap/stable_features_top{k}.csv')
    print(f"Top-{k}: {len(df)} 个特征")
```

---

## 💡 最佳实践

1. **先运行小规模测试**: 单个癌症、单个XAI方法
2. **检查中间结果**: 确认稳定性曲线合理
3. **关注稳定核心**: frequency >= 0.7 的特征
4. **对比多个XAI方法**: 找共识特征
5. **结合性能分析**: 查看 performance_correlations.csv

---

## 📞 获取帮助

1. 查看完整文档: `FEATURE_STABILITY_ANALYSIS_README.md`
2. 查看代码注释: 所有方法都有详细文档字符串
3. 查看输出报告: `stability_analysis_report.txt` 中有解读说明

---

**提示**: 这是快速参考，详细信息请查看完整README文档。


























