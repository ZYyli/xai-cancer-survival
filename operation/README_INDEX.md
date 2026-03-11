# 特征稳定性分析 - 文档导航

欢迎使用特征稳定性分析工具！本目录包含完整的文档资料，帮助您理解和使用这个代码。

---

## 📚 文档列表

### 1. 📖 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md)
**适合**: 首次使用者、需要全面了解的用户

**内容**:
- ✅ 项目概述和目的
- ✅ 核心功能介绍
- ✅ 稳定性指标详解
- ✅ 完整的代码结构说明
- ✅ 使用方法和示例
- ✅ 配置参数说明
- ✅ 输出文件详解
- ✅ 关键方法说明
- ✅ 技术细节和注意事项

**推荐阅读顺序**: 第一个阅读

---

### 2. ⚡ [快速参考](QUICK_REFERENCE.md)
**适合**: 已熟悉代码、需要快速查询的用户

**内容**:
- ⚡ 快速开始指南
- ⚡ 核心指标速查表
- ⚡ 配置参数一览
- ⚡ 关键输出文件位置
- ⚡ 常用代码片段
- ⚡ 稳定性判断标准
- ⚡ 常见问题解答
- ⚡ 方法速查表

**推荐阅读顺序**: 第二个阅读，或作为日常参考

---

### 3. 🔄 [工作流程图](WORKFLOW_DIAGRAM.md)
**适合**: 想要深入理解代码执行流程的用户

**内容**:
- 🔄 整体流程图
- 🔄 单癌症-XAI分析详细流程
- 🔄 跨XAI一致性分析流程
- 🔄 关键数据结构说明
- 🔄 计算复杂度分析
- 🔄 稳定性指标计算示例
- 🔄 性能相关性分析流程
- 🔄 可视化输出示例
- 🔄 文件I/O流程

**推荐阅读顺序**: 第三个阅读，帮助理解内部逻辑

---

### 4. 💻 [代码详细注释](CODE_DOCUMENTATION_CN.md)
**适合**: 开发者、需要修改代码的用户

**内容**:
- 💻 类定义说明
- 💻 所有方法的详细注释（中文）
- 💻 参数说明和返回值
- 💻 实现逻辑解释
- 💻 代码示例
- 💻 设计模式说明
- 💻 使用建议

**推荐阅读顺序**: 第四个阅读，适合深入研究代码

---

## 🎯 根据需求选择文档

### 我是新手，想快速上手
1. 阅读 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md) 的"快速开始"部分
2. 查看 [快速参考](QUICK_REFERENCE.md) 的代码示例
3. 运行代码并查看输出

### 我想理解代码做了什么
1. 先看 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md) 了解功能
2. 再看 [工作流程图](WORKFLOW_DIAGRAM.md) 理解执行流程
3. 查看实际输出文件验证理解

### 我需要修改或扩展代码
1. 阅读 [代码详细注释](CODE_DOCUMENTATION_CN.md) 了解各方法
2. 参考 [工作流程图](WORKFLOW_DIAGRAM.md) 理解整体架构
3. 使用 [快速参考](QUICK_REFERENCE.md) 快速查找信息

### 我在使用中遇到问题
1. 先查 [快速参考](QUICK_REFERENCE.md) 的常见问题部分
2. 检查 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md) 的注意事项
3. 查看 [代码详细注释](CODE_DOCUMENTATION_CN.md) 了解参数要求

### 我想理解输出结果
1. 查看 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md) 的"输出文件"和"稳定性解读指南"
2. 参考 [快速参考](QUICK_REFERENCE.md) 的稳定性判断标准
3. 查看 [工作流程图](WORKFLOW_DIAGRAM.md) 的可视化输出示例

---

## 📂 代码文件

### 主要文件
- `feature_stability_analysis.py` - 主分析脚本（已优化版本）

### 文档文件
- `FEATURE_STABILITY_ANALYSIS_README.md` - 完整说明文档
- `QUICK_REFERENCE.md` - 快速参考
- `WORKFLOW_DIAGRAM.md` - 工作流程图
- `CODE_DOCUMENTATION_CN.md` - 代码详细注释
- `README_INDEX.md` - 本文档（导航索引）

---

## 🚀 快速开始（5分钟上手）

### 步骤1: 安装依赖
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 步骤2: 配置路径
```python
results_dir = '/path/to/your/results'    # 改为你的数据路径
output_dir = '/path/to/output'           # 改为你的输出路径
```

### 步骤3: 运行分析
```bash
python feature_stability_analysis.py
```

或者在Python中：
```python
from feature_stability_analysis import FeatureStabilityAnalyzer

analyzer = FeatureStabilityAnalyzer(
    results_dir=results_dir,
    output_dir=output_dir,
    xai_methods=['shap', 'IG', 'LRP', 'PFI']
)

analyzer.run_full_analysis()
```

### 步骤4: 查看结果
结果保存在 `output_dir` 目录下：
- 📊 CSV文件 - 数值结果
- 📈 PNG图片 - 可视化
- 📄 TXT报告 - 文字说明

详细的输出文件说明请查看 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md#输出文件)

---

## 📊 核心概念速览

### 稳定性分析
评估不同模型训练是否选择相似的重要特征

### 主要指标
- **Jaccard**: 特征集重叠度 (>0.5为好)
- **Kuncheva**: 机会校正的稳定性 (>0.3显著)
- **RBO**: 考虑排名的相似度
- **Spearman**: 排名一致性

### 分析对象
- **10×5 CV**: 10次重复，5折交叉验证 = 50个模型
- **4种XAI**: SHAP, IG, LRP, PFI
- **15种癌症**: BLCA, BRCA, LUAD等
- **多个Top-k**: 20, 50, 100, 200

### 主要输出
- ⭐ 稳定核心特征（在≥70%模型中都重要的特征）
- 🔗 跨XAI共识特征（所有XAI方法都认为重要的特征）
- 📈 性能相关性（稳定性与模型性能的关系）

---

## ❓ 常见问题快速解答

**Q: 我的数据在哪个目录？**  
A: 查看 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md#使用方法) 的目录结构部分

**Q: Jaccard值是0.3，这好吗？**  
A: 中等偏下，查看 [快速参考](QUICK_REFERENCE.md#稳定性判断标准)

**Q: 如何找到最稳定的特征？**  
A: 查看输出的 `stable_features_top{k}.csv` 文件，按frequency排序

**Q: 什么是稳定核心特征？**  
A: 在≥70%的模型中都是Top-k的特征，详见 [完整说明文档](FEATURE_STABILITY_ANALYSIS_README.md#稳定性解读指南)

**Q: 如何修改稳定性阈值？**  
A: 修改 `analyzer.stability_threshold = 0.8`，详见 [快速参考](QUICK_REFERENCE.md#配置参数)

**Q: 代码运行很慢？**  
A: 正常，1,225个成对比较×60个组合，查看 [工作流程图](WORKFLOW_DIAGRAM.md#计算复杂度分析)

---

## 📞 获取更多帮助

1. **查看文档**: 按照上述推荐顺序阅读文档
2. **查看代码注释**: 代码中有详细的文档字符串
3. **查看输出报告**: `stability_analysis_report.txt` 有解读说明
4. **检查示例**: 文档中有大量代码示例

---

## 🔄 文档更新记录

- **2025-10-14**: 创建所有文档
  - 完整说明文档
  - 快速参考
  - 工作流程图
  - 代码详细注释
  - 导航索引

---

## 📝 推荐阅读路径

### 路径1: 快速使用（30分钟）
```
完整说明文档（概述+使用方法）
    ↓
快速参考（代码片段）
    ↓
运行代码
    ↓
查看输出
```

### 路径2: 深入理解（2小时）
```
完整说明文档（全部）
    ↓
工作流程图
    ↓
代码详细注释
    ↓
查看实际代码
```

### 路径3: 问题解决（10分钟）
```
快速参考（常见问题）
    ↓
完整说明文档（注意事项）
    ↓
代码详细注释（相关方法）
```

---

**开始探索**: 点击上方链接访问各个文档！

**提示**: 所有文档都是中文的，便于理解。建议按推荐顺序阅读，循序渐进。


























