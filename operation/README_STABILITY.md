# 特征稳定性分析 - 快速使用指南

## 🎯 三种运行方式

### ⭐ 方式1: 一键运行（最简单，推荐）
```bash
cd /home/zuoyiyi/SNN/TCGA/operation
./run_stability_quick.sh
```
这个脚本会：
1. 先运行测试（2-5分钟）
2. 测试通过后询问是否继续
3. 可选择前台或后台运行完整分析

---

### 方式2: 手动分步运行
```bash
cd /home/zuoyiyi/SNN/TCGA/operation

# 步骤1: 测试单个癌症（验证代码）
python run_stability_analysis_fast.py --mode test --cancer BLCA

# 步骤2: 如果测试成功，运行完整分析
python run_stability_analysis_fast.py --mode full
```

---

### 方式3: 后台运行（防止SSH断开）
```bash
cd /home/zuoyiyi/SNN/TCGA/operation

# 启动后台运行
nohup python run_stability_analysis_fast.py --mode full > stability.log 2>&1 &

# 查看进度
tail -f stability.log

# 查看进程
ps aux | grep stability
```

---

## ⏱️ 时间预估

| 运行模式 | 预计时间 | 说明 |
|---------|---------|------|
| 测试单个癌症（1个XAI） | 2-5 分钟 | 快速验证 |
| 测试单个癌症（4个XAI） | 8-20 分钟 | 完整测试 |
| 完整分析（60个组合） | **2-8 小时** | 全部数据 |

---

## 🐌 为什么运行慢？

### 计算量巨大

```
分析规模:
  ├─ 4 种XAI方法 (SHAP, IG, LRP, PFI)
  ├─ 15 种癌症类型
  ├─ 60 个组合总数
  │
  └─ 每个组合:
      ├─ 加载 50 个CSV文件 (10 repeat × 5 fold)
      ├─ 计算 1,225 对成对比较 C(50,2)
      ├─ 分析 4 个Top-k值 (20, 50, 100, 200)
      └─ 绘制 20 个k值的Overlap曲线 (最耗时!)

总计算量 ≈ 147万次成对稳定性计算
```

### 最耗时的部分

**Overlap曲线绘制** (第717-778行)：
- 从k=10到200，每隔10计算一次（20个点）
- 每个点都要重新计算1,225对的稳定性
- 占总时间的 **60-70%**

---

## 🚀 性能优化技巧

### 快速版本配置

如果时间紧张，可以修改配置以大幅提速：

**编辑** `feature_stability_analysis.py` **第45-48行:**

```python
# 原配置（慢但完整）
self.top_k_values = [20, 50, 100, 200]
self.random_simulations = 1000

# 快速配置（速度提升5-10倍）
self.top_k_values = [100]  # 只分析Top-100
self.random_simulations = 100  # 减少模拟次数
```

**并注释掉** 第1028-1035行（跳过Overlap曲线）：

```python
# 注释掉这部分可节省60%时间
# print(f"   📈 绘制Overlap@k曲线...")
# jaccard_curve, kuncheva_curve = self.plot_overlap_curves(...)
```

**结果**: 运行时间从 2-8小时 降至 **30-60分钟**

---

## 📊 进度监控

运行时会看到实时进度：

```
[15/60] BLCA-shap ✅ 完成 (3.2s) - ETA: 14:35:20
[16/60] BLCA-IG ✅ 完成 (2.8s) - ETA: 14:35:18
[17/60] BLCA-LRP ✅ 完成 (3.5s) - ETA: 14:35:25
```

显示信息：
- `[15/60]`: 当前进度（第15个/共60个）
- `BLCA-shap`: 当前分析的癌症-XAI组合
- `(3.2s)`: 该组合耗时
- `ETA: 14:35:20`: 预计完成时间

---

## 📁 输出结果

```
stability_analysis/
├── cross_cancer_stability_summary.csv      # ⭐ 跨癌症汇总数据
├── cross_cancer_summary_report.txt         # ⭐ 主要结果报告
├── cross_xai_consistency_report.txt        # ⭐ 跨XAI一致性分析
│
└── [癌症名称]/
    ├── [XAI方法]/
    │   ├── stability_summary_report.csv    # 稳定性统计
    │   ├── performance_correlations.csv    # 性能相关性
    │   ├── overlap_curves.png              # 稳定性曲线图
    │   ├── stability_distribution_top*.png # 分布图
    │   └── stable_features_top*.csv        # 稳定特征列表
    │
    └── cross_xai_analysis/
        ├── cross_xai_consistency_summary.csv
        ├── consensus_features_top*.csv      # 共识特征
        └── features_by_consensus_top*.csv
```

---

## ❓ 常见问题

### Q1: 如何只分析特定的癌症？

**A:** 测试模式可以指定单个癌症：
```bash
python run_stability_analysis_fast.py --mode test --cancer LUAD --xai IG
```

### Q2: 运行到一半卡住不动？

**A:** 可能在绘制Overlap曲线，这是最耗时的部分，请耐心等待。可以通过日志确认：
```bash
tail -f stability.log
```

### Q3: 如何停止正在运行的分析？

**A:** 
```bash
# 查找进程ID
ps aux | grep stability

# 停止进程
kill <PID>
```

### Q4: 运行失败了怎么办？

**A:** 检查以下几点：
1. 数据文件是否存在
2. 路径是否正确（检查 `_results_2/` 目录）
3. 是否有足够磁盘空间（建议预留 20GB）
4. Python依赖包是否完整

### Q5: 想提速但不想修改代码？

**A:** 运行时使用优先级较高的进程：
```bash
nice -n -10 python run_stability_analysis_fast.py --mode full
```

---

## 💡 推荐工作流程

### 第一次使用
```bash
# 1. 快速测试（5分钟）
./run_stability_quick.sh

# 2. 查看测试结果
ls -lh ../stability_analysis_test/BLCA/shap/

# 3. 如果满意，启动完整分析（后台）
nohup python run_stability_analysis_fast.py --mode full > log.txt 2>&1 &

# 4. 定期查看进度
tail -20 log.txt
```

### 时间紧急时
1. 修改配置为快速版本（见上文"性能优化技巧"）
2. 运行完整分析（30-60分钟）
3. 后续如需要可以补充完整分析

---

## 📞 技术支持

如有问题，查看详细指南：
```bash
cat STABILITY_ANALYSIS_GUIDE.md
```

或检查代码文档：
```bash
python feature_stability_analysis.py --help
```
























