# Bootstrap XAI 快速开始指南

## 🚀 5分钟快速上手

### 第一步：测试单个癌症
```bash
cd /home/zuoyiyi/SNN/TCGA/operation

# 运行测试脚本（只测试5个bootstrap，确保环境正常）
./test_single_bootstrap_xai.sh
```

### 第二步：运行完整分析（如果测试成功）
```bash
# 运行所有癌症的4种XAI方法（500次bootstrap）
./run_bootstrap_xai_example.sh
```

## 📝 基本命令

### 单个方法，单个癌症
```bash
python save_feature_importance_bootstrap.py \
    --csv_path /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_cancer_single \
    --results_dir /home/zuoyiyi/SNN/TCGA/results_bootstrap_500 \
    --output_dir /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500 \
    --method shap \
    --num_bootstrap 500
```

### 自定义参数示例
```bash
# 100次bootstrap，使用LRP方法
python save_feature_importance_bootstrap.py \
    --csv_path /path/to/data \
    --results_dir /path/to/results_bootstrap \
    --output_dir /path/to/output \
    --method lrp \
    --num_bootstrap 100 \
    --seed_base 1
```

## 🔍 检查结果

### 查看生成的文件
```bash
# 查看某个癌症的SHAP结果
ls -lh /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500/BLCA/shap_feature_importance_bootstrap/

# 查看CSV文件内容（前10行）
head -10 /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500/BLCA/shap_feature_importance_bootstrap/seed1_shap_feature_importance_ranking.csv
```

### 统计成功率
```bash
# 统计某个癌症的SHAP结果文件数
ls /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500/BLCA/shap_feature_importance_bootstrap/*.csv | wc -l

# 应该等于 num_bootstrap（如500）
```

## ⚙️ 参数对照表

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--csv_path` | 预处理数据路径 | - | `/path/to/preprocess_cancer_single` |
| `--results_dir` | Bootstrap结果目录 | - | `/path/to/results_bootstrap_500` |
| `--output_dir` | 输出保存目录 | - | `/path/to/output` |
| `--method` | XAI方法 | - | `shap`, `ig`, `lrp`, `pfi` |
| `--num_bootstrap` | Bootstrap次数 | 100 | 100, 500 |
| `--seed_base` | 起始seed | 1 | 1, 101, 201 |

## 🎯 4种XAI方法对比

| 方法 | 速度 | 内存 | 理论基础 | 推荐场景 |
|------|------|------|----------|----------|
| **LRP** | ⚡⚡⚡ 最快 | 💾 低 | 守恒性原则 | 大规模快速分析 |
| **IG** | ⚡⚡ 快 | 💾💾 中 | 积分梯度 | 需要归因路径 |
| **SHAP** | ⚡ 慢 | 💾💾💾 高 | Shapley值 | 需要理论保证 |
| **PFI** | 🐌 很慢 | 💾 低 | 排列测试 | 全局重要性 |

### 运行时间估算（500次bootstrap）
- LRP: ~40小时
- IG: ~50小时
- SHAP: ~60小时
- PFI: ~80小时

## 🐛 快速故障排除

### 问题1：找不到模型文件
```bash
# 检查模型是否存在
ls /home/zuoyiyi/SNN/TCGA/results_bootstrap_500/BLCA/models/bootstrap_model_seed*.pt | head -5

# 如果没有文件，说明bootstrap训练有问题
```

### 问题2：找不到OOB索引
```bash
# 检查索引文件
ls /home/zuoyiyi/SNN/TCGA/results_bootstrap_500/BLCA/models/oob_idx_seed*.npy | head -5

# 如果没有，说明训练时没有保存索引
```

### 问题3：CUDA内存不足
```bash
# 方法1：减少并行，逐个运行
python save_feature_importance_bootstrap.py ... --method shap

# 方法2：使用CPU
export CUDA_VISIBLE_DEVICES=""
python save_feature_importance_bootstrap.py ...
```

### 问题4：数据文件找不到
```bash
# 检查数据文件名（注意大小写）
ls /home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_cancer_single/*_preprocessed.csv

# 确保文件名格式：{cancer_lower}_preprocessed.csv
```

## 📊 查看进度

### 实时监控
```bash
# 打开新终端，实时查看输出文件
watch -n 10 "ls /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500/*/shap_feature_importance_bootstrap/*.csv | wc -l"

# 预期总数：15个癌症 × 500个bootstrap = 7500个文件
```

### GPU使用监控
```bash
# 监控GPU使用
watch -n 1 nvidia-smi
```

## 🔗 相关文件

| 文件 | 用途 |
|------|------|
| `save_feature_importance_bootstrap.py` | 主程序（Bootstrap版本） |
| `save_feature_importance_only.py` | 主程序（Nested CV版本） |
| `run_bootstrap_xai_example.sh` | 完整运行脚本 |
| `test_single_bootstrap_xai.sh` | 测试脚本 |
| `BOOTSTRAP_XAI_README.md` | 详细文档 |
| `QUICK_START_BOOTSTRAP_XAI.md` | 本文件 |

## ✅ 验证结果

### 检查文件完整性
```bash
# 检查每个癌症的文件数量
for cancer in BLCA BRCA COADREAD GBMLGG HNSC KIRC KIRP LGG LIHC LUAD LUSC PAAD SKCM STAD UCEC; do
    count=$(ls /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500/${cancer}/shap_feature_importance_bootstrap/*.csv 2>/dev/null | wc -l)
    echo "${cancer}: ${count}/500"
done
```

### 检查CSV格式
```bash
# 查看第一个文件的格式
head -5 /home/zuoyiyi/SNN/TCGA/bootstrap_xai_results_500/BLCA/shap_feature_importance_bootstrap/seed1_shap_feature_importance_ranking.csv

# 应该包含：feature_name, importance_score, rank
```

## 💡 最佳实践

1. **先测试后运行**：使用 `test_single_bootstrap_xai.sh` 先测试
2. **分批运行**：先运行快速的方法（LRP），再运行慢的（PFI）
3. **监控资源**：使用 `nvidia-smi` 和 `htop` 监控
4. **定期备份**：完成一个方法后立即备份结果
5. **日志记录**：使用 `nohup` 或 `screen` 保存运行日志

### 推荐运行顺序
```bash
# 1. 先运行最快的LRP（验证流程）
python save_feature_importance_bootstrap.py ... --method lrp

# 2. 运行IG（中等速度）
python save_feature_importance_bootstrap.py ... --method ig

# 3. 运行SHAP（较慢但常用）
python save_feature_importance_bootstrap.py ... --method shap

# 4. 最后运行PFI（最慢）
python save_feature_importance_bootstrap.py ... --method pfi
```

## 📞 获取帮助

```bash
# 查看程序帮助
python save_feature_importance_bootstrap.py --help

# 查看详细文档
cat BOOTSTRAP_XAI_README.md
```

---

**提示**: 如果遇到问题，请先运行测试脚本，查看详细的错误信息！












