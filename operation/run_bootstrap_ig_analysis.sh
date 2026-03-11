#!/bin/bash

# Bootstrap IG 可解释性分析脚本
# 对100个bootstrap模型进行IG分析和Cox回归

echo "🚀 开始Bootstrap IG可解释性分析..."
echo "时间: $(date)"

# 配置路径
CSV_PATH="/home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_cancer_single"
BOOTSTRAP_RESULTS_DIR="/home/zuoyiyi/SNN/TCGA/results_bootstrap"
IG_BOOTSTRAP_DIR="/home/zuoyiyi/SNN/TCGA/IG_bootstrap_results"

# 检查路径是否存在
if [ ! -d "$CSV_PATH" ]; then
    echo "❌ 错误: 数据路径不存在: $CSV_PATH"
    exit 1
fi

if [ ! -d "$BOOTSTRAP_RESULTS_DIR" ]; then
    echo "❌ 错误: Bootstrap结果路径不存在: $BOOTSTRAP_RESULTS_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$IG_BOOTSTRAP_DIR"

echo "📁 使用路径："
echo "   数据路径: $CSV_PATH"
echo "   Bootstrap结果: $BOOTSTRAP_RESULTS_DIR"
echo "   IG输出目录: $IG_BOOTSTRAP_DIR"

# 激活pytorch环境
echo "🔧 激活pytorch环境..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# 运行bootstrap IG分析
echo "📊 开始分析100个bootstrap模型..."
python ig_bootstrap_analysis.py \
    --csv_path "$CSV_PATH" \
    --results_dir "$BOOTSTRAP_RESULTS_DIR" \
    --ig_dir "$IG_BOOTSTRAP_DIR" \
    --seed 1

echo "✅ Bootstrap IG分析完成！"
echo "📁 结果保存在: $IG_BOOTSTRAP_DIR"
echo "⏰ 完成时间: $(date)" 