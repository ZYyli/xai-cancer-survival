#!/bin/bash

echo "🚀 开始所有Bootstrap XAI可解释性分析..."
echo "时间: $(date)"

# 设置路径
CSV_PATH="/home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_cancer_single"
BOOTSTRAP_RESULTS_DIR="/home/zuoyiyi/SNN/TCGA/results_bootstrap"
BASE_OUTPUT_DIR="/home/zuoyiyi/SNN/TCGA"

# 验证输入路径
if [ ! -d "$CSV_PATH" ]; then
    echo "❌ 错误: 数据路径不存在: $CSV_PATH"
    exit 1
fi

if [ ! -d "$BOOTSTRAP_RESULTS_DIR" ]; then
    echo "❌ 错误: Bootstrap结果路径不存在: $BOOTSTRAP_RESULTS_DIR"
    exit 1
fi

# 创建输出目录
IG_BOOTSTRAP_DIR="${BASE_OUTPUT_DIR}/IG_bootstrap_results"
LRP_BOOTSTRAP_DIR="${BASE_OUTPUT_DIR}/LRP_bootstrap_results"
SHAP_BOOTSTRAP_DIR="${BASE_OUTPUT_DIR}/SHAP_bootstrap_results"
PFI_BOOTSTRAP_DIR="${BASE_OUTPUT_DIR}/PFI_bootstrap_results"

mkdir -p "$IG_BOOTSTRAP_DIR"
mkdir -p "$LRP_BOOTSTRAP_DIR"
mkdir -p "$SHAP_BOOTSTRAP_DIR"
mkdir -p "$PFI_BOOTSTRAP_DIR"

echo "📁 使用路径："
echo "   数据路径: $CSV_PATH"
echo "   Bootstrap结果: $BOOTSTRAP_RESULTS_DIR"
echo "   IG输出目录: $IG_BOOTSTRAP_DIR"
echo "   LRP输出目录: $LRP_BOOTSTRAP_DIR"
echo "   SHAP输出目录: $SHAP_BOOTSTRAP_DIR"
echo "   PFI输出目录: $PFI_BOOTSTRAP_DIR"

echo "🔧 激活pytorch环境..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# 获取用户选择
echo ""
echo "请选择要运行的XAI方法："
echo "1) IG (Integrated Gradients)"
echo "2) LRP (Layer-wise Relevance Propagation)"
echo "3) SHAP (SHapley Additive exPlanations)"
echo "4) PFI (Permutation Feature Importance)"
echo "5) 全部运行"
echo ""
read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "📊 开始IG Bootstrap分析..."
        python ig_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --ig_dir "$IG_BOOTSTRAP_DIR" \
            --seed 1
        ;;
    2)
        echo "📊 开始LRP Bootstrap分析..."
        python lrp_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --lrp_dir "$LRP_BOOTSTRAP_DIR" \
            --seed 1
        ;;
    3)
        echo "📊 开始SHAP Bootstrap分析..."
        python shap_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --shap_dir "$SHAP_BOOTSTRAP_DIR" \
            --seed 1
        ;;
    4)
        echo "📊 开始PFI Bootstrap分析..."
        python pfi_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --pfi_dir "$PFI_BOOTSTRAP_DIR" \
            --seed 1
        ;;
    5)
        echo "📊 开始全部XAI Bootstrap分析..."
        
        echo "正在运行 IG 分析..."
        python ig_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --ig_dir "$IG_BOOTSTRAP_DIR" \
            --seed 1
        
        echo "正在运行 LRP 分析..."
        python lrp_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --lrp_dir "$LRP_BOOTSTRAP_DIR" \
            --seed 1
        
        echo "正在运行 SHAP 分析..."
        python shap_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --shap_dir "$SHAP_BOOTSTRAP_DIR" \
            --seed 1
        
        echo "正在运行 PFI 分析..."
        python pfi_bootstrap_analysis.py \
            --csv_path "$CSV_PATH" \
            --results_dir "$BOOTSTRAP_RESULTS_DIR" \
            --pfi_dir "$PFI_BOOTSTRAP_DIR" \
            --seed 1
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo "✅ Bootstrap XAI分析完成！"
echo "📁 结果保存在相应的输出目录中"
echo "⏰ 完成时间: $(date)" 