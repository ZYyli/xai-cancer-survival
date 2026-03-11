#!/bin/bash

# 完整的XAI分析流程
# 包括：4种XAI方法的个体分析 + 跨方法比较
# 使用方法：bash run_complete_xai_analysis.sh

echo "🚀 开始完整的XAI分析流程"
echo "包括：SHAP、PFI、LRP、IG 四种方法的个体分析 + 跨方法性能比较"
echo ""

# 设置基础路径
BASE_DIR="/home/zuoyiyi/SNN/TCGA"

# 步骤1：运行所有XAI方法的个体分析
echo "步骤1：运行所有XAI方法的个体分析"
echo "======================================"
bash "${BASE_DIR}/operation/run_all_xai_individual.sh"

# 检查步骤1是否成功
if [ $? -eq 0 ]; then
    echo "✅ 步骤1完成：所有XAI方法个体分析完成"
else
    echo "❌ 步骤1失败：XAI个体分析出现错误"
    exit 1
fi

echo ""

# 步骤2：XAI方法跨方法比较分析
echo "步骤2：XAI方法跨方法比较分析"
echo "======================================"

COMPARISON_DIR="${BASE_DIR}/xai_methods_comparison"
mkdir -p "${COMPARISON_DIR}"

cd "${BASE_DIR}/operation"

python compare_xai_methods.py \
    --base_dir "${BASE_DIR}" \
    --output_dir "${COMPARISON_DIR}"

# 检查步骤2是否成功
if [ $? -eq 0 ]; then
    echo "✅ 步骤2完成：XAI方法比较分析完成"
else
    echo "❌ 步骤2失败：XAI方法比较分析出现错误"
    exit 1
fi

echo ""
echo "🎉 完整的XAI分析流程已完成！"
echo ""
echo "结果保存位置："
echo "=================================================="
echo "个体分析结果："
echo "  - SHAP: ${BASE_DIR}/shap_individual_analysis/"
echo "  - PFI:  ${BASE_DIR}/pfi_individual_analysis/"
echo "  - LRP:  ${BASE_DIR}/lrp_individual_analysis/"
echo "  - IG:   ${BASE_DIR}/ig_individual_analysis/"
echo ""
echo "跨方法比较结果："
echo "  - ${COMPARISON_DIR}/"
echo ""
echo "关键结果文件："
echo "  - 各方法汇总: */all_cancers_summary.csv"
echo "  - 各癌症详情: */{CANCER}/detailed_results.csv"
echo "  - 方法比较汇总: ${COMPARISON_DIR}/xai_methods_summary_table.csv"
echo "  - 性能比较图: ${COMPARISON_DIR}/xai_methods_comparison_boxplots.png"
echo "  - 相关性热图: ${COMPARISON_DIR}/correlation_heatmap_by_cancer.png"
echo "=================================================="

echo ""
echo "📊 分析说明："
echo "1. 每种XAI方法分析了50个实验（10 repeat × 5 fold）"
echo "2. 识别的预后因子 = 有明确生物学意义的driver + protector因子"
echo "3. 相关性分析评估C-index与预后因子数量的关系"
echo "4. 跨方法比较帮助选择最适合的XAI方法" 