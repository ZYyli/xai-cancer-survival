#!/bin/bash

# 运行所有XAI方法的个体分析脚本
# 分析每个 repeat-fold 组合的预后因子数量与 C-index 的相关性
# 使用方法：bash run_all_xai_individual.sh

# 设置基础路径
BASE_DIR="/home/zuoyiyi/SNN/TCGA"
CSV_PATH="${BASE_DIR}/datasets_csv/preprocess_1"
RESULTS_DIR="${BASE_DIR}/results_2"

# XAI方法分析结果保存目录
SHAP_DIR="${BASE_DIR}/shap_individual_analysis"
PFI_DIR="${BASE_DIR}/pfi_individual_analysis"
LRP_DIR="${BASE_DIR}/lrp_individual_analysis"
IG_DIR="${BASE_DIR}/ig_individual_analysis"

# 检查必要目录
echo "检查必要文件和目录..."

if [ ! -d "${CSV_PATH}" ]; then
    echo "错误：数据目录不存在: ${CSV_PATH}"
    exit 1
fi

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "错误：结果目录不存在: ${RESULTS_DIR}"
    exit 1
fi

echo "数据目录: ${CSV_PATH}"
echo "模型结果目录: ${RESULTS_DIR}"
echo "各XAI分析结果保存目录："
echo "  - SHAP: ${SHAP_DIR}"
echo "  - PFI: ${PFI_DIR}" 
echo "  - LRP: ${LRP_DIR}"
echo "  - IG: ${IG_DIR}"

# 创建输出目录
mkdir -p "${SHAP_DIR}" "${PFI_DIR}" "${LRP_DIR}" "${IG_DIR}"

cd "${BASE_DIR}/operation"

echo "开始运行所有XAI方法的个体分析..."
echo "注意：这个过程可能需要很长时间，因为要处理 4种方法 × 15种癌症 × 50个模型"

# 1. 运行 SHAP 分析
echo ""
echo "====================="
echo "1. 运行 SHAP 分析"
echo "====================="
python shap_individual_analysis.py \
    --csv_path "${CSV_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --shap_dir "${SHAP_DIR}" \
    --seed 1

if [ $? -eq 0 ]; then
    echo "✅ SHAP 分析完成"
else
    echo "❌ SHAP 分析失败"
fi

# 2. 运行 PFI 分析
echo ""
echo "====================="
echo "2. 运行 PFI 分析"
echo "====================="
python pfi_individual_analysis.py \
    --csv_path "${CSV_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --pfi_dir "${PFI_DIR}" \
    --seed 1

if [ $? -eq 0 ]; then
    echo "✅ PFI 分析完成"
else
    echo "❌ PFI 分析失败"
fi

# 3. 运行 LRP 分析  
echo ""
echo "====================="
echo "3. 运行 LRP 分析"
echo "====================="
python lrp_individual_analysis.py \
    --csv_path "${CSV_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --lrp_dir "${LRP_DIR}" \
    --seed 1 \
    --epsilon 1e-6

if [ $? -eq 0 ]; then
    echo "✅ LRP 分析完成"
else
    echo "❌ LRP 分析失败"
fi

# 4. 运行 IG 分析
echo ""
echo "====================="
echo "4. 运行 IG 分析"  
echo "====================="
python ig_individual_analysis.py \
    --csv_path "${CSV_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --ig_dir "${IG_DIR}" \
    --seed 1

if [ $? -eq 0 ]; then
    echo "✅ IG 分析完成"
else
    echo "❌ IG 分析失败"
fi

echo ""
echo "====================="
echo "所有XAI分析完成！"
echo "====================="
echo "结果保存位置："
echo "  - SHAP: ${SHAP_DIR}/all_cancers_summary.csv"
echo "  - PFI: ${PFI_DIR}/all_cancers_summary.csv"
echo "  - LRP: ${LRP_DIR}/all_cancers_summary.csv"
echo "  - IG: ${IG_DIR}/all_cancers_summary.csv"
echo ""
echo "每个癌症类型的详细结果可在各自目录的子文件夹中找到" 