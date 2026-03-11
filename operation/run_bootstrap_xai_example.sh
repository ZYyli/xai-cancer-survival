#!/bin/bash

# ============================================================
# Bootstrap XAI 特征重要性分析脚本
# ============================================================
# 用途：对100次（或500次）bootstrap训练的模型进行XAI分析
# 支持4种方法：SHAP, IG, LRP, PFI
# 输出路径自动设置为：/home/zuoyiyi/SNN/TCGA/{method}_bootstrap_results/
# ============================================================

# 配置参数
CSV_PATH="/home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_cancer_single"
RESULTS_DIR="/home/zuoyiyi/SNN/TCGA/results_bootstrap_500"  # 或 results_bootstrap
NUM_BOOTSTRAP=500  # 与训练时的bootstrap次数一致
SEED_BASE=1

# ============================================================
# 运行不同的XAI方法
# ============================================================

# 1. SHAP方法
echo "=========================================="
echo "运行 SHAP 分析"
echo "=========================================="
python save_feature_importance_bootstrap.py \
    --csv_path ${CSV_PATH} \
    --results_dir ${RESULTS_DIR} \
    --method shap \
    --num_bootstrap ${NUM_BOOTSTRAP} \
    --seed_base ${SEED_BASE}

# 2. IG方法
echo "=========================================="
echo "运行 IG 分析"
echo "=========================================="
python save_feature_importance_bootstrap.py \
    --csv_path ${CSV_PATH} \
    --results_dir ${RESULTS_DIR} \
    --method ig \
    --num_bootstrap ${NUM_BOOTSTRAP} \
    --seed_base ${SEED_BASE}

# 3. LRP方法
echo "=========================================="
echo "运行 LRP 分析"
echo "=========================================="
python save_feature_importance_bootstrap.py \
    --csv_path ${CSV_PATH} \
    --results_dir ${RESULTS_DIR} \
    --method lrp \
    --num_bootstrap ${NUM_BOOTSTRAP} \
    --seed_base ${SEED_BASE}

# 4. PFI方法
echo "=========================================="
echo "运行 PFI 分析"
echo "=========================================="
python save_feature_importance_bootstrap.py \
    --csv_path ${CSV_PATH} \
    --results_dir ${RESULTS_DIR} \
    --method pfi \
    --num_bootstrap ${NUM_BOOTSTRAP} \
    --seed_base ${SEED_BASE}

echo "=========================================="
echo "所有XAI分析完成！"
echo "=========================================="
echo "结果保存在："
echo "  /home/zuoyiyi/SNN/TCGA/shap_bootstrap_results/"
echo "  /home/zuoyiyi/SNN/TCGA/ig_bootstrap_results/"
echo "  /home/zuoyiyi/SNN/TCGA/lrp_bootstrap_results/"
echo "  /home/zuoyiyi/SNN/TCGA/pfi_bootstrap_results/"
echo ""
echo "文件结构（每种方法）："
echo "  {method}_bootstrap_results/[癌症]/[method]_feature_importance/seed[X]_{method}_ranking.csv"

