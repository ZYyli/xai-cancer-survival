"""
XAI 生物学合理性评估 - 配置文件
方案 B 评分体系
"""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

BASE_DIR = Path(__file__).parent
TCGA_DIR = BASE_DIR.parent
DATABASE_DIR = BASE_DIR / "databases"
OUTPUT_DIR = BASE_DIR / "outputs"

# 创建必要目录
for d in [DATABASE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 癌种与 XAI 方法配置
# ============================================================================

# 癌种与 XAI 方法配置
CANCER_TYPES = [
    "COADREAD", "LUSC", "HNSC", "STAD", "BLCA", "BRCA", "LUAD", 
    "PAAD", "LIHC", "SKCM", "KIRC", "UCEC", "KIRP", "GBMLGG", "LGG"
]

XAI_METHODS = ["LRP", "deepshap", "IG", "shap", "DeepLIFT", "PFI"]

TOP_K = 100  # 默认分析 top100 基因 (未使用，脚本硬编码 [:100])

# Nested CV 结果路径 (保持相对 TCGA 根目录)
XAI_RESULT_DIRS = {
    'DeepLIFT': TCGA_DIR / "DeepLIFT_results_2",
    'deepshap': TCGA_DIR / "deepshap_results_2",
    'shap': TCGA_DIR / "shap_results_2",
    'IG': TCGA_DIR / "IG_results_2",
    'LRP': TCGA_DIR / "LRP_results_2",
    'PFI': TCGA_DIR / "PFI_results_2",
}

# TCGA 癌种到 Open Targets EFO ID 的映射 (用于 Open Targets 分析)
TCGA_TO_EFO = {
    "BLCA": "EFO_0000292",
    "BRCA": "EFO_0000305",
    "COADREAD": "EFO_0000365",
    "GBMLGG": "EFO_0000519",
    "HNSC": "EFO_0000181",
    "KIRC": "EFO_0000349",
    "KIRP": "EFO_0002890",
    "LGG": "EFO_0005543",
    "LIHC": "EFO_0000182",
    "LUAD": "EFO_0000571",
    "LUSC": "EFO_0000708",
    "PAAD": "EFO_0002618",
    "SKCM": "EFO_0000389",
    "STAD": "EFO_0000503",
    "UCEC": "EFO_0001075",
}

# Open Targets GraphQL API 基础 URL
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# ============================================================================
# 统计检验配置
# ============================================================================

BOOTSTRAP_N = 10000  # Bootstrap 重采样次数
ALPHA = 0.05         # 显著性水平
RANDOM_SEED = 42

# ============================================================================
# API 配置
# ============================================================================

# OncoKB API token (申请地址: https://www.oncokb.org/account/register)
ONCOKB_API_TOKEN = os.environ.get("ONCOKB_API_TOKEN", "")
