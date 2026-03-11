#!/usr/bin/env python3
"""
生物学合理性验证 - 四分类统计 (Four-Class Database Validation)

对每种癌症的每个模型的 Top 100 关键基因，在四个数据库中进行四分类验证：
1. OncoKB - 癌症基因临床分级
2. DGIdb - 药物-基因互作 (仅二分类)
3. Open Targets Platform - 基因-疾病关联
4. CancerMine - 文献挖掘癌症基因

四分类定义:
- same_only: 基因仅在当前癌症类型有记录（无其他癌种）
- same_and_other: 基因在当前癌症类型有记录，同时也在其他癌种有记录
- other_only: 基因仅在其他癌症类型有记录（当前癌种无记录）
- not_supported: 基因在数据库中没有任何癌症关联记录

输出指标:
- 各数据库四分类计数
- 广义当前癌种支持 (same_any = same_only + same_and_other)
- hit 指标:
  * hit_same_any: 对当前癌种有支持的命中数

使用方法:
    python 02_calculate_gene_scores.py --cancer BLCA --xai DeepLIFT
    python 02_calculate_gene_scores.py --all  # 运行所有组合
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加脚本目录和父目录到路径
sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_loader import DatabaseLoader
from config import (
    BASE_DIR, TCGA_DIR, DATABASE_DIR, OUTPUT_DIR,
    CANCER_TYPES, XAI_METHODS,
    XAI_RESULT_DIRS
)

DBS_FOR_VIS = ['oncokb', 'dgidb', 'opentargets', 'cancermine']

# ============================================================================
# 四分类验证器
# ============================================================================

class BooleanValidator:
    """
    四分类生物学验证器
    
    对每个基因在每个数据库中进行四分类:
    - same_only: 基因仅在当前癌症类型有记录（无其他癌种）
    - same_and_other: 基因在当前癌症类型有记录，同时也在其他癌种有记录
    - other_only: 基因仅在其他癌症类型有记录（当前癌种无记录）
    - not_supported: 基因在数据库中没有任何癌症关联记录
    """
    
    def __init__(self, db_loader: DatabaseLoader):
        self.db = db_loader
    
    def validate_gene(self, gene: str, cancer_type: str) -> Dict:
        """
        对单个基因进行四分类验证
        
        Returns:
            包含各数据库四分类结果的字典
        """
        gene_upper = gene.upper()
        
        # 获取所有数据库的四分类结果
        classifications = self.db.get_all_classifications(gene_upper, cancer_type)
        
        result = {
            'gene': gene_upper,
            'cancer_type': cancer_type,
        }
        
        # 各数据库的四分类
        for db_name, classification in classifications.items():
            result[f'{db_name}_class'] = classification
        
        # 计算跨库统计
        # n_same_any: 在多少个库中对当前癌种有支持 (same_only 或 same_and_other)
        result['n_same_any'] = sum(
            1 for c in classifications.values() 
            if c in ['same_only', 'same_and_other']
        )
        
        # n_other_any: 在多少个库中对其他癌种有记录 (same_and_other 或 other_only)
        result['n_other_any'] = sum(
            1 for c in classifications.values() 
            if c in ['same_and_other', 'other_only']
        )
        
        # 精细分类计数（用于后续分析）
        result['n_same_only'] = sum(1 for c in classifications.values() if c == 'same_only')
        result['n_same_and_other'] = sum(1 for c in classifications.values() if c == 'same_and_other')
        result['n_other_only'] = sum(1 for c in classifications.values() if c == 'other_only')
        result['n_not_supported'] = sum(1 for c in classifications.values() if c == 'not_supported')
        
        return result
    
    def validate_gene_list(self, genes: List[str], cancer_type: str, 
                           show_progress: bool = True) -> pd.DataFrame:
        """对基因列表进行四分类验证"""
        results = []
        iterator = tqdm(genes, desc="  验证中", leave=False) if show_progress else genes
        
        for gene in iterator:
            result = self.validate_gene(gene, cancer_type)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def calculate_hit_statistics(self, validation_df: pd.DataFrame) -> Dict:
        """
        计算四分类统计
        
        Returns:
            包含各数据库四分类计数、三 hit 指标和支持度统计的字典
        """
        n_genes = len(validation_df)
        
        stats = {'n_genes': n_genes}

        dbs = []
        for col in validation_df.columns:
            if col.endswith('_class'):
                dbs.append(col.replace('_class', ''))
        dbs = sorted(set(dbs))

        for db in dbs:
            col = f'{db}_class'
            stats[f'{db}_same_only'] = int((validation_df[col] == 'same_only').sum())
            stats[f'{db}_same_and_other'] = int((validation_df[col] == 'same_and_other').sum())
            stats[f'{db}_other_only'] = int((validation_df[col] == 'other_only').sum())
            stats[f'{db}_not_supported'] = int((validation_df[col] == 'not_supported').sum())

            stats[f'{db}_hit_same_any'] = stats[f'{db}_same_only'] + stats[f'{db}_same_and_other']

        stats['total_hit_same_any'] = int(
            sum(stats.get(f'{db}_hit_same_any', 0) for db in DBS_FOR_VIS)
        )

        return stats

# ============================================================================
# 数据加载
# ============================================================================

def load_xai_results(cancer_type: str, xai_method: str) -> Optional[List[Dict]]:
    """
    加载 XAI 结果
    输入文件: {XAI}_results_2/{cancer}/{cancer}_complete_results.pkl
    """
    result_dir = XAI_RESULT_DIRS.get(xai_method)
    if not result_dir:
        print(f"  ⚠ 未找到 {xai_method} 的结果目录")
        return None
    
    pkl_path = result_dir / cancer_type / f"{cancer_type}_complete_results.pkl"
    
    if not pkl_path.exists():
        print(f"  ⚠ 文件不存在: {pkl_path}")
        return None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  输入文件: {pkl_path}")
    return data


def extract_top100_genes(result: Dict) -> List[str]:
    """从结果中提取 top100 基因名"""
    prognostic_factors = result.get('prognostic_factors', [])
    
    genes = []
    for factor in prognostic_factors:
        gene_name = factor.get('gene', '')
        if isinstance(gene_name, str):
            # 去除后缀，标准化为 HUGO Symbol
            gene = gene_name.replace('_rnaseq', '').replace('_rppa', '').upper()
            genes.append(gene)
    
    return genes[:100]  # 确保最多100个

# ============================================================================
# 主处理函数
# ============================================================================

def process_cancer_xai(cancer_type: str, xai_method: str, 
                       validator: BooleanValidator) -> Optional[List[Dict]]:
    """
    处理单个癌种×XAI组合的四分类验证
    
    对每个模型独立分析其 Top 100 基因，进行四分类验证和三 hit 统计
    
    输出:
        - outputs/{cancer}/{xai}/gene_validation.csv (所有模型的基因级别验证)
        - outputs/{cancer}/{xai}/model_statistics.csv (每个模型的统计)
    
    Returns:
        包含每个模型统计结果的列表
    """
    print(f"\n处理: {cancer_type} - {xai_method}")
    
    # 加载 XAI 结果
    results = load_xai_results(cancer_type, xai_method)
    if results is None:
        return None
    
    n_models = len(results)
    print(f"  加载了 {n_models} 个模型结果")
    
    # 对每个模型独立验证
    all_gene_validations = []
    model_stats_list = []
    
    for model_idx, result in enumerate(tqdm(results, desc="  模型验证", leave=False)):
        # 提取该模型的 Top 100 基因
        top100_genes = extract_top100_genes(result)
        
        if len(top100_genes) == 0:
            continue
        
        # 布尔验证
        validation_df = validator.validate_gene_list(top100_genes, cancer_type, show_progress=False)
        validation_df['model_idx'] = model_idx
        all_gene_validations.append(validation_df)
        
        # 计算该模型的统计
        stats = validator.calculate_hit_statistics(validation_df)
        stats['model_idx'] = model_idx
        stats['cancer_type'] = cancer_type
        stats['xai_method'] = xai_method
        model_stats_list.append(stats)
    
    if not model_stats_list:
        print(f"  ⚠ 无有效模型结果")
        return None
    
    # 保存结果
    output_dir = OUTPUT_DIR / cancer_type / xai_method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存基因级别验证结果 (所有模型)
    all_genes_df = pd.concat(all_gene_validations, ignore_index=True)
    all_genes_df.to_csv(output_dir / "gene_validation.csv", index=False)
    print(f"  输出: {output_dir / 'gene_validation.csv'} ({len(all_genes_df)} 行)")
    
    # 保存模型级别统计
    model_stats_df = pd.DataFrame(model_stats_list)
    model_stats_df.to_csv(output_dir / "model_statistics.csv", index=False)
    print(f"  输出: {output_dir / 'model_statistics.csv'} ({len(model_stats_df)} 行)")
    
    return model_stats_list


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='生物学合理性验证 - 四分类统计')
    parser.add_argument('--cancer', type=str, help='指定癌症类型')
    parser.add_argument('--xai', type=str, help='指定 XAI 方法')
    parser.add_argument('--all', action='store_true', help='运行所有癌症类型和 XAI 方法')
    args = parser.parse_args()
    
    print("=" * 60)
    print("生物学合理性验证 - 四分类统计")
    print("=" * 60)
    print("\n四分类定义:")
    print("  - same_only: 基因仅在当前癌症类型有记录（无其他癌种）")
    print("  - same_and_other: 基因在当前癌症类型有记录，同时也在其他癌种有记录")
    print("  - other_only: 基因仅在其他癌症类型有记录（当前癌种无记录）")
    print("  - not_supported: 基因在数据库中没有任何癌症关联记录")
    print("\n三 hit 指标:")
    print("  - hit_same_any: 对当前癌种有支持 (same_only + same_and_other)")
    print("  - hit_other_any: 对其他癌种有记录 (same_and_other + other_only)")
    print("  - hit_any_cancer: 对任意癌种有记录 (same_only + same_and_other + other_only)")
    print("\n评估数据库: OncoKB, DGIdb*, OpenTargets, CancerMine")
    print("  (* DGIdb 为药物-基因数据库，无癌症类型信息，仅支持二分类)")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据库
    print("\n加载数据库...")
    db_loader = DatabaseLoader()
    validator = BooleanValidator(db_loader)
    
    # 确定要处理的组合
    if args.all:
        combinations = [(c, x) for c in CANCER_TYPES for x in XAI_METHODS]
    elif args.cancer and args.xai:
        combinations = [(args.cancer, args.xai)]
    elif args.cancer:
        combinations = [(args.cancer, x) for x in XAI_METHODS]
    elif args.xai:
        combinations = [(c, args.xai) for c in CANCER_TYPES]
    else:
        print("\n请指定 --cancer, --xai, 或 --all")
        print("示例:")
        print("  python 02_calculate_gene_scores.py --cancer BRCA --xai DeepLIFT")
        print("  python 02_calculate_gene_scores.py --all")
        print("  python 02_calculate_gene_scores.py --cancer BRCA")
        return
    
    print(f"\n将处理 {len(combinations)} 个组合...")
    
    all_model_stats = []
    for cancer_type, xai_method in combinations:
        model_stats_list = process_cancer_xai(cancer_type, xai_method, validator)
        if model_stats_list:
            all_model_stats.extend(model_stats_list)

    if all_model_stats:
        all_stats_df = pd.DataFrame(all_model_stats)
        out_path = OUTPUT_DIR / "all_model_statistics.csv"
        all_stats_df.to_csv(out_path, index=False)
        print(f"\n已保存: {out_path} ({len(all_stats_df)} 行)")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("主要输出文件:")
    print("  - all_model_statistics.csv  (每个模型的统计，用于箱线图)")

if __name__ == "__main__":
    main()
