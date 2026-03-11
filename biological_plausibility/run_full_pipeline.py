#!/usr/bin/env python3
"""
XAI 生物学合理性评估 - 一键运行全流程 (方案B: Nested CV)

运行方式:
    # 单个癌种×单个XAI
    python run_full_pipeline.py --cancer BLCA --xai DeepLIFT
    
    # 单个癌种所有XAI
    python run_full_pipeline.py --cancer BLCA
    
    # 所有癌种×所有XAI
    python run_full_pipeline.py --all
    
    # 启用 PubMed 查询（更慢）
    python run_full_pipeline.py --cancer BLCA --use_pubmed
    
    # 只运行可视化
    python run_full_pipeline.py --viz_only
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(Path(__file__).parent))

from config import CANCER_TYPES, XAI_METHODS, OUTPUT_DIR


def run_step(script_name, args=None):
    """运行单个脚本"""
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        print(f"✗ 脚本不存在: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"运行: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 脚本执行失败: {e}")
        return False


def check_databases():
    """检查数据库文件是否存在"""
    from config import DATABASE_DIR
    
    required_files = [
        "oncokb_genes.tsv",
        "cgc_genes.csv",
        "dgidb_interactions.tsv",
        "civic_genes.tsv"  # ClinicalEvidenceSummaries (精确 A-E 等级)
    ]
    
    missing = []
    for f in required_files:
        if not (DATABASE_DIR / f).exists():
            missing.append(f)
    
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="XAI 生物学合理性评估 - 完整流程"
    )
    parser.add_argument('--cancer', type=str, help='指定癌种')
    parser.add_argument('--xai', type=str, help='指定 XAI 方法')
    parser.add_argument('--all', action='store_true', help='处理所有癌种×所有XAI')
    parser.add_argument('--use_pubmed', action='store_true', help='启用 PubMed 查询')
    parser.add_argument('--skip_download', action='store_true', help='跳过数据库下载')
    parser.add_argument('--viz_only', action='store_true', help='只运行可视化')
    parser.add_argument('--stats_only', action='store_true', help='只运行统计分析')
    
    args = parser.parse_args()
    
    print("="*60)
    print("XAI 生物学合理性评估 - 完整流程")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确定要处理的组合
    if args.all:
        mode_str = "所有癌种 × 所有XAI"
        score_args = ["--all"]
    elif args.cancer and args.xai:
        mode_str = f"{args.cancer} × {args.xai}"
        score_args = ["--cancer", args.cancer, "--xai", args.xai]
    elif args.cancer:
        mode_str = f"{args.cancer} × 所有XAI"
        score_args = ["--cancer", args.cancer]
    elif args.xai:
        mode_str = f"所有癌种 × {args.xai}"
        score_args = ["--xai", args.xai]
    else:
        print("请指定 --cancer, --xai, 或 --all")
        return
    
    print(f"处理模式: {mode_str}")
    print(f"方案: Nested CV (5 folds × 10 repeats = 50 models)")
    
    # 只运行可视化
    if args.viz_only:
        print("\n[模式] 只运行可视化")
        run_step("04_visualization.py")
        return
    
    # 只运行统计
    if args.stats_only:
        print("\n[模式] 只运行统计分析")
        run_step("03_statistical_analysis.py")
        return
    
    # Step 1: 下载数据库
    if not args.skip_download:
        missing = check_databases()
        if missing:
            print(f"\n缺少数据库文件: {missing}")
            print("运行数据库下载...")
            run_step("01_download_databases.py")
        else:
            print("\n数据库文件已存在，跳过下载")
    
    # 再次检查
    missing = check_databases()
    if missing:
        print(f"\n⚠ 警告: 以下数据库文件缺失或为模板: {missing}")
        print("  请按照 README.md 说明手动下载完整数据")
        user_input = input("  是否继续运行（使用模板数据）？(y/n): ")
        if user_input.lower() != 'y':
            print("已取消")
            return
    
    # Step 2: 计算基因评分 (Nested CV 方案B)
    print("\n" + "="*60)
    print("Step 2: 计算基因评分 (50个 CV 模型)")
    print("="*60)
    
    if args.use_pubmed:
        score_args.append("--use_pubmed")
    
    success = run_step("02_calculate_gene_scores.py", score_args)
    if not success:
        print("✗ 基因评分计算失败")
        return
    
    # Step 3: 统计分析
    print("\n" + "="*60)
    print("Step 3: 统计分析")
    print("="*60)
    
    success = run_step("03_statistical_analysis.py")
    if not success:
        print("✗ 统计分析失败")
        return
    
    # Step 4: 可视化
    print("\n" + "="*60)
    print("Step 4: 可视化")
    print("="*60)
    
    success = run_step("04_visualization.py")
    
    # 汇总
    print("\n" + "="*60)
    print("流程完成！")
    print("="*60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n输出目录:")
    print(f"  - Nested CV 汇总: outputs/nested_cv_summary.csv")
    print(f"  - 模型分数: outputs/{{cancer}}/{{xai}}/cv_model_scores.csv")
    print(f"  - 基因评分: outputs/{{cancer}}/{{xai}}/all_gene_scores.csv")
    print(f"  - 统计结果: outputs/statistics/")
    print(f"  - 图表: outputs/figures/")


if __name__ == "__main__":
    main()
