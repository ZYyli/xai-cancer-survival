#!/usr/bin/env python3
"""
增强版特征稳定性分析运行脚本

新增功能:
1. 与模型性能关联分析 - 分析特征稳定性与C-index的相关性
2. 跨XAI方法一致性分析 - 比较4种XAI方法的一致程度

使用方法:
1. 确认路径配置正确
2. 运行: python run_advanced_stability_analysis.py
"""

import os
import sys
from feature_stability_analysis import FeatureStabilityAnalyzer

def main():
    # ===== 配置部分 =====
    
    # 根目录配置
    base_dir = "/home/zuoyiyi/SNN/TCGA"
    output_dir = "/home/zuoyiyi/SNN/TCGA/advanced_stability_analysis"
    
    # XAI方法配置 - 支持四种方法
    xai_methods = ['shap', 'IG', 'LRP', 'PFI']
    
    # 癌症类型配置 - 完整的15种癌症
    cancer_types = [
        'BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC',
        'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC'
    ]
    
    print("🚀 增强版特征稳定性分析")
    print("=" * 50)
    print(f"📁 基础目录: {base_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🔬 XAI方法: {', '.join(xai_methods)}")
    print(f"🎯 癌症类型: {len(cancer_types)} 种")
    print("\n🆕 新增功能:")
    print("   📈 模型性能关联分析")
    print("   🔗 跨XAI方法一致性分析")
    print("   📊 共识特征识别")
    
    # ===== 检查必要文件 =====
    print(f"\n🔍 检查必要文件和目录...")
    
    # 检查各XAI方法的结果目录
    missing_dirs = []
    for method in xai_methods:
        if method.lower() == 'shap':
            method_dir = os.path.join(base_dir, 'shap_individual_analysis')
        else:
            method_dir = os.path.join(base_dir, f'{method.upper()}_results_2')
        
        if not os.path.exists(method_dir):
            missing_dirs.append(method_dir)
        else:
            print(f"   ✅ {method}: {method_dir}")
    
    if missing_dirs:
        print(f"\n❌ 以下目录不存在:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("请确保所有XAI方法的结果都已生成")
        return
    
    # 检查模型性能数据
    performance_dir = os.path.join(base_dir, 'results_2')
    if not os.path.exists(performance_dir):
        print(f"❌ 模型性能数据目录不存在: {performance_dir}")
        return
    else:
        print(f"   ✅ 模型性能数据: {performance_dir}")
    
    # 检查样例癌症的数据完整性
    sample_cancer = cancer_types[0]
    print(f"\n📊 检查 {sample_cancer} 的数据完整性...")
    
    for method in xai_methods:
        if method.lower() == 'shap':
            method_dir = os.path.join(base_dir, 'shap_individual_analysis')
            feature_dir = os.path.join(method_dir, sample_cancer, 'feature_importance')
        else:
            method_dir = os.path.join(base_dir, f'{method.upper()}_results_2')
            feature_subdir = f'{method.lower()}_feature_importance'
            feature_dir = os.path.join(method_dir, sample_cancer, feature_subdir)
        
        if os.path.exists(feature_dir):
            files = [f for f in os.listdir(feature_dir) if f.endswith('.csv')]
            print(f"   {method}: {len(files)} 个特征重要性文件")
        else:
            print(f"   ❌ {method}: 特征重要性目录不存在")
    
    # 检查性能数据
    perf_file = os.path.join(performance_dir, sample_cancer, 'detailed_results.csv')
    if os.path.exists(perf_file):
        import pandas as pd
        df = pd.read_csv(perf_file)
        print(f"   性能数据: {len(df)} 个实验")
    else:
        print(f"   ❌ 性能数据文件不存在")
    
    # ===== 运行分析 =====
    print(f"\n🎯 开始运行增强版稳定性分析...")
    
    try:
        # 创建分析器
        analyzer = FeatureStabilityAnalyzer(
            results_dir=base_dir,
            output_dir=output_dir,
            xai_methods=xai_methods
        )
        
        # 运行完整分析（包含新功能）
        analyzer.run_full_analysis(cancer_types=cancer_types)
        
        print(f"\n🎉 增强版分析完成！")
        print(f"\n📋 主要结果:")
        print(f"   📂 主目录: {output_dir}")
        print(f"   📄 跨癌症汇总: cross_cancer_summary_report.txt")
        print(f"   📊 稳定性数据: cross_cancer_stability_summary.csv")
        print(f"   🔗 跨XAI一致性: cross_xai_consistency_report.txt")
        
        print(f"\n🆕 新功能输出:")
        print(f"   📈 各癌症性能相关性: [CANCER]/[XAI]/performance_correlations.csv")
        print(f"   🔗 跨XAI一致性分析: [CANCER]/cross_xai_analysis/")
        print(f"   📊 共识特征列表: [CANCER]/cross_xai_analysis/consensus_features_top*.csv")
        
        print(f"\n💡 使用建议:")
        print(f"   1. 查看 cross_xai_consistency_report.txt 了解整体一致性")
        print(f"   2. 检查各癌症的 consensus_features_top100.csv 获取稳健特征")
        print(f"   3. 分析 performance_correlations.csv 了解稳定性与性能关系")
        print(f"   4. 使用共识特征进行下游生物学分析")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'itertools', 'collections'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ 缺少以下依赖包: {', '.join(missing)}")
        print(f"请安装: pip install {' '.join(missing)}")
        return False
    
    return True


if __name__ == "__main__":
    print("增强版特征重要性稳定性分析")
    print("=" * 60)
    print("🆕 包含模型性能关联和跨XAI一致性分析")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 运行主程序
    main()




