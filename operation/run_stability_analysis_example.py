#!/usr/bin/env python3
"""
特征稳定性分析运行示例脚本

使用方法:
1. 修改下方的路径配置
2. 直接运行: python run_stability_analysis_example.py
"""

import os
import sys
from feature_stability_analysis import FeatureStabilityAnalyzer

def main():
    # ===== 配置部分 - 请根据您的实际路径修改 =====
    
    # 结果根目录 - 包含各癌症的特征重要性文件
    # 预期结构: results_dir/CANCER_TYPE/feature_importance/repeat*_fold*_feature_importance_ranking.csv
    results_dir = "/home/zuoyiyi/SNN/TCGA/results/shap_analysis"
    
    # 输出目录 - 稳定性分析结果将保存在此
    output_dir = "/home/zuoyiyi/SNN/TCGA/results/stability_analysis"
    
    # XAI方法列表 - 如果您有多种方法，请添加到列表中
    # 目前只有SHAP的结果，如果有其他方法请修改
    xai_methods = ['shap']  # 例如: ['shap', 'lime', 'ig', 'gradcam']
    
    # 指定分析的癌症类型 - None表示分析所有癌症
    # 也可以指定特定癌症: ['BLCA', 'BRCA', 'LUAD']
    cancer_types = None  # 分析所有癌症
    
    # ===== 检查路径是否存在 =====
    if not os.path.exists(results_dir):
        print(f"❌ 错误：结果目录不存在 - {results_dir}")
        print("请检查路径是否正确")
        return
    
    print(f"🔍 检查结果目录结构...")
    
    # 查看目录结构
    cancer_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
    
    if not cancer_dirs:
        print(f"❌ 错误：在 {results_dir} 中没有找到癌症子目录")
        return
    
    print(f"✅ 发现 {len(cancer_dirs)} 个癌症目录: {', '.join(sorted(cancer_dirs))}")
    
    # 检查特征重要性文件
    sample_cancer = cancer_dirs[0]
    feature_importance_dir = os.path.join(results_dir, sample_cancer, 'feature_importance')
    
    if not os.path.exists(feature_importance_dir):
        print(f"❌ 错误：特征重要性目录不存在")
        print(f"期望路径: {feature_importance_dir}")
        print("请确认您的目录结构是否正确")
        return
    
    feature_files = [f for f in os.listdir(feature_importance_dir) 
                    if f.endswith('_feature_importance_ranking.csv')]
    
    print(f"✅ 在 {sample_cancer} 中发现 {len(feature_files)} 个特征重要性文件")
    
    if len(feature_files) < 10:
        print(f"⚠️ 警告：特征重要性文件数量较少，可能影响分析结果")
    
    # ===== 运行分析 =====
    print(f"\n🚀 开始运行稳定性分析...")
    print(f"   结果目录: {results_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   XAI方法: {xai_methods}")
    
    try:
        # 创建分析器
        analyzer = FeatureStabilityAnalyzer(
            results_dir=results_dir,
            output_dir=output_dir,
            xai_methods=xai_methods
        )
        
        # 运行完整分析
        analyzer.run_full_analysis(cancer_types=cancer_types)
        
        print(f"\n🎉 分析完成！")
        print(f"📂 查看结果: {output_dir}")
        print(f"\n主要输出文件:")
        print(f"   📋 cross_cancer_summary_report.txt - 汇总文本报告")
        print(f"   📊 cross_cancer_stability_summary.csv - 汇总数据表")
        print(f"   🎨 cross_cancer_stability_comparison.png - 汇总对比图")
        print(f"\n各癌症详细结果:")
        print(f"   📁 {output_dir}/[CANCER]/[XAI_METHOD]/")
        print(f"      ├── stability_summary_report.csv")
        print(f"      ├── stability_analysis_report.txt")
        print(f"      ├── overlap_curves.png")
        print(f"      └── stability_distribution_top*.png")
        
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
    print("特征重要性稳定性分析")
    print("=" * 40)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 运行主程序
    main()


