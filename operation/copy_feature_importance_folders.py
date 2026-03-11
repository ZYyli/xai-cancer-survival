"""
批量复制特征重要性文件夹脚本

功能：将小写命名的XAI bootstrap结果文件夹中的feature_importance子目录
     复制到大写命名的目标文件夹中

例如：
  源: /home/zuoyiyi/SNN/TCGA/pfi_bootstrap_results/BLCA/pfi_feature_importance/
  目标: /home/zuoyiyi/SNN/TCGA/PFI_bootstrap_results/BLCA/pfi_feature_importance/

作者：AI助手
日期：2025-10-23
"""

import os
import shutil
from pathlib import Path

class FeatureImportanceCopier:
    """特征重要性文件夹复制器"""
    
    def __init__(self, base_dir, xai_methods, cancer_types):
        """
        初始化复制器
        
        参数:
        - base_dir: 基础目录路径
        - xai_methods: XAI方法列表（小写），如['pfi', 'lrp', 'ig']
        - cancer_types: 癌症类型列表
        """
        self.base_dir = base_dir
        self.xai_methods = xai_methods
        self.cancer_types = cancer_types
        
        print(f"🔧 初始化特征重要性文件夹复制器")
        print(f"   基础目录: {base_dir}")
        print(f"   XAI方法: {', '.join(xai_methods)}")
        print(f"   癌症类型数: {len(cancer_types)}")
    
    def get_source_path(self, xai_method, cancer):
        """
        获取源路径
        
        参数:
        - xai_method: XAI方法（小写）
        - cancer: 癌症类型
        
        返回: 源目录路径
        """
        return os.path.join(
            self.base_dir,
            f'{xai_method}_bootstrap_results',
            cancer,
            f'{xai_method}_feature_importance'
        )
    
    def get_target_path(self, xai_method, cancer):
        """
        获取目标路径
        
        参数:
        - xai_method: XAI方法（小写）
        - cancer: 癌症类型
        
        返回: 目标目录路径
        """
        #xai_upper = xai_method.upper()
        return os.path.join(
            self.base_dir,
            f'{xai_method}_bootstrap_results_300',
            cancer,
            f'{xai_method}_feature_importance'
        )
    
    def copy_folder(self, source_path, target_path):
        """
        复制文件夹
        
        参数:
        - source_path: 源路径
        - target_path: 目标路径
        
        返回: (success, message)
        """
        try:
            # 检查源路径是否存在
            if not os.path.exists(source_path):
                return False, f"源路径不存在: {source_path}"
            
            # 检查源路径是否为目录
            if not os.path.isdir(source_path):
                return False, f"源路径不是目录: {source_path}"
            
            # 检查源路径是否为空
            if not os.listdir(source_path):
                return False, f"源路径为空目录: {source_path}"
            
            # 创建目标路径的父目录
            target_parent = os.path.dirname(target_path)
            os.makedirs(target_parent, exist_ok=True)
            
            # 如果目标路径已存在，先删除
            if os.path.exists(target_path):
                print(f"   ⚠️  目标路径已存在，将被覆盖: {target_path}")
                shutil.rmtree(target_path)
            
            # 复制文件夹
            shutil.copytree(source_path, target_path)
            
            # 统计文件数
            file_count = sum([len(files) for _, _, files in os.walk(target_path)])
            
            return True, f"成功复制 {file_count} 个文件"
            
        except Exception as e:
            return False, f"复制失败: {str(e)}"
    
    def copy_single_combination(self, xai_method, cancer):
        """
        复制单个XAI方法-癌症组合
        
        参数:
        - xai_method: XAI方法（小写）
        - cancer: 癌症类型
        
        返回: (success, message)
        """
        source_path = self.get_source_path(xai_method, cancer)
        target_path = self.get_target_path(xai_method, cancer)
        
        print(f"\n📂 处理 {xai_method.upper()}-{cancer}")
        print(f"   源: {source_path}")
        print(f"   目标: {target_path}")
        
        success, message = self.copy_folder(source_path, target_path)
        
        if success:
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
        
        return success, message
    
    def copy_all(self, dry_run=False):
        """
        复制所有组合
        
        参数:
        - dry_run: 如果为True，只打印操作而不实际复制
        
        返回: 统计字典
        """
        print(f"\n🚀 开始批量复制{'（试运行模式）' if dry_run else ''}")
        print(f"{'='*60}\n")
        
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }
        
        for xai_method in self.xai_methods:
            print(f"\n{'='*60}")
            print(f"🔬 处理 {xai_method.upper()} 方法")
            print(f"{'='*60}")
            
            method_success = 0
            method_failed = 0
            method_skipped = 0
            
            for cancer in self.cancer_types:
                stats['total'] += 1
                
                if dry_run:
                    # 试运行模式：只检查路径
                    source_path = self.get_source_path(xai_method, cancer)
                    target_path = self.get_target_path(xai_method, cancer)
                    
                    print(f"\n📂 {cancer}")
                    print(f"   源: {source_path}")
                    print(f"   目标: {target_path}")
                    
                    if os.path.exists(source_path):
                        file_count = sum([len(files) for _, _, files in os.walk(source_path)])
                        print(f"   ✓ 源路径存在 ({file_count} 个文件)")
                        stats['success'] += 1
                        method_success += 1
                    else:
                        print(f"   ✗ 源路径不存在")
                        stats['skipped'] += 1
                        method_skipped += 1
                else:
                    # 实际复制
                    success, message = self.copy_single_combination(xai_method, cancer)
                    
                    if success:
                        stats['success'] += 1
                        method_success += 1
                    elif "不存在" in message or "为空" in message:
                        stats['skipped'] += 1
                        method_skipped += 1
                    else:
                        stats['failed'] += 1
                        method_failed += 1
                    
                    stats['details'].append({
                        'xai_method': xai_method,
                        'cancer': cancer,
                        'success': success,
                        'message': message
                    })
            
            # 打印每个方法的统计
            print(f"\n📊 {xai_method.upper()} 方法统计:")
            print(f"   成功: {method_success}/{len(self.cancer_types)}")
            if method_skipped > 0:
                print(f"   跳过: {method_skipped}")
            if method_failed > 0:
                print(f"   失败: {method_failed}")
        
        return stats
    
    def print_summary(self, stats):
        """打印统计摘要"""
        print(f"\n{'='*60}")
        print(f"📊 总体统计摘要")
        print(f"{'='*60}")
        print(f"   总任务数: {stats['total']}")
        print(f"   ✅ 成功: {stats['success']}")
        print(f"   ⏭️  跳过: {stats['skipped']}")
        print(f"   ❌ 失败: {stats['failed']}")
        print(f"   成功率: {stats['success']/stats['total']*100:.1f}%")
        
        # 如果有失败的，列出详情
        if stats['failed'] > 0:
            print(f"\n❌ 失败详情:")
            for detail in stats['details']:
                if not detail['success'] and "不存在" not in detail['message'] and "为空" not in detail['message']:
                    print(f"   - {detail['xai_method'].upper()}-{detail['cancer']}: {detail['message']}")
        
        print(f"\n{'='*60}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量复制特征重要性文件夹")
    parser.add_argument('--base_dir', type=str, 
                       default='/home/zuoyiyi/SNN/TCGA',
                       help='基础目录路径')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['shap'],
                       help='XAI方法列表（小写）')
    parser.add_argument('--dry_run', action='store_true',
                       help='试运行模式（只检查不复制）')
    args = parser.parse_args()
    
    # 癌症类型列表
    cancer_types = [
        'BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC',
        'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC'
    ]
    
    print(f"{'='*60}")
    print(f"🎯 特征重要性文件夹批量复制工具")
    print(f"{'='*60}")
    print(f"配置信息:")
    print(f"  基础目录: {args.base_dir}")
    print(f"  XAI方法: {', '.join([m.upper() for m in args.methods])}")
    print(f"  癌症类型: {len(cancer_types)} 个")
    print(f"  模式: {'试运行（不会实际复制）' if args.dry_run else '实际复制'}")
    print(f"{'='*60}")
    
    # 创建复制器
    copier = FeatureImportanceCopier(
        base_dir=args.base_dir,
        xai_methods=args.methods,
        cancer_types=cancer_types
    )
    
    # 执行复制
    stats = copier.copy_all(dry_run=args.dry_run)
    
    # 打印摘要
    copier.print_summary(stats)
    
    if args.dry_run:
        print(f"\n💡 这是试运行模式，没有实际复制文件")
        print(f"   如需实际复制，请去掉 --dry_run 参数")
    else:
        print(f"\n🎉 复制操作完成！")
        print(f"\n📁 复制后的目录结构示例:")
        for method in args.methods:
            method_upper = method.upper()
            print(f"   {args.base_dir}/{method_upper}_bootstrap_results_300/")
            print(f"   ├── BLCA/")
            print(f"   │   └── {method}_feature_importance/")
            print(f"   │       ├── seed1_{method}_ranking.csv")
            print(f"   │       ├── seed2_{method}_ranking.csv")
            print(f"   │       └── ...")
            print(f"   ├── BRCA/")
            print(f"   └── ...")
            print()


if __name__ == "__main__":
    main()

