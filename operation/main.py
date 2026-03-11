from __future__ import print_function
from functools import partial
import numpy as np
import random
import argparse
import os
import sys
from timeit import default_timer as timer
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

### Internal Imports
from dataset_survival import RNAseqSurvivalDataset
from file_utils import save_pkl
from core_utils import train_loop_survival_omic, validate_survival_omic, summary_survival_omic, train

### PyTorch Imports
import torch
from torch.utils.data import DataLoader, Subset


def seed_worker(seed, worker_id):
    """为DataLoader的工作进程设置随机种子"""
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def set_seed(seed):
    """设置全局随机种子，确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    # 确保CUDA卷积算法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量，确保某些库的确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def split_train_dataset(train_df, val_split_ratio=0.2, seed=42):
    """
    将训练集进一步分割为内部训练集和内部验证集
    
    Args:
        train_df: 原始训练集DataFrame
        val_split_ratio: 验证集比例，默认0.2
        seed: 随机种子
    
    Returns:
        inner_train_df: 内部训练集
        inner_val_df: 内部验证集
    """
    # 按照删失状态分层抽样，确保训练集和验证集中删失样本比例相近
    inner_train_df, inner_val_df = train_test_split(
        train_df, 
        test_size=val_split_ratio,
        stratify=train_df['censorship'],  # 按删失状态分层
        random_state=seed
    )
    
    return inner_train_df.reset_index(drop=True), inner_val_df.reset_index(drop=True)

def cleanup_intermediate_files(results_dir, keep_logs=False, verbose=True):
    """
    清理不必要的中间文件，减少存储空间占用
    
    Args:
        results_dir: 结果目录路径
        keep_logs: 是否保留TensorBoard训练日志
        verbose: 是否显示详细信息
    """
    if verbose:
        print(f"\n开始清理中间文件...")
    
    # 删除训练中间模型
    patterns_to_delete = [
        os.path.join(results_dir, "repeat*_s_*_best_cindex_checkpoint.pt"),
        os.path.join(results_dir, "repeat*_s_*_best_loss_checkpoint.pt")
    ]
    
    if not keep_logs:
        # 删除TensorBoard日志文件
        patterns_to_delete.extend([
            os.path.join(results_dir, "*/events.out.tfevents.*"),
            os.path.join(results_dir, "[0-9]/")  # fold目录
        ])
    
    deleted_files = 0
    deleted_dirs = 0
    
    for pattern in patterns_to_delete:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_files += 1
                    if verbose:
                        print(f"删除文件: {os.path.basename(file_path)}")
                elif os.path.isdir(file_path) and not keep_logs:
                    # 删除空的TensorBoard日志目录
                    import shutil
                    shutil.rmtree(file_path)
                    deleted_dirs += 1
                    if verbose:
                        print(f"删除目录: {os.path.basename(file_path)}")
            except Exception as e:
                if verbose:
                    print(f"⚠️ 删除失败 {file_path}: {e}")
    
    if verbose:
        print(f"\n清理完成！")
        print(f"删除了 {deleted_files} 个文件")
        if not keep_logs and deleted_dirs > 0:
            print(f"删除了 {deleted_dirs} 个目录")
        
        # 显示保留的关键文件
        remaining_models = glob.glob(os.path.join(results_dir, "repeat*_s_*_final_test_model.pt"))
        remaining_results = glob.glob(os.path.join(results_dir, "repeat*_fold*_results.pkl"))
        print(f"保留的核心文件：")
        print(f"  - 最终测试模型: {len(remaining_models)} 个")
        print(f"  - 实验结果文件: {len(remaining_results)} 个")

def main(args):
    """主训练函数，支持5-fold × 10 repeats交叉验证"""
    
    # 创建总结果目录
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    
    # 设置fold范围
    if args.k_start == -1:
        start_fold = 0
    else:
        start_fold = args.k_start
    if args.k_end == -1:
        end_fold = args.k
    else:
        end_fold = args.k_end
    
    # 用于保存所有重复实验的结果
    all_results = []
    all_test_cindexes = []
    
    folds = np.arange(start_fold, end_fold)
    
    print(f"开始 {args.n_repeats} 次重复的 {args.k} 折交叉验证")
    
    ### Start N-Repeats × K-Fold CV Evaluation
    for repeat in range(args.n_repeats):
        print(f"\n{'='*60}")
        print(f"重复实验 {repeat + 1}/{args.n_repeats}")
        print(f"{'='*60}")
        
        # 每次重复使用不同的种子
        current_seed = args.seed + repeat * 1000
        set_seed(current_seed)
        
        # 当前重复实验的结果
        repeat_test_cindexes = []
        
        for i in folds:
            fold_start = timer()
            print(f'\n正在处理 重复{repeat+1} Fold {i}!')
            
            # 构造保存当前fold结果的文件路径
            results_pkl_path = os.path.join(args.results_dir, f'repeat{repeat}_fold{i}_results.pkl')
            
            # 如果结果文件已存在且未设置 overwrite 参数，则跳过
            if os.path.isfile(results_pkl_path) and (not args.overwrite):
                print(f"跳过 重复{repeat} Fold {i}")
                continue
            
            # 读取预分割的训练集和测试集
            train_df = pd.read_csv(f'{args.csv_path}/{args.cancer}_{i}_train.csv')
            test_df = pd.read_csv(f'{args.csv_path}/{args.cancer}_{i}_val.csv')  # 原来的val现在是test
            
            print(f"原始训练集大小: {len(train_df)}")
            print(f"测试集大小: {len(test_df)}")
            
            # 将训练集进一步分割为内部训练集和内部验证集
            inner_train_df, inner_val_df = split_train_dataset(
                train_df, 
                val_split_ratio=args.inner_val_ratio,
                seed=current_seed + i  # 每个fold使用不同的种子
            )
            
            print(f"内部训练集大小: {len(inner_train_df)}")
            print(f"内部验证集大小: {len(inner_val_df)}")
            
            # 保存数据分割索引以支持复现
            split_info = {
                'repeat': repeat,
                'fold': i,
                'seed': current_seed + i,
                'original_train_indices': train_df.index.tolist(),
                'inner_train_indices': inner_train_df.index.tolist(),
                'inner_val_indices': inner_val_df.index.tolist(),
                'test_indices': test_df.index.tolist(),
                'inner_val_ratio': args.inner_val_ratio
            }
            
            split_info_path = os.path.join(args.results_dir, f'repeat{repeat}_fold{i}_split_info.pkl')
            save_pkl(split_info_path, split_info)

            # 创建数据集
            inner_train_dataset = RNAseqSurvivalDataset(data=inner_train_df, label_col='survival_months', seed=current_seed)
            inner_val_dataset = RNAseqSurvivalDataset(data=inner_val_df, label_col='survival_months', seed=current_seed)
            test_dataset = RNAseqSurvivalDataset(data=test_df, label_col='survival_months', seed=current_seed)
            
            print(f'内部训练集特征维度: {inner_train_dataset.features.shape}')
            print(f'内部验证集特征维度: {inner_val_dataset.features.shape}')
            print(f'测试集特征维度: {test_dataset.features.shape}')
            
            # 打印删失情况统计
            print("内部训练集删失情况统计：", inner_train_df['censorship'].value_counts())
            print("内部验证集删失情况统计：", inner_val_df['censorship'].value_counts())
            print("测试集删失情况统计：", test_df['censorship'].value_counts())
            
            # 创建DataLoader
            g = torch.Generator()
            g.manual_seed(current_seed)
            
            inner_train_loader = DataLoader(
                inner_train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=4, 
                worker_init_fn=partial(seed_worker, current_seed), 
                generator=g
            )
            inner_val_loader = DataLoader(
                inner_val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=4, 
                worker_init_fn=partial(seed_worker, current_seed), 
                generator=g
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=4, 
                worker_init_fn=partial(seed_worker, current_seed), 
                generator=g
            )
            
            # 设置基因组特征维度
            args.omic_input_dim = inner_train_dataset.features.shape[1]
            print("基因特征维度：", args.omic_input_dim)
            
            # 训练模型 - 现在传入三个loader
            # inner_train_loader: 用于训练
            # inner_val_loader: 用于早停和模型选择  
            # test_loader: 用于最终评估
            test_results, test_cindex = train(inner_train_loader, inner_val_loader, test_loader, i, args, current_seed, repeat)
            repeat_test_cindexes.append(test_cindex)
            
            # 保存当前fold的结果
            fold_result = {
                'repeat': repeat,
                'fold': i,
                'seed': current_seed,
                'data_split_seed': current_seed + i,
                'test_results': test_results,
                'test_cindex': test_cindex,
                'inner_train_size': len(inner_train_df),
                'inner_val_size': len(inner_val_df),
                'test_size': len(test_df),
                'model_path': os.path.join(args.results_dir, f"repeat{repeat}_s_{i}_final_test_model.pt"),
                'best_cindex_model_path': os.path.join(args.results_dir, f"repeat{repeat}_s_{i}_best_cindex_checkpoint.pt"),
                'best_loss_model_path': os.path.join(args.results_dir, f"repeat{repeat}_s_{i}_best_loss_checkpoint.pt"),
                'split_info_path': split_info_path
            }
            
            save_pkl(results_pkl_path, fold_result)
            
            fold_end = timer()
            print(f'重复{repeat+1} Fold {i} 完成时间: {fold_end - fold_start:.2f} 秒')
            print(f'测试集 C-index: {test_cindex:.4f}')
        
        # 当前重复的汇总结果
        if len(repeat_test_cindexes) > 0:
            avg_cindex = np.mean(repeat_test_cindexes)
            std_cindex = np.std(repeat_test_cindexes)
            print(f"\n重复{repeat+1} 平均测试 C-index: {avg_cindex:.4f} ± {std_cindex:.4f}")
            
            all_test_cindexes.extend(repeat_test_cindexes)
            all_results.append({
                'repeat': repeat,
                'mean_test_cindex': avg_cindex,
                'std_test_cindex': std_cindex,
                'test_cindexes': repeat_test_cindexes
            })
    
    # 所有重复实验的最终汇总
    if len(all_test_cindexes) > 0:
        final_mean = np.mean(all_test_cindexes)
        final_std = np.std(all_test_cindexes)
        
        print(f"\n{'='*60}")
        print(f"最终结果汇总 ({args.n_repeats} × {args.k} = {len(all_test_cindexes)} 次实验)")
        print(f"{'='*60}")
        print(f"测试集 C-index: {final_mean:.4f} ± {final_std:.4f}")
        
        # 保存详细结果
        summary_results = {
            'total_experiments': len(all_test_cindexes),
            'n_repeats': args.n_repeats,
            'n_folds': args.k,
            'final_mean_cindex': final_mean,
            'final_std_cindex': final_std,
            'all_test_cindexes': all_test_cindexes,
            'repeat_results': all_results,
            'args': vars(args)
        }
        
        summary_pkl_path = os.path.join(args.results_dir, 'final_summary.pkl')
        save_pkl(summary_pkl_path, summary_results)
        
        # 保存CSV汇总
        results_df = []
        for result in all_results:
            for fold_idx, cindex in enumerate(result['test_cindexes']):
                results_df.append({
                    'repeat': result['repeat'],
                    'fold': fold_idx,
                    'test_cindex': cindex
                })
        
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(os.path.join(args.results_dir, 'detailed_results.csv'), index=False)
        
        # 保存每个重复的汇总
        repeat_summary = pd.DataFrame([{
            'repeat': r['repeat'],
            'mean_test_cindex': r['mean_test_cindex'],
            'std_test_cindex': r['std_test_cindex']
        } for r in all_results])
        
        # 添加总体汇总行
        final_summary_row = pd.DataFrame({
            'repeat': ['Overall'],
            'mean_test_cindex': [final_mean],
            'std_test_cindex': [final_std]
        })
        repeat_summary = pd.concat([repeat_summary, final_summary_row], ignore_index=True)
        repeat_summary.to_csv(os.path.join(args.results_dir, 'repeat_summary.csv'), index=False)
        
    else:
        print("没有处理任何有效的fold，跳过汇总生成。")

    # 实验完成后进行文件清理
    if args.cleanup_files:
        cleanup_intermediate_files(
            results_dir=args.results_dir, 
            keep_logs=args.keep_logs,
            verbose=True
        )

### Training settings
parser = argparse.ArgumentParser(description='SNN for Survival Analysis with RNAseq Data - Nested CV with Repeats.')

# 数据路径
parser.add_argument('--csv_path', type=str, required=True, help='RNAseq数据csv路径')
parser.add_argument('--seed', type=int, default=1, help='随机种子基数')

# 交叉验证设置
parser.add_argument('--k', type=int, default=5, help='交叉验证折数')
parser.add_argument('--n_repeats', type=int, default=10, help='重复实验次数')
parser.add_argument('--inner_val_ratio', type=float, default=0.2, help='从训练集中分割验证集的比例')
parser.add_argument('--k_start', type=int, default=-1, help='起始fold编号')
parser.add_argument('--k_end', type=int, default=-1, help='结束fold编号（-1表示到最后一折）')
parser.add_argument('--results_dir', type=str, required=True, help='结果保存目录')
parser.add_argument('--cancer', type=str, required=True, help='癌症名字')
parser.add_argument('--log_data', default=True, help='使用tensorboard记录日志')

# 模型配置
parser.add_argument('--model_size_omic', type=str, default='small', help='SNN网络规模')
parser.add_argument('--n_classes', type=int, default=4, help='生存分析分箱数')

# 优化器与训练设置
parser.add_argument('--batch_size', type=int, default=1, help='训练batch size')
parser.add_argument('--max_epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
parser.add_argument('--lambda_reg', type=float, default=1e-5, help='L1正则化系数')
parser.add_argument('--reg', type=float, default=1e-5, help='L2正则化系数')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='删失样本损失权重')
parser.add_argument('--gc', type=int, default=32, help='梯度累积步数')
parser.add_argument('--early_stopping', default=True, help='是否启用早停')

# 文件管理
parser.add_argument('--overwrite', action='store_true', default=False, help='是否覆盖已有实验')
parser.add_argument('--cleanup_files', default=True, help='实验完成后是否清理中间文件')
parser.add_argument('--keep_logs', action='store_true', default=False, help='清理时是否保留训练日志')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    start = timer()
    main(args)
    end = timer()
    print("实验完成!")
    print("脚本结束")
    print('总运行时间: %.2f 秒' % (end - start))