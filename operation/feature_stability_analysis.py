"""
特征重要性稳定性分析脚本

功能：分析同一癌症内、同一XAI方法下，跨不同模型的Top-k特征稳定性
作者：AI助手
日期：2025-10-14
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from itertools import combinations
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class FeatureStabilityAnalyzer:
    """特征稳定性分析器"""
    
    def __init__(self, results_dir, output_dir, xai_methods=None):
        """
        初始化分析器
        
        参数:
        - results_dir: 结果根目录
        - output_dir: 输出目录
        - xai_methods: XAI方法列表，如['shap', 'ig', 'lrp', 'pfi']
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.xai_methods = xai_methods
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置参数（集中管理，便于修改）
        self.top_k_values = [20, 50, 100, 200, 300]  # 不同的Top-k值
        self.total_features = 2000  # 总特征数
        self.stability_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]  # 稳定特征阈值列表（可同时分析多个阈值）
        self.stability_threshold = [0.75, 0.7]  # 可以是单个值如 0.75，或列表如 [0.75, 0.8]
        self.random_simulations = 1000  # 随机基线模拟次数
        self.min_rankings_required = 3  # 最少需要的排名文件数
        self.results_subdir = 'results_2'  # 结果子目录名
        self.n_repeats = 10  # 重复实验次数
        self.n_folds = 5  # 交叉验证折数
        
        print(f"🔧 初始化稳定性分析器")
        print(f"   结果目录: {results_dir}")
        print(f"   输出目录: {output_dir}")
        print(f"   XAI方法: {', '.join(self.xai_methods)}")
        print(f"   Top-k值: {self.top_k_values}")
        # 规范化 stability_threshold：如果是列表，使用第一个值；如果是单个值，保持不变
        if isinstance(self.stability_threshold, (list, tuple)):
            if len(self.stability_threshold) == 0:
                # 如果列表为空，使用 stability_thresholds 的第一个值
                self._default_stability_threshold = self.stability_thresholds[0]
                self._default_stability_thresholds = self.stability_thresholds
            else:
                self._default_stability_threshold = self.stability_threshold[0]
                self._default_stability_thresholds = list(self.stability_threshold)
        else:
            self._default_stability_threshold = self.stability_threshold
            self._default_stability_thresholds = [self.stability_threshold]
        
        print(f"   稳定性阈值: {self.stability_thresholds} (默认: {self._default_stability_threshold})")

    def _get_default_threshold(self):
        """获取默认阈值（单个值）"""
        return self._default_stability_threshold
    
    def _get_default_thresholds(self):
        """获取默认阈值列表"""
        return self._default_stability_thresholds

    def _ensure_dir(self, dir_path):
        """
        确保目录存在（辅助方法）
        
        参数:
        - dir_path: 目录路径
        
        返回: 目录路径（便于链式调用）
        """
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def _build_xai_results_path(self, xai_method, cancer_type, subpath=''):
        """
        统一的XAI结果路径构建方法
        
        参数:
        - xai_method: XAI方法名
        - cancer_type: 癌症类型
        - subpath: 子路径（可选）
        
        返回: 完整路径
        """
        xai_dir = f'{xai_method}_results_2'
        base_path = os.path.join(self.results_dir, xai_dir, cancer_type)
        if subpath:
            return os.path.join(base_path, subpath)
        return base_path
    
    def _build_feature_importance_path(self, xai_method, cancer_type):
        """
        构建特征重要性文件路径
        
        返回: (目录路径, 文件名模式)
        """
        # 统一小写处理
        method = xai_method.lower()

        # 定义映射（可扩展）
        valid_methods = ['deepshap', 'shap', 'ig', 'lrp', 'deeplift', 'pfi']
        if method not in valid_methods:
            raise ValueError(f"未知的 XAI 方法: {xai_method}")

        # 构建路径与文件模式
        subdir = f"{method}_feature_importance"
        pattern = f"repeat{{}}_fold{{}}_{method}_feature_importance_ranking.csv"

        # 调用已有函数拼接完整路径
        feature_dir = self._build_xai_results_path(xai_method, cancer_type, subdir)
        return feature_dir, pattern

    def _calculate_stats_summary(self, values):
        """
        统一的统计指标计算方法
        
        参数:
        - values: 数值列表
        
        返回: 包含统计指标的字典
        """
        valid_values = [v for v in values if not np.isnan(v)]
        
        if not valid_values:
            return {
                'mean': np.nan, 'std': np.nan, 'median': np.nan,
                'q25': np.nan, 'q75': np.nan, 'min': np.nan, 'max': np.nan
            }
        
        return {
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'median': np.median(valid_values),
            'q25': np.percentile(valid_values, 25),
            'q75': np.percentile(valid_values, 75),
            'min': np.min(valid_values),
            'max': np.max(valid_values)
        }
    
    def load_feature_rankings(self, cancer_type, xai_method):
        """
        加载指定癌症和XAI方法的所有特征排名文件
        
        返回: dict {(repeat, fold): DataFrame}
        """
        rankings = {}
        # 使用统一的路径构建方法
        feature_dir, file_pattern = self._build_feature_importance_path(xai_method, cancer_type)
        
        if not os.path.exists(feature_dir):
            print(f"⚠️ 目录不存在: {feature_dir}")
            return rankings
        
        # 加载所有repeat-fold组合
        success_count = 0
        for repeat in range(self.n_repeats):
            for fold in range(self.n_folds):
                file_path = os.path.join(feature_dir, file_pattern.format(repeat, fold))
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # 验证必要列存在
                        required_cols = ['feature_name', 'importance_score', 'rank']
                        if all(col in df.columns for col in required_cols):
                            # 按重要性得分排序（确保排名正确）
                            df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
                            df['rank'] = range(1, len(df) + 1)
                            
                            rankings[(repeat, fold)] = df
                            success_count += 1
                        else:
                            print(f"⚠️ 文件缺少必要列: {file_path}")
                    
                    except Exception as e:
                        print(f"❌ 加载文件失败: {file_path}, 错误: {e}")
        
        total_expected = self.n_repeats * self.n_folds
        print(f"✅ {cancer_type}-{xai_method}: 成功加载 {success_count}/{total_expected} 个排名文件")
        return rankings
    
    def calculate_kuncheva_index(self, set1, set2, k, m):
        """
        计算Kuncheva指数（机会校正的重叠度指标）
        
        参数:
        - set1, set2: 两个特征集合
        - k: Top-k大小
        - m: 总特征数
        """
        intersection = len(set1.intersection(set2))
        expected_overlap = k * k / m
        denominator = k - expected_overlap
        
        # 避免分母为0或接近0
        if abs(denominator) < 1e-10:
            return 1.0 if intersection == k else 0.0
        
        return (intersection - expected_overlap) / denominator
    
    def calculate_rank_correlation(self, rank1, rank2):
        """计算排名相关性（主要是Spearman）"""
        # 找到共同特征
        common_features = set(rank1).intersection(set(rank2))
        
        if len(common_features) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        # 构建排名向量
        ranks1 = [rank1.index(f) + 1 for f in common_features]
        ranks2 = [rank2.index(f) + 1 for f in common_features]
        
        # 计算相关性
        spearman_r, spearman_p = stats.spearmanr(ranks1, ranks2)
        # kendall_tau, kendall_p = stats.kendalltau(ranks1, ranks2)  # 删除：冗余指标，节省计算时间
        
        return spearman_r, spearman_p, np.nan, np.nan  # kendall结果用nan替代
    
    def analyze_pairwise_stability(self, rankings, k):
        """
        计算成对稳定性指标
        
        返回: dict 包含各种指标的列表
        """
        results = {
            'kuncheva': [],
            'spearman_r': [],
            'spearman_p': [],
            'pair_info': []  # 存储pair信息用于分层分析
        }
        
        # 获取所有ranking对
        keys = list(rankings.keys())
        
        for i, j in itertools.combinations(range(len(keys)), 2):
            key1, key2 = keys[i], keys[j]
            df1, df2 = rankings[key1], rankings[key2]
            
            # 提取Top-k特征
            top_k_1 = set(df1.head(k)['feature_name'].tolist())
            top_k_2 = set(df2.head(k)['feature_name'].tolist())
            
            # 计算各种指标
            kuncheva = self.calculate_kuncheva_index(top_k_1, top_k_2, k, self.total_features)
            
            spear_r, spear_p, _, _ = self.calculate_rank_correlation(
                df1['feature_name'].tolist(), df2['feature_name'].tolist()
            )
            
            # 存储结果
            results['kuncheva'].append(kuncheva)
            results['spearman_r'].append(spear_r)
            results['spearman_p'].append(spear_p)
            
            # 记录pair信息（用于分层分析）
            repeat1, fold1 = key1
            repeat2, fold2 = key2
            pair_type = 'same_fold' if fold1 == fold2 else 'cross_fold'
            results['pair_info'].append({
                'key1': key1,
                'key2': key2,
                'type': pair_type
            })
        
        return results
    
    def calculate_feature_frequency(self, rankings, k):
        """计算特征在Top-k中的出现频次"""
        feature_counts = Counter()
        
        for df in rankings.values():
            top_k_features = df.head(k)['feature_name'].tolist()
            feature_counts.update(top_k_features)
        
        # 转换为频次（比例）
        total_rankings = len(rankings)
        feature_freq = {
            feature: count / total_rankings 
            for feature, count in feature_counts.items()
        }
        
        return feature_freq
    
    def calculate_random_baseline(self, k, m, n_simulations=None):
        """
        计算随机基线
        
        参数:
        - k: Top-k大小
        - m: 总特征数
        - n_simulations: 模拟次数（默认使用配置值）
        """
        if n_simulations is None:
            n_simulations = self.random_simulations
        random_kunchevas = []
        
        for _ in range(n_simulations):
            # 随机选择两个Top-k集合
            set1 = set(np.random.choice(m, k, replace=False))
            set2 = set(np.random.choice(m, k, replace=False))
            kuncheva = self.calculate_kuncheva_index(set1, set2, k, m)
            random_kunchevas.append(kuncheva)
        
        return {
            'kuncheva_mean': np.mean(random_kunchevas),
            'kuncheva_std': np.std(random_kunchevas)
        }
    
    def create_stability_summary(self, pairwise_results):
        """创建稳定性指标摘要（使用统一的统计计算方法）"""
        summary = {}
        
        for metric in ['kuncheva', 'spearman_r']:
            # 使用统一的统计计算方法
            stats = self._calculate_stats_summary(pairwise_results[metric])
            
            # 将统计结果添加到摘要中
            for stat_name, stat_value in stats.items():
                summary[f'{metric}_{stat_name}'] = stat_value
        
        return summary
    
    def create_stratified_analysis(self, pairwise_results):
        """创建分层分析（同fold内 vs 跨fold）"""
        same_fold_indices = [
            i for i, info in enumerate(pairwise_results['pair_info'])
            if info['type'] == 'same_fold'
        ]
        cross_fold_indices = [
            i for i, info in enumerate(pairwise_results['pair_info'])
            if info['type'] == 'cross_fold'
        ]
        
        stratified = {}
        
        for metric in ['kuncheva']:
            values = pairwise_results[metric]
            
            same_fold_values = [values[i] for i in same_fold_indices if not np.isnan(values[i])]
            cross_fold_values = [values[i] for i in cross_fold_indices if not np.isnan(values[i])]
            
            stratified[f'{metric}_same_fold'] = {
                'mean': np.mean(same_fold_values) if same_fold_values else np.nan,
                'std': np.std(same_fold_values) if same_fold_values else np.nan,
                'n': len(same_fold_values)
            }
            
            stratified[f'{metric}_cross_fold'] = {
                'mean': np.mean(cross_fold_values) if cross_fold_values else np.nan,
                'std': np.std(cross_fold_values) if cross_fold_values else np.nan,
                'n': len(cross_fold_values)
            }
        
        return stratified
    
    def _save_pairwise_raw_data(self, cancer_type, xai_method, pairwise_results, k, output_dir):
        """
        保存成对比较的原始数据（用于高质量可视化）
        
        参数:
        - cancer_type: 癌症类型
        - xai_method: XAI方法
        - pairwise_results: 成对比较结果
        - k: Top-k值
        - output_dir: 输出目录
        """
        # 构建原始数据DataFrame
        raw_data = []
        
        for i, pair_info in enumerate(pairwise_results['pair_info']):
            row = {
                'cancer_type': cancer_type,
                'xai_method': xai_method,
                'top_k': k,
                'pair_id': i + 1,
                'repeat1': pair_info['key1'][0],
                'fold1': pair_info['key1'][1],
                'repeat2': pair_info['key2'][0],
                'fold2': pair_info['key2'][1],
                'pair_type': pair_info['type'],  # 'same_fold' or 'cross_fold'
                'kuncheva': pairwise_results['kuncheva'][i],
                'spearman_r': pairwise_results['spearman_r'][i],
                'spearman_p': pairwise_results['spearman_p'][i]
            }
            raw_data.append(row)
        
        # 保存为CSV
        if raw_data:
            df = pd.DataFrame(raw_data)
            csv_path = os.path.join(output_dir, f'pairwise_stability_raw_top{k}.csv')
            df.to_csv(csv_path, index=False)
            print(f"   ✅ 保存成对比较原始数据: pairwise_stability_raw_top{k}.csv ({len(raw_data)} pairs)")
    
    def load_model_performance_data(self, cancer_type):
        """
        加载模型性能数据（C-index）用于性能关联分析
        
        返回: dict {(repeat, fold): cindex}
        """
        performance_data = {}
        
        # 使用统一的路径构建方法
        results_file = os.path.join(self.results_dir, self.results_subdir, cancer_type, 'detailed_results.csv')
        
        if not os.path.exists(results_file):
            print(f"⚠️ 性能数据文件不存在: {results_file}")
            return performance_data
        
        try:
            df = pd.read_csv(results_file)
            
            # 验证必要列存在
            if not all(col in df.columns for col in ['repeat', 'fold', 'test_cindex']):
                print(f"⚠️ 性能数据文件缺少必要列: {results_file}")
                return performance_data
            
            # 构建性能数据字典
            for _, row in df.iterrows():
                repeat = int(row['repeat'])
                fold = int(row['fold'])
                cindex = float(row['test_cindex'])
                performance_data[(repeat, fold)] = cindex
                
            print(f"✅ 成功加载 {cancer_type} 的性能数据: {len(performance_data)} 个实验")
            
        except Exception as e:
            print(f"❌ 加载性能数据失败: {results_file}, 错误: {e}")
        
        return performance_data
    
    def analyze_stability_performance_correlation(self, cancer_type, xai_method, stability_results, performance_data):
        """
        分析稳定性与模型性能的相关性
        
        参数:
        - stability_results: 稳定性分析结果
        - performance_data: 模型性能数据
        
        返回: dict 包含相关性分析结果
        """
        print(f"📈 分析 {cancer_type}-{xai_method} 的稳定性与性能相关性...")
        
        correlations = {}
        
        for k in self.top_k_values:
            if k not in stability_results:
                continue
            
            # 提取每个实验的稳定性指标
            experiment_data = []
            
            for (repeat, fold), ranking_df in stability_results['rankings'].items():
                if (repeat, fold) not in performance_data:
                    continue
                
                cindex = performance_data[(repeat, fold)]
                
                # 计算该实验与其他实验的平均稳定性
                other_experiments = [(r, f) for (r, f) in stability_results['rankings'].keys() 
                                   if (r, f) != (repeat, fold)]
                
                if len(other_experiments) == 0:
                    continue
                
                # 计算与其他实验的平均Kuncheva稳定性
                top_k_current = set(ranking_df.head(k)['feature_name'].tolist())
                kuncheva_scores = []
                
                for other_repeat, other_fold in other_experiments:
                    other_df = stability_results['rankings'][(other_repeat, other_fold)]
                    top_k_other = set(other_df.head(k)['feature_name'].tolist())
                    ki = self.calculate_kuncheva_index(top_k_current, top_k_other, k, self.total_features)
                    kuncheva_scores.append(ki)
                
                avg_kuncheva = np.mean(kuncheva_scores) if kuncheva_scores else np.nan
                
                experiment_data.append({
                    'repeat': repeat,
                    'fold': fold,
                    'cindex': cindex,
                    'avg_kuncheva': avg_kuncheva
                })
            
            # 计算相关性
            if len(experiment_data) >= 3:  # 至少需要3个数据点
                df_corr = pd.DataFrame(experiment_data)
                df_corr = df_corr.dropna(subset=['avg_kuncheva', 'cindex'])
                if len(df_corr) < 3:
                    print(f"   Top-{k}: 数据不足 (只有{len(df_corr)}个有效实验)")
                    continue
                
                # Spearman相关性
                spearman_r, spearman_p = stats.spearmanr(df_corr['cindex'], df_corr['avg_kuncheva'])
                
                correlations[k] = {
                    'n_experiments': len(experiment_data),
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'experiment_data': experiment_data
                }
                
                print(f"   Top-{k}: Spearman ρ={spearman_r:.3f} (p={spearman_p:.3f})")
            else:
                print(f"   Top-{k}: 数据不足 (只有{len(experiment_data)}个实验)")
        
        return correlations
    
    def analyze_cross_xai_consistency(self, cancer_type, all_xai_results, k=100, threshold=None):
        """
        分析跨XAI方法的一致性
        
        参数:
        - all_xai_results: 所有XAI方法的稳定特征结果 {xai_method: stable_features_dict}
        - k: Top-k设置
        - threshold: 稳定性阈值，如果为None则使用self.stability_threshold
        
        返回: dict 包含一致性分析结果（如果threshold为None，返回多个阈值的结果）
        """
        # 如果未指定阈值，则对多个阈值进行分析
        # 使用 self.stability_threshold 中指定的阈值（如果是列表）或 self.stability_thresholds（如果是单个值）
        if threshold is None:
            results_by_threshold = {}
            # 如果 stability_threshold 是列表，使用列表中的阈值；否则使用 stability_thresholds
            thresholds_to_use = self._get_default_thresholds() if isinstance(self.stability_threshold, (list, tuple)) else self.stability_thresholds
            for thresh in thresholds_to_use:
                result = self.analyze_cross_xai_consistency(cancer_type, all_xai_results, k, threshold=thresh)
                if result:
                    results_by_threshold[thresh] = result
            # 返回默认阈值的结果（保持向后兼容）
            default_result = results_by_threshold.get(self._get_default_threshold(), {})
            # 同时保存所有阈值的结果（即使默认阈值没有结果，也要保存其他阈值的结果）
            if results_by_threshold:
                if default_result:
                    default_result['results_by_threshold'] = results_by_threshold
                else:
                    # 如果默认阈值没有结果，但其他阈值有结果，创建一个包含所有结果的结构
                    default_result = {'results_by_threshold': results_by_threshold}
            return default_result
        
        # 使用指定阈值进行分析
        print(f"🔗 分析 {cancer_type} 跨XAI方法一致性 (Top-{k}, 阈值≥{int(threshold*100)}%)...")
        
        if len(all_xai_results) < 2:
            print("⚠️ 至少需要2种XAI方法进行一致性分析")
            return {}
        
        # 提取各方法的稳定特征
        method_stable_features = {}
        for method, results in all_xai_results.items():
            if k in results and 'stable_features' in results[k]:
                # 获取频次>=稳定性阈值的稳定特征
                stable_features = {
                    feature for feature, freq in results[k]['stable_features'].items() 
                    if freq >= threshold
                }
                method_stable_features[method] = stable_features
                print(f"   {method}: {len(stable_features)} 个稳定特征")
        
        if len(method_stable_features) < 2:
            print("⚠️ 没有足够的方法有稳定特征数据")
            return {}
        
        # 计算成对一致性
        consistency_results = {}
        methods = list(method_stable_features.keys())
        
        # 计算所有方法的交集（共识特征）
        all_sets = list(method_stable_features.values())
        if len(all_sets) >= 2:
            consensus_features = set.intersection(*all_sets)
            
            print(f"   共识特征: {len(consensus_features)} 个")
        else:
            consensus_features = set()
        
        # 特征出现频次统计 + 记录每个特征被哪些方法选中
        feature_method_count = Counter()
        feature_selected_by_methods = defaultdict(list)  # 记录每个特征被哪些方法选中
        
        for method, features in method_stable_features.items():
            feature_method_count.update(features)
            for feature in features:
                feature_selected_by_methods[feature].append(method)
        
        # 按出现方法数排序
        sorted_features = sorted(feature_method_count.items(), 
                               key=lambda x: x[1], reverse=True)
        
        consistency_results = {
            'threshold': threshold,  # 记录使用的阈值
            'methods_analyzed': methods,
            'n_methods': len(methods),
            'consensus_features': list(consensus_features),
            'n_consensus': len(consensus_features),
            'feature_method_counts': dict(feature_method_count),
            'feature_selected_by_methods': dict(feature_selected_by_methods),  # 新增：记录每个特征被哪些方法选中
            'sorted_features_by_consensus': sorted_features[:50]  # Top 50
        }
        
        return consistency_results
    
    def save_cross_xai_results(self, cancer_type, cross_xai_data):
        """保存跨XAI一致性分析结果"""
        output_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, 'cross_xai_analysis'))
        
        # 保存各Top-k的一致性数据
        summary_data = []
        
        for k, consistency in cross_xai_data.items():
            # 检查是否有多个阈值的结果
            results_by_threshold = consistency.get('results_by_threshold', {})
            threshold_used = consistency.get('threshold', self._get_default_threshold())
            
            # 如果有多个阈值的结果，保存每个阈值的数据
            if results_by_threshold:
                for thresh, thresh_consistency in results_by_threshold.items():
                    # 基础统计信息
                    row = {
                        'cancer_type': cancer_type,
                        'top_k': k,
                        'threshold': thresh,
                        'n_methods': thresh_consistency['n_methods'],
                        'n_consensus': thresh_consistency['n_consensus']
                    }
                    
                    summary_data.append(row)
                    
                    # 保存共识特征列表（按阈值）
                    if thresh_consistency['consensus_features']:
                        consensus_df = pd.DataFrame({
                            'consensus_feature': thresh_consistency['consensus_features']
                        })
                        consensus_df.to_csv(
                            os.path.join(output_dir, f'consensus_features_top{k}_thresh{int(thresh*100)}.csv'), 
                            index=False
                        )
                    
                    # 🆕 保存按一致性排序的特征（包含选中该特征的方法信息）- 为每个阈值保存
                    if 'sorted_features_by_consensus' in thresh_consistency and thresh_consistency['sorted_features_by_consensus']:
                        feature_data = []
                        for feature, n_methods in thresh_consistency['sorted_features_by_consensus']:
                            selected_methods = thresh_consistency.get('feature_selected_by_methods', {}).get(feature, [])
                            feature_data.append({
                                'feature': feature,
                                'n_methods': n_methods,
                                'selected_by_methods': ', '.join(sorted(selected_methods))
                            })
                        
                        sorted_df = pd.DataFrame(feature_data)
                        sorted_df.to_csv(
                            os.path.join(output_dir, f'features_by_consensus_top{k}_thresh{int(thresh*100)}.csv'), 
                            index=False
                        )
            else:
                # 向后兼容：单个阈值的结果
                row = {
                    'cancer_type': cancer_type,
                    'top_k': k,
                    'threshold': threshold_used,
                    'n_methods': consistency['n_methods'],
                    'n_consensus': consistency['n_consensus']
                }
                summary_data.append(row)
            
            # 保存共识特征列表（默认阈值，保持向后兼容）
            if 'consensus_features' in consistency and consistency['consensus_features']:
                consensus_df = pd.DataFrame({
                    'consensus_feature': consistency['consensus_features']
                })
                consensus_df.to_csv(
                    os.path.join(output_dir, f'consensus_features_top{k}.csv'), 
                    index=False
                )
            
            # 保存按一致性排序的特征（包含选中该特征的方法信息）
            if 'sorted_features_by_consensus' in consistency and consistency['sorted_features_by_consensus']:
                # 构建包含方法信息的数据
                feature_data = []
                for feature, n_methods in consistency['sorted_features_by_consensus']:
                    selected_methods = consistency.get('feature_selected_by_methods', {}).get(feature, [])
                    feature_data.append({
                        'feature': feature,
                        'n_methods': n_methods,
                        'selected_by_methods': ', '.join(sorted(selected_methods))  # 按字母顺序排列方法名
                    })
                
                sorted_df = pd.DataFrame(feature_data)
                sorted_df.to_csv(
                    os.path.join(output_dir, f'features_by_consensus_top{k}.csv'), 
                    index=False
                )
        
        # 保存汇总表
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                os.path.join(output_dir, 'cross_xai_consistency_summary.csv'), 
                index=False
            )
        
        print(f"✅ 保存 {cancer_type} 跨XAI一致性结果到: {output_dir}")
    
    def create_cross_xai_summary_report(self, cross_xai_results):
        """创建跨XAI一致性汇总报告"""
        print(f"\n📋 创建跨XAI一致性汇总报告...")
        
        report_path = os.path.join(self.output_dir, 'cross_xai_consistency_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"跨XAI方法一致性分析报告\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"分析概述:\n")
            f.write(f"  XAI方法: {', '.join(self.xai_methods)}\n")
            f.write(f"  癌症类型: {len(cross_xai_results)} 个\n")
            f.write(f"  Top-k设置: {self.top_k_values}\n")
            f.write(f"  稳定性阈值: {self.stability_thresholds} (默认: {self._get_default_threshold()})\n")
            f.write(f"  生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 各癌症的一致性分析
            for cancer_type, cancer_data in cross_xai_results.items():
                f.write(f"🔬 {cancer_type} 一致性分析\n")
                f.write(f"{'-'*40}\n")
                
                for k in self.top_k_values:
                    if k in cancer_data:
                        consistency = cancer_data[k]
                        threshold_used = consistency.get('threshold', self._get_default_threshold())
                        results_by_threshold = consistency.get('results_by_threshold', {})
                        
                        # 如果有多个阈值的结果，显示所有阈值
                        if results_by_threshold:
                            f.write(f"\nTop-{k} 特征一致性 (多阈值分析):\n")
                            for thresh in sorted(results_by_threshold.keys(), reverse=True):
                                thresh_consistency = results_by_threshold[thresh]
                                f.write(f"\n  阈值≥{int(thresh*100)}%:\n")
                                f.write(f"    分析方法数: {thresh_consistency['n_methods']}\n")
                                f.write(f"    共识特征数: {thresh_consistency['n_consensus']}\n")
                        else:
                            # 向后兼容：单个阈值的结果
                            f.write(f"\nTop-{k} 特征一致性 (阈值≥{int(threshold_used*100)}%):\n")
                            f.write(f"  分析方法数: {consistency['n_methods']}\n")
                            f.write(f"  共识特征数: {consistency['n_consensus']}\n")
                        
                        # Top 10共识特征（显示默认阈值的结果）
                        if 'sorted_features_by_consensus' in consistency and consistency['sorted_features_by_consensus']:
                            f.write(f"  Top 10 高一致性特征 (阈值≥{int(threshold_used*100)}%):\n")
                            n_methods = consistency.get('n_methods', 0)
                            for i, (feature, n_methods_count) in enumerate(
                                consistency['sorted_features_by_consensus'][:10]
                            ):
                                f.write(f"    {i+1:2d}. {feature}: {n_methods_count}/{n_methods} 方法\n")
                
                f.write(f"\n")
            
            # 整体统计
            f.write(f"\n整体统计\n")
            f.write(f"{'-'*20}\n")
            
            # 计算各Top-k的平均一致性
            for k in self.top_k_values:
                n_consensus_counts = []
                
                for cancer_data in cross_xai_results.values():
                    if k in cancer_data:
                        n_consensus_counts.append(cancer_data[k]['n_consensus'])
                
                if n_consensus_counts:
                    f.write(f"Top-{k} 跨癌症统计:\n")
                    f.write(f"  平均共识特征数: {np.mean(n_consensus_counts):.1f} ± {np.std(n_consensus_counts):.1f}\n")
            
            f.write(f"\n解读说明\n")
            f.write(f"{'-'*20}\n")
            f.write(f"• 稳定性阈值: 分析使用了多个阈值 {self.stability_thresholds}，默认阈值为 {self._get_default_threshold()}\n")
            f.write(f"• 共识特征: 在≥指定阈值%的模型中都被各XAI方法选为Top-k的特征\n")
            f.write(f"• selected_by_methods: 列出了选中该特征的具体XAI方法名称\n")
            f.write(f"• 多阈值分析: 报告中对每个阈值都进行了分析，可以比较不同阈值下的一致性结果\n")
        
        print(f"✅ 跨XAI一致性汇总报告保存到: {report_path}")
    
    def calculate_overlap_curves(self, cancer_type, xai_method, all_rankings):
        """计算Overlap@k稳定性曲线数据（不绘图）"""
        kuncheva_curves = []
        
        for k in range(20, 301, 20):  # 扩展到300以支持更大的top-k值
            pairwise_results = self.analyze_pairwise_stability(all_rankings, k)
            
            kuncheva_values = [v for v in pairwise_results['kuncheva'] if not np.isnan(v)]
            
            kuncheva_curves.append({
                'k': k,
                'mean': np.mean(kuncheva_values) if kuncheva_values else np.nan,
                'std': np.std(kuncheva_values) if kuncheva_values else np.nan,
                'median': np.median(kuncheva_values) if kuncheva_values else np.nan,
                'q25': np.percentile(kuncheva_values, 25) if kuncheva_values else np.nan,
                'q75': np.percentile(kuncheva_values, 75) if kuncheva_values else np.nan,
                'ci_lower': np.percentile(kuncheva_values, 2.5) if kuncheva_values else np.nan,
                'ci_upper': np.percentile(kuncheva_values, 97.5) if kuncheva_values else np.nan
            })
        
        # 转换为DataFrame并返回
        df_kuncheva = pd.DataFrame(kuncheva_curves)
        
        return df_kuncheva
    
    
    def _analyze_single_topk_stability(self, cancer_type, xai_method, rankings, k):
        """
        分析单个Top-k值的稳定性（辅助方法）
        
        返回: dict 包含该Top-k的所有稳定性分析结果
        """
        print(f"   📊 分析 Top-{k} 稳定性...")
        
        # 成对稳定性分析
        pairwise_results = self.analyze_pairwise_stability(rankings, k)
        
        # 稳定性摘要
        stability_summary = self.create_stability_summary(pairwise_results)
        
        # 分层分析
        stratified_analysis = self.create_stratified_analysis(pairwise_results)
        
        # 特征频次分析
        feature_freq = self.calculate_feature_frequency(rankings, k)
        
        # 随机基线
        random_baseline = self.calculate_random_baseline(k, self.total_features)
        
        # 稳定核心特征（频次≥阈值的特征）- 为每个阈值计算
        stable_cores = {
            threshold: {
                feature: freq for feature, freq in feature_freq.items() 
                if freq >= threshold
            }
            for threshold in self.stability_thresholds
        }
        
        # 保持向后兼容：使用默认阈值（确保默认阈值在阈值列表中）
        default_thresh = self._get_default_threshold()
        if default_thresh not in stable_cores:
            # 如果默认阈值不在列表中，使用第一个阈值
            stable_core = stable_cores.get(self.stability_thresholds[0], {})
        else:
            stable_core = stable_cores[default_thresh]
        
        # 保存Top稳定特征列表
        # 按频次排序特征
        sorted_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_features:
            stable_features_df = pd.DataFrame(sorted_features, columns=['feature_name', 'frequency'])
            stable_features_df['rank'] = range(1, len(stable_features_df) + 1)
            
            output_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
            stable_features_df.to_csv(
                os.path.join(output_dir, f'stable_features_top{k}.csv'), 
                index=False
            )
            
            # 为每个阈值保存稳定核心特征列表
            for threshold in self.stability_thresholds:
                if threshold in stable_cores:
                    threshold_features = sorted(stable_cores[threshold].items(), 
                                              key=lambda x: x[1], reverse=True)
                    if threshold_features:
                        threshold_df = pd.DataFrame(threshold_features, 
                                                   columns=['feature_name', 'frequency'])
                        threshold_df['rank'] = range(1, len(threshold_df) + 1)
                        threshold_df.to_csv(
                            os.path.join(output_dir, f'stable_core_top{k}_threshold{int(threshold*100)}.csv'),
                            index=False
                        )
        
        # 🆕 保存成对比较的原始数据（用于高质量箱线图）
        self._save_pairwise_raw_data(cancer_type, xai_method, pairwise_results, k, output_dir)
        
        return {
            'pairwise_results': pairwise_results,
            'stability_summary': stability_summary,
            'stratified_analysis': stratified_analysis,
            'feature_frequency': feature_freq,
            'stable_features': feature_freq,  # 添加用于跨XAI分析
            'random_baseline': random_baseline,
            'stable_core': stable_core,  # 默认阈值的稳定核心
            'stable_cores': stable_cores,  # 所有阈值的稳定核心字典
            'n_pairs': len(pairwise_results['kuncheva']),
            'n_rankings': len(rankings)
        }
    
    def _analyze_all_topk_stability(self, cancer_type, xai_method, rankings):
        """
        分析所有Top-k值的稳定性（辅助方法）
        
        返回: dict {k: stability_result}
        """
        stability_results = {}
        stability_results['rankings'] = rankings  # 保存rankings用于性能相关性分析
        
        for k in self.top_k_values:
            stability_results[k] = self._analyze_single_topk_stability(
                cancer_type, xai_method, rankings, k
            )
        
        return stability_results
    
    
    def _analyze_and_save_performance_correlations(self, cancer_type, xai_method, 
                                                   stability_results, performance_data, output_dir):
        """
        分析并保存性能相关性结果（辅助方法）
        
        返回: dict 性能相关性结果
        """
        performance_correlations = {}
        
        if performance_data:
            print(f"   📈 进行性能相关性分析...")
            performance_correlations = self.analyze_stability_performance_correlation(
                cancer_type, xai_method, stability_results, performance_data
            )
            
            # 保存性能相关性结果
            if performance_correlations:
                perf_corr_df = []
                for k, corr_data in performance_correlations.items():
                    perf_corr_df.append({
                        'top_k': k,
                        'n_experiments': corr_data['n_experiments'],
                        'spearman_r': corr_data['spearman_r'],
                        'spearman_p': corr_data['spearman_p']
                    })
                
                if perf_corr_df:
                    perf_df = pd.DataFrame(perf_corr_df)
                    perf_df.to_csv(os.path.join(output_dir, 'performance_correlations.csv'), index=False)
        
        return performance_correlations
    
    def analyze_cancer_xai_combination(self, cancer_type, xai_method):
        """分析单个癌症-XAI组合的稳定性"""
        print(f"\n🔬 分析 {cancer_type} - {xai_method}")
        
        # 加载特征排名数据
        rankings = self.load_feature_rankings(cancer_type, xai_method)
        
        if len(rankings) < self.min_rankings_required:
            print(f"⚠️ 数据不足，跳过 {cancer_type}-{xai_method} (只有{len(rankings)}个文件，需要至少{self.min_rankings_required}个)")
            return None
        
        # 加载模型性能数据用于相关性分析
        performance_data = self.load_model_performance_data(cancer_type)
        
        # 为每个Top-k值分析稳定性
        stability_results = self._analyze_all_topk_stability(cancer_type, xai_method, rankings)
        
        # 计算Overlap@k曲线数据
        print(f"   📈 计算Overlap@k曲线数据...")
        kuncheva_curve = self.calculate_overlap_curves(cancer_type, xai_method, rankings)
        
        # 保存曲线数据
        output_dir = os.path.join(self.output_dir, cancer_type, xai_method)
        kuncheva_curve.to_csv(os.path.join(output_dir, 'kuncheva_curve_data.csv'), index=False)
        
        # 🆕 性能相关性分析
        performance_correlations = self._analyze_and_save_performance_correlations(
            cancer_type, xai_method, stability_results, performance_data, output_dir
        )
        
        print(f"✅ 完成 {cancer_type}-{xai_method} 稳定性分析")
        
        return {
            'cancer_type': cancer_type,
            'xai_method': xai_method,
            'n_rankings': len(rankings),
            'stability_results': stability_results,
            'performance_correlations': performance_correlations,
            'kuncheva_curve': kuncheva_curve,
            
        }
    
    def create_summary_report(self, cancer_type, xai_method, analysis_result):
        """创建稳定性分析报告"""
        if analysis_result is None:
            return
        
        output_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
        
        # 创建主报告
        report_data = []
        
        for k in self.top_k_values:
            if k in analysis_result['stability_results']:
                result = analysis_result['stability_results'][k]
                summary = result['stability_summary']
                baseline = result['random_baseline']
                stratified = result['stratified_analysis']
                
                # 基础统计
                row = {
                    'cancer_type': cancer_type,
                    'xai_method': xai_method,
                    'top_k': k,
                    'n_rankings': result['n_rankings'],
                    'n_pairs': result['n_pairs'],
                }
                
                # 添加多阈值的稳定核心特征数量
                if 'stable_cores' in result:
                    for threshold in sorted(self.stability_thresholds, reverse=True):
                        if threshold in result['stable_cores']:
                            row[f'n_stable_core_{int(threshold*100)}pct'] = len(result['stable_cores'][threshold])
                else:
                    row['n_stable_core_70pct'] = len(result['stable_core'])
                
                # 稳定性指标
                for metric in ['kuncheva', 'spearman_r']:
                    row.update({
                        f'{metric}_mean': summary.get(f'{metric}_mean', np.nan),
                        f'{metric}_std': summary.get(f'{metric}_std', np.nan),
                        f'{metric}_median': summary.get(f'{metric}_median', np.nan),
                        f'{metric}_q25': summary.get(f'{metric}_q25', np.nan),
                        f'{metric}_q75': summary.get(f'{metric}_q75', np.nan)
                    })
                
                # 随机基线对比
                row.update({
                    'random_kuncheva_mean': baseline['kuncheva_mean'],
                    'kuncheva_vs_random': summary.get('kuncheva_mean', np.nan) - baseline['kuncheva_mean']
                })
                
                # 分层分析
                row.update({
                    'kuncheva_same_fold_mean': stratified.get('kuncheva_same_fold', {}).get('mean', np.nan),
                    'kuncheva_cross_fold_mean': stratified.get('kuncheva_cross_fold', {}).get('mean', np.nan)
                })
                
                report_data.append(row)
        
        # 保存报告
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(os.path.join(output_dir, 'stability_summary_report.csv'), index=False)
            
    
    def run_full_analysis(self, cancer_types=None):
        """运行完整的稳定性分析"""
        print(f"🚀 开始特征稳定性分析")
        print(f"   分析 {len(self.xai_methods)} 种XAI方法")
        print(f"   Top-k设置: {self.top_k_values}")
        
        # 获取癌症类型列表
        if cancer_types is None:
            cancer_types = [d for d in os.listdir(self.results_dir) 
                          if os.path.isdir(os.path.join(self.results_dir, d))]
            cancer_types.sort()
        
        print(f"   发现 {len(cancer_types)} 种癌症类型: {', '.join(cancer_types)}")
        
        all_results = []
        
        # 按癌症类型组织结果，用于跨XAI分析
        cancer_xai_results = {cancer: {} for cancer in cancer_types}
        
        # 分析每个癌症-XAI组合
        for cancer_type in cancer_types:
            for xai_method in self.xai_methods:
                result = self.analyze_cancer_xai_combination(cancer_type, xai_method)
                
                if result is not None:
                    all_results.append(result)
                    cancer_xai_results[cancer_type][xai_method] = result
                    
                    # 创建分析报告
                    self.create_summary_report(cancer_type, xai_method, result)
        
        # 🆕 跨XAI方法一致性分析
        print(f"\n🔗 开始跨XAI方法一致性分析...")
        cross_xai_results = {}
        
        for cancer_type in cancer_types:
            if len(cancer_xai_results[cancer_type]) >= 2:  # 至少需要2种XAI方法
                print(f"\n📊 分析 {cancer_type} 跨XAI一致性...")
                
                # 准备XAI结果数据
                xai_stability_data = {}
                for xai_method, result in cancer_xai_results[cancer_type].items():
                    if 'stability_results' in result:
                        xai_stability_data[xai_method] = result['stability_results']
                
                if len(xai_stability_data) >= 2:
                    # 对每个Top-k值进行跨XAI一致性分析
                    cancer_cross_xai = {}
                    for k in self.top_k_values:
                        consistency = self.analyze_cross_xai_consistency(
                            cancer_type, xai_stability_data, k
                        )
                        if consistency:
                            cancer_cross_xai[k] = consistency
                    
                    if cancer_cross_xai:
                        cross_xai_results[cancer_type] = cancer_cross_xai
                        
                        # 保存跨XAI一致性结果
                        self.save_cross_xai_results(cancer_type, cancer_cross_xai)
        
        # 创建跨XAI汇总报告
        if cross_xai_results:
            self.create_cross_xai_summary_report(cross_xai_results)
        
        # 创建跨癌症汇总报告
        if all_results:
            self.create_cross_cancer_summary(all_results)
        
        print(f"\n🎉 稳定性分析完成!")
        print(f"   成功分析 {len(all_results)} 个癌症-XAI组合")
        print(f"   结果保存在: {self.output_dir}")
    
    def create_cross_cancer_summary(self, all_results):
        """创建跨癌症汇总报告"""
        print(f"\n📋 创建跨癌症汇总报告...")
        
        summary_data = []
        
        for result in all_results:
            cancer_type = result['cancer_type']
            xai_method = result['xai_method']
            n_rankings = result['n_rankings']
            
            for k in self.top_k_values:
                if k in result['stability_results']:
                    stability_result = result['stability_results'][k]
                    summary = stability_result['stability_summary']
                    
                    row = {
                        'cancer_type': cancer_type,
                        'xai_method': xai_method,
                        'top_k': k,
                        'n_rankings': n_rankings,
                        'kuncheva_mean': summary.get('kuncheva_mean', np.nan),
                        'kuncheva_std': summary.get('kuncheva_std', np.nan),
                        'n_stable_core': len(stability_result['stable_core'])
                    }
                    
                    # 添加所有阈值的稳定核心特征数量
                    if 'stable_cores' in stability_result:
                        for threshold in self.stability_thresholds:
                            threshold_key = f'n_stable_core_{int(threshold*100)}pct'
                            if threshold in stability_result['stable_cores']:
                                row[threshold_key] = len(stability_result['stable_cores'][threshold])
                            else:
                                row[threshold_key] = 0
                    
                    summary_data.append(row)
        
        # 保存汇总表
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.output_dir, 'cross_cancer_stability_summary.csv'), 
                             index=False)
            
            # 🆕 创建聚合数据文件（用于可视化）
            self._create_aggregated_data_files(summary_df)
    
    def _create_aggregated_data_files(self, summary_df):
        """
        创建聚合数据文件（用于可视化）
        
        生成文件:
        1. aggregated_by_xai_method.csv - 按XAI方法聚合（跨所有癌症）
        """
        print(f"\n📊 创建聚合数据文件...")
        
        # 按XAI方法聚合（每个XAI方法跨所有癌症的统计）
        xai_aggregated = []
        for xai_method in summary_df['xai_method'].unique():
            for top_k in self.top_k_values:
                method_topk_data = summary_df[
                    (summary_df['xai_method'] == xai_method) & 
                    (summary_df['top_k'] == top_k)
                ]
                
                if not method_topk_data.empty:
                    xai_aggregated.append({
                        'xai_method': xai_method,
                        'top_k': top_k,
                        'n_cancers': len(method_topk_data),
                        'kuncheva_mean': method_topk_data['kuncheva_mean'].mean(),
                        'kuncheva_std': method_topk_data['kuncheva_mean'].std(),
                        'kuncheva_median': method_topk_data['kuncheva_mean'].median(),
                        'kuncheva_min': method_topk_data['kuncheva_mean'].min(),
                        'kuncheva_max': method_topk_data['kuncheva_mean'].max()
                    })
        
        if xai_aggregated:
            df_xai_agg = pd.DataFrame(xai_aggregated)
            df_xai_agg.to_csv(os.path.join(self.output_dir, 'aggregated_by_xai_method.csv'), index=False)
            print(f"   ✅ 保存XAI方法聚合数据: aggregated_by_xai_method.csv")
        
        print(f"   ✅ 聚合数据文件创建完成")


def main():
    """主函数"""
    import argparse

    TCGA_DIR = Path(os.environ.get('TCGA_DIR', Path(__file__).resolve().parents[1])).resolve()
    
    parser = argparse.ArgumentParser(description="特征重要性稳定性分析")
    parser.add_argument('--results_dir', type=str,
                       default=str(TCGA_DIR),
                       help='结果根目录（包含各xai方法各癌症的feature_importance子目录）')
    parser.add_argument('--output_dir', type=str,
                       default=str(TCGA_DIR / 'stability_analysis'),
                       help='稳定性分析结果输出目录')
    args = parser.parse_args()
    
    # 设置固定的XAI方法和癌症类型
    xai_methods = ['shap', 'IG', 'LRP', 'PFI', 'deepshap', 'DeepLIFT']
    cancer_types = [
        'BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC',
        'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC'
    ]

    print(f"🎯 分析配置:")
    print(f"   XAI方法: {', '.join(xai_methods)}")
    print(f"   癌症类型: {', '.join(cancer_types)}")
    print(f"   根目录: {args.results_dir}")
    print(f"   输出目录: {args.output_dir}")

    # 创建分析器
    analyzer = FeatureStabilityAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        xai_methods=xai_methods
    )
    
    # 运行分析
    analyzer.run_full_analysis(cancer_types=cancer_types)
    
    print(f"\n🎉 综合稳定性分析完成！")
    print(f"\n📂 主要结果文件:")
    print(f"   📁 主目录: {args.output_dir}")
    print(f"   � 跨癌症汇总数据: cross_cancer_stability_summary.csv")
    print(f"   📊 聚合数据: aggregated_by_xai_method.csv")
    print(f"   � 跨XAI一致性报告: cross_xai_consistency_report.txt")
    print(f"\n🔬 各癌症详细结果:")
    print(f"   📁 [CANCER]/[XAI_METHOD]/")
    print(f"      ├── stability_summary_report.csv              # 数值汇总（含多阈值）")
    print(f"      ├── pairwise_stability_raw_top*.csv           # 成对比较原始数据")
    print(f"      ├── kuncheva_curve_data.csv                   # Kuncheva曲线数据")
    print(f"      ├── performance_correlations.csv              # 性能相关性统计")
    print(f"      ├── stable_features_top*.csv                  # 所有稳定特征列表")
    print(f"      ├── stable_core_top*_threshold70.csv          # 阈值≥70%的稳定核心特征")
    print(f"      ├── stable_core_top*_threshold75.csv          # 阈值≥75%的稳定核心特征")
    print(f"      ├── stable_core_top*_threshold80.csv          # 阈值≥80%的稳定核心特征")
    print(f"      └── stable_core_top*_threshold85.csv          # 阈值≥85%的稳定核心特征")
    print(f"   📁 [CANCER]/cross_xai_analysis/")
    print(f"      ├── cross_xai_consistency_summary.csv         # 跨XAI一致性汇总")
    print(f"      ├── consensus_features_top*.csv               # 共识特征（所有XAI共同选择）")
    print(f"      └── features_by_consensus_top*.csv            # 按一致性排序的特征")
    print(f"\n💡 核心功能:")
    print(f"   📊 稳定性分析: Kuncheva指标（保存原始数据用于可视化）")
    print(f"   🎯 性能关联: 特征稳定性与C-index的相关性分析")
    print(f"   🔗 跨XAI一致性: 比较6种XAI方法选择特征的一致程度")
    print(f"   🎯 共识特征: 识别所有XAI方法都认为重要的稳健特征（用于数据库检索）")
    print(f"   📈 多阈值分析: 支持70%, 75%, 80%, 85%, 90%多个稳定性阈值")

if __name__ == "__main__":
    main()