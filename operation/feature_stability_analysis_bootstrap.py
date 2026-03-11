"""
Bootstrap特征重要性稳定性分析脚本

功能：分析同一癌症内、同一XAI方法下，跨不同bootstrap模型的Top-k特征稳定性
作者：AI助手
日期：2025-10-23
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体（图表将使用英文标签）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class BootstrapFeatureStabilityAnalyzer:
    """Bootstrap特征稳定性分析器"""
    
    def __init__(self, results_dir, output_dir, xai_methods=None, num_bootstraps=100):
        """
        初始化分析器
        
        参数:
        - results_dir: 结果根目录
        - output_dir: 输出目录
        - xai_methods: XAI方法列表，如['shap', 'ig', 'lrp', 'pfi', 'deepshap']
        - num_bootstraps: Bootstrap迭代次数（100或300）
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.xai_methods = xai_methods
        self.num_bootstraps = num_bootstraps
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置参数
        self.top_k_values = [20, 50, 100, 200, 300]  # 不同的Top-k值
        self.total_features = 2000  # 总特征数
        self.stability_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]  # 稳定特征阈值列表
        self.stability_threshold = [0.75, 0.7, 0.8]  # 可以是单个值如 0.75，或列表如 [0.75, 0.8]
        self.random_simulations = 1000  # 随机基线模拟次数
        self.min_rankings_required = 10  # 最少需要的排名文件数（至少10%的bootstrap成功）
        
        print(f"🔧 初始化Bootstrap稳定性分析器")
        print(f"   结果目录: {results_dir}")
        print(f"   输出目录: {output_dir}")
        print(f"   XAI方法: {', '.join(self.xai_methods)}")
        print(f"   Bootstrap次数: {num_bootstraps}")
        print(f"   Top-k值: {self.top_k_values}")
        # 规范化 stability_threshold：如果是列表，使用第一个值；如果是单个值，保持不变
        if isinstance(self.stability_threshold, (list, tuple)):
            if len(self.stability_threshold) == 0:
                # 如果列表为空，使用 stability_thresholds 的第一个值
                self._default_stability_threshold = self.stability_thresholds[0]
                self._default_stability_thresholds = self.stability_thresholds
            else:
                self._default_stability_threshold = float(self.stability_threshold[0])
                self._default_stability_thresholds = [float(t) for t in self.stability_threshold]
        else:
            self._default_stability_threshold = float(self.stability_threshold)
            self._default_stability_thresholds = [self._default_stability_threshold]
        print(f"   稳定性阈值: {self.stability_thresholds} (默认: {self._default_stability_threshold})")
    
    def _get_default_threshold(self):
        """获取默认阈值（单个值）"""
        return self._default_stability_threshold
    
    def _get_default_thresholds(self):
        """获取默认阈值列表"""
        return self._default_stability_thresholds
    
    def _ensure_dir(self, dir_path):
        """确保目录存在"""
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
        xai_dir = f'{xai_method}_bootstrap_results'
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

        # Bootstrap文件命名格式: seed{X}_{xai}_ranking.csv
        subdir = f'{method}_feature_importance'
        pattern = f"seed{{}}_{method}_ranking.csv"
        
        feature_dir = self._build_xai_results_path(xai_method, cancer_type, subdir)
        return feature_dir, pattern
    
    def _calculate_stats_summary(self, values):
        """统一的统计指标计算方法"""
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
        加载指定癌症和XAI方法的所有bootstrap特征排名文件
        
        返回: dict {seed: DataFrame}
        """
        rankings = {}
        feature_dir, file_pattern = self._build_feature_importance_path(xai_method, cancer_type)
        
        if not os.path.exists(feature_dir):
            print(f"⚠️ 目录不存在: {feature_dir}")
            return rankings
        
        # 加载所有bootstrap seeds
        success_count = 0
        for seed in range(1, self.num_bootstraps + 1):
            file_path = os.path.join(feature_dir, file_pattern.format(seed))
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # 验证必要列存在
                    required_cols = ['feature_name', 'importance_score', 'rank']
                    if all(col in df.columns for col in required_cols):
                        # 按重要性得分排序（确保排名正确）
                        df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
                        df['rank'] = range(1, len(df) + 1)
                        
                        rankings[seed] = df
                        success_count += 1
                    else:
                        print(f"⚠️ 文件缺少必要列: {file_path}")
                
                except Exception as e:
                    print(f"❌ 加载文件失败: {file_path}, 错误: {e}")
        
        print(f"✅ {cancer_type}-{xai_method}: 成功加载 {success_count}/{self.num_bootstraps} 个排名文件")
        return rankings
    
    def calculate_kuncheva_index(self, set1, set2, k, m):
        """计算Kuncheva指数（机会校正的重叠度指标）"""
        intersection = len(set1.intersection(set2))
        expected_overlap = k * k / m
        denominator = k - expected_overlap
        
        if abs(denominator) < 1e-10:
            return 1.0 if intersection == k else 0.0
        
        return (intersection - expected_overlap) / denominator
    
    def calculate_rank_correlation(self, rank1, rank2):
        """计算排名相关性（Spearman）"""
        common_features = set(rank1).intersection(set(rank2))
        
        if len(common_features) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        ranks1 = [rank1.index(f) + 1 for f in common_features]
        ranks2 = [rank2.index(f) + 1 for f in common_features]
        
        spearman_r, spearman_p = stats.spearmanr(ranks1, ranks2)
        
        return spearman_r, spearman_p, np.nan, np.nan
    
    def analyze_pairwise_stability(self, rankings, k):
        """
        计算成对稳定性指标
        
        返回: dict 包含各种指标的列表
        """
        results = {
            'kuncheva': [],
            'spearman_r': [],
            'spearman_p': [],
            'pair_info': []
        }
        
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
            
            # 记录pair信息
            results['pair_info'].append({
                'seed1': key1,
                'seed2': key2
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
        """计算随机基线"""
        if n_simulations is None:
            n_simulations = self.random_simulations
        random_kunchevas = []
        
        for _ in range(n_simulations):
            set1 = set(np.random.choice(m, k, replace=False))
            set2 = set(np.random.choice(m, k, replace=False))
            
            kuncheva = self.calculate_kuncheva_index(set1, set2, k, m)
            
            random_kunchevas.append(kuncheva)
        
        return {
            'kuncheva_mean': np.mean(random_kunchevas),
            'kuncheva_std': np.std(random_kunchevas)
        }
    
    def create_stability_summary(self, pairwise_results):
        """创建稳定性指标摘要"""
        summary = {}
        
        for metric in ['kuncheva', 'spearman_r']:
            stats = self._calculate_stats_summary(pairwise_results[metric])
            
            for stat_name, stat_value in stats.items():
                summary[f'{metric}_{stat_name}'] = stat_value
        
        return summary
    
    def load_model_performance_data(self, cancer_type):
        """
        加载Bootstrap模型性能数据（C-index）用于性能关联分析
        
        返回: dict {seed: cindex}
        """
        performance_data = {}
        
        # Bootstrap的性能数据保存在cindex_array.npy中
        results_file = os.path.join(self.results_dir, 'results_bootstrap', cancer_type, 'cindex_array.npy')
        
        if not os.path.exists(results_file):
            print(f"⚠️ 性能数据文件不存在: {results_file}")
            return performance_data
        
        try:
            cindex_array = np.load(results_file)
            
            # cindex_array索引从0开始，seed从1开始
            for seed in range(1, min(len(cindex_array) + 1, self.num_bootstraps + 1)):
                cindex = float(cindex_array[seed - 1])
                performance_data[seed] = cindex
                
            print(f"✅ 成功加载 {cancer_type} 的性能数据: {len(performance_data)} 个bootstrap模型")
            
        except Exception as e:
            print(f"❌ 加载性能数据失败: {results_file}, 错误: {e}")
        
        return performance_data
    
    def analyze_stability_performance_correlation(self, cancer_type, xai_method, stability_results, performance_data):
        """
        分析稳定性与模型性能的相关性
        
        返回: dict 包含相关性分析结果
        """
        print(f"📈 分析 {cancer_type}-{xai_method} 的稳定性与性能相关性...")
        
        correlations = {}
        
        for k in self.top_k_values:
            if k not in stability_results:
                continue
            
            # 提取每个bootstrap的稳定性指标
            experiment_data = []
            
            for seed, ranking_df in stability_results['rankings'].items():
                if seed not in performance_data:
                    continue
                
                cindex = performance_data[seed]
                
                # 计算该bootstrap与其他bootstrap的平均稳定性
                other_seeds = [s for s in stability_results['rankings'].keys() if s != seed]
                
                if len(other_seeds) == 0:
                    continue
                
                # 计算与其他bootstrap的平均Kuncheva稳定性
                top_k_current = set(ranking_df.head(k)['feature_name'].tolist())
                kuncheva_scores = []
                
                for other_seed in other_seeds:
                    other_df = stability_results['rankings'][other_seed]
                    top_k_other = set(other_df.head(k)['feature_name'].tolist())
                    ki = self.calculate_kuncheva_index(top_k_current, top_k_other, k, self.total_features)
                    kuncheva_scores.append(ki)
                
                avg_kuncheva = np.mean(kuncheva_scores) if kuncheva_scores else np.nan
                
                experiment_data.append({
                    'seed': seed,
                    'cindex': cindex,
                    'avg_kuncheva': avg_kuncheva
                })
            
            # 计算相关性
            if len(experiment_data) >= 3:
                df_corr = pd.DataFrame(experiment_data)
                df_corr = df_corr.dropna(subset=['avg_kuncheva', 'cindex'])
                if len(df_corr) < 3:
                    print(f"   Top-{k}: 数据不足 (只有{len(df_corr)}个有效bootstrap)")
                    continue
                
                spearman_r, spearman_p = stats.spearmanr(df_corr['cindex'], df_corr['avg_kuncheva'])
                
                correlations[k] = {
                    'n_experiments': len(experiment_data),
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'experiment_data': experiment_data
                }
                
                print(f"   Top-{k}: Spearman ρ={spearman_r:.3f} (p={spearman_p:.3f})")
            else:
                print(f"   Top-{k}: 数据不足 (只有{len(experiment_data)}个bootstrap)")
        
        return correlations
    
    def analyze_cross_xai_consistency(self, cancer_type, all_xai_results, k=100, threshold=None):
        """
        分析跨XAI方法的一致性
        
        参数:
        - all_xai_results: 所有XAI方法的稳定特征结果 {xai_method: stable_features_dict}
        - k: Top-k设置
        - threshold: 稳定性阈值，如果为None则使用默认阈值配置

        返回: dict 包含一致性分析结果
        """
        # 如果未指定阈值，则对多个阈值进行分析
        # 使用 `self.stability_threshold` 中指定的阈值（如果是列表）或 `self.stability_thresholds`（如果是单个值）
        if threshold is None:
            results_by_threshold = {}
            # 如果 `self.stability_threshold` 是列表，使用列表中的阈值；否则使用稳定阈值集合
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
                stable_features = {
                    feature for feature, freq in results[k]['stable_features'].items() 
                    if freq >= threshold
                }
                method_stable_features[method] = stable_features
                print(f"   {method}: {len(stable_features)} 个稳定特征")
        
        if len(method_stable_features) < 2:
            print("⚠️ 没有足够的方法有稳定特征数据")
            return {}
        
        # 计算所有方法的交集（共识特征）
        all_sets = list(method_stable_features.values())
        if len(all_sets) >= 2:
            consensus_features = set.intersection(*all_sets)
            print(f"   共识特征: {len(consensus_features)} 个")
        else:
            consensus_features = set()
        
        # 特征出现频次统计
        feature_method_count = Counter()
        feature_selected_by_methods = defaultdict(list)
        
        for method, features in method_stable_features.items():
            feature_method_count.update(features)
            for feature in features:
                feature_selected_by_methods[feature].append(method)
        
        sorted_features = sorted(feature_method_count.items(), 
                               key=lambda x: x[1], reverse=True)
        
        consistency_results = {
            'methods_analyzed': methods,
            'n_methods': len(methods),
            'threshold': threshold,
            'consensus_features': list(consensus_features),
            'n_consensus': len(consensus_features),
            'feature_method_counts': dict(feature_method_count),
            'feature_selected_by_methods': dict(feature_selected_by_methods),
            'sorted_features_by_consensus': sorted_features[:50]
        }
        
        return consistency_results
    
    def save_cross_xai_results(self, cancer_type, cross_xai_data):
        """保存跨XAI一致性分析结果"""
        output_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, 'cross_xai_analysis'))
        
        summary_data = []
        
        for k, consistency in cross_xai_data.items():
            row = {
                'cancer_type': cancer_type,
                'top_k': k,
                'threshold': consistency['threshold'],
                'n_methods': consistency['n_methods'],
                'n_consensus': consistency['n_consensus']
            }
            
            summary_data.append(row)
            
            # 保存共识特征列表
            if consistency['consensus_features']:
                consensus_df = pd.DataFrame({
                    'consensus_feature': consistency['consensus_features']
                })
                consensus_df.to_csv(
                    os.path.join(output_dir, f'consensus_features_top{k}.csv'), 
                    index=False
                )
            
            # 保存按一致性排序的特征
            if consistency['sorted_features_by_consensus']:
                feature_data = []
                for feature, n_methods in consistency['sorted_features_by_consensus']:
                    selected_methods = consistency['feature_selected_by_methods'].get(feature, [])
                    feature_data.append({
                        'feature': feature,
                        'n_methods': n_methods,
                        'selected_by_methods': ', '.join(sorted(selected_methods))
                    })
                
                sorted_df = pd.DataFrame(feature_data)
                sorted_df.to_csv(
                    os.path.join(output_dir, f'features_by_consensus_top{k}.csv'), 
                    index=False
                )
        
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
            f.write(f"Bootstrap跨XAI方法一致性分析报告\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"分析概述:\n")
            f.write(f"  XAI方法: {', '.join(self.xai_methods)}\n")
            f.write(f"  癌症类型: {len(cross_xai_results)} 个\n")
            f.write(f"  Bootstrap次数: {self.num_bootstraps}\n")
            f.write(f"  Top-k设置: {self.top_k_values}\n")
            f.write(f"  生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for cancer_type, cancer_data in cross_xai_results.items():
                f.write(f"🔬 {cancer_type} 一致性分析\n")
                f.write(f"{'-'*40}\n")
                
                for k in self.top_k_values:
                    if k in cancer_data:
                        consistency = cancer_data[k]
                        f.write(f"\nTop-{k} 特征一致性:\n")
                        f.write(f"  分析方法数: {consistency['n_methods']}\n")
                        f.write(f"  共识特征数: {consistency['n_consensus']}\n")
                        
                        if consistency['sorted_features_by_consensus']:
                            f.write(f"  Top 10 高一致性特征:\n")
                            for i, (feature, n_methods) in enumerate(
                                consistency['sorted_features_by_consensus'][:10]
                            ):
                                f.write(f"    {i+1:2d}. {feature}: {n_methods}/{consistency['n_methods']} 方法\n")
                
                f.write(f"\n")
            
            f.write(f"\n整体统计\n")
            f.write(f"{'-'*20}\n")
            
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
            f.write(f"• 共识特征: 在≥{int(self._get_default_threshold()*100)}%的bootstrap模型中都被各XAI方法选为Top-k的特征\n")
            f.write(f"• selected_by_methods: 列出了选中该特征的具体XAI方法名称\n")
        
        print(f"✅ 跨XAI一致性汇总报告保存到: {report_path}")
    
    def plot_overlap_curves(self, cancer_type, xai_method, all_rankings):
        """绘制Overlap@k稳定性曲线"""
        kuncheva_curves = []
        
        for k in range(20, 301, 20):
            pairwise_results = self.analyze_pairwise_stability(all_rankings, k)
            
            kuncheva_values = [v for v in pairwise_results['kuncheva'] if not np.isnan(v)]
            
            kuncheva_curves.append({
                'k': k,
                'mean': np.mean(kuncheva_values) if kuncheva_values else np.nan,
                'std': np.std(kuncheva_values) if kuncheva_values else np.nan,
                'ci_lower': np.percentile(kuncheva_values, 2.5) if kuncheva_values else np.nan,
                'ci_upper': np.percentile(kuncheva_values, 97.5) if kuncheva_values else np.nan
            })
        
        # 绘图
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        
        df_kuncheva = pd.DataFrame(kuncheva_curves)
        ax.plot(df_kuncheva['k'], df_kuncheva['mean'], 'r-s', label='Kuncheva Index', markersize=4)
        ax.fill_between(df_kuncheva['k'], df_kuncheva['ci_lower'], df_kuncheva['ci_upper'], 
                        alpha=0.3, color='#E94F1E')
        ax.set_xlabel('Number of Top-k Features')
        ax.set_ylabel('Kuncheva Index')
        ax.set_title(f'{cancer_type} - {xai_method}\nKuncheva Index vs Top-k (Bootstrap)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        plot_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
        plt.savefig(os.path.join(plot_dir, 'overlap_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return df_kuncheva
    
    def plot_stability_distribution(self, cancer_type, xai_method, pairwise_results, k):
        """绘制稳定性指标分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Kuncheva分布
        kuncheva_values = [v for v in pairwise_results['kuncheva'] if not np.isnan(v)]
        if kuncheva_values:
            axes[0].hist(kuncheva_values, bins=20, alpha=0.7, color='#E74C3C', edgecolor='black')
            axes[0].axvline(np.mean(kuncheva_values), color='#1D80C1', linestyle='--', 
                            label=f'Mean: {np.mean(kuncheva_values):.3f}')
            axes[0].axvline(0, color='gray', linestyle='-', alpha=0.7, label='Random Baseline')
            axes[0].set_xlabel('Kuncheva Index')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Kuncheva Index Distribution (Top-{k})')
            axes[0].legend(loc='upper right')
        
        # Spearman相关性分布
        spearman_values = [v for v in pairwise_results['spearman_r'] if not np.isnan(v)]
        if spearman_values:
            axes[1].hist(spearman_values, bins=20, alpha=0.7, color='#8E44AD', edgecolor='black')
            axes[1].axvline(np.mean(spearman_values), color='#E74C3C', linestyle='--', 
                            label=f'Mean: {np.mean(spearman_values):.3f}')
            axes[1].set_xlabel('Spearman Correlation')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Spearman Correlation Distribution (Top-{k})')
            axes[1].legend(loc='upper right')
        
        plt.suptitle(f'{cancer_type} - {xai_method} Bootstrap Stability Analysis', fontsize=16)
        plt.tight_layout()
        
        plot_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
        plt.savefig(os.path.join(plot_dir, f'stability_distribution_top{k}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_frequency(self, cancer_type, xai_method, feature_freq, k):
        """绘制特征频次分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        frequencies = list(feature_freq.values())
        axes[0].hist(frequencies, bins=20, alpha=0.7, color='#8E44AD', edgecolor='black')
        axes[0].set_xlabel('Feature Occurrence Frequency')
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title(f'Feature Frequency Distribution (Top-{k}, Bootstrap)')
        axes[0].axvline(np.mean(frequencies), color='#E74C3C', linestyle='--', 
                       label=f'Mean: {np.mean(frequencies):.3f}')
        axes[0].legend(loc='upper right')
        
        sorted_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
        top_20 = sorted_features[:20]
        
        if top_20:
            features, freqs = zip(*top_20)
            y_pos = range(len(features))
            
            bars = axes[1].barh(y_pos, freqs, color='#F1C40F', alpha=0.7)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])
            axes[1].set_xlabel('Occurrence Frequency')
            axes[1].set_title(f'Top-20 Stable Features (Top-{k}, Bootstrap)')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        
        plot_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
        plt.savefig(os.path.join(plot_dir, f'feature_frequency_top{k}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return sorted_features
    
    def _analyze_single_topk_stability(self, cancer_type, xai_method, rankings, k):
        """分析单个Top-k值的稳定性"""
        print(f"   📊 分析 Top-{k} 稳定性...")
        
        pairwise_results = self.analyze_pairwise_stability(rankings, k)
        stability_summary = self.create_stability_summary(pairwise_results)
        feature_freq = self.calculate_feature_frequency(rankings, k)
        random_baseline = self.calculate_random_baseline(k, self.total_features)
        
        # 稳定核心特征
        stable_cores = {
            threshold: {
                feature: freq for feature, freq in feature_freq.items() 
                if freq >= threshold
            }
            for threshold in self.stability_thresholds
        }
        
        default_threshold = self._get_default_threshold()
        if default_threshold in stable_cores:
            stable_core = stable_cores[default_threshold]
        else:
            stable_core = next(iter(stable_cores.values()), {})
        
        # 生成可视化
        self.plot_stability_distribution(cancer_type, xai_method, pairwise_results, k)
        sorted_features = self.plot_feature_frequency(cancer_type, xai_method, feature_freq, k)
        
        # 保存特征列表
        if sorted_features:
            stable_features_df = pd.DataFrame(sorted_features, columns=['feature_name', 'frequency'])
            stable_features_df['rank'] = range(1, len(stable_features_df) + 1)
            
            output_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
            stable_features_df.to_csv(
                os.path.join(output_dir, f'stable_features_top{k}.csv'), 
                index=False
            )
            
            # 保存各阈值的稳定核心特征
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
        
        return {
            'pairwise_results': pairwise_results,
            'stability_summary': stability_summary,
            'feature_frequency': feature_freq,
            'stable_features': feature_freq,
            'random_baseline': random_baseline,
            'stable_core': stable_core,
            'stable_cores': stable_cores,
            'n_pairs': len(pairwise_results['kuncheva']),
            'n_rankings': len(rankings)
        }
    
    def _analyze_all_topk_stability(self, cancer_type, xai_method, rankings):
        """分析所有Top-k值的稳定性"""
        stability_results = {}
        stability_results['rankings'] = rankings
        
        for k in self.top_k_values:
            stability_results[k] = self._analyze_single_topk_stability(
                cancer_type, xai_method, rankings, k
            )
        
        return stability_results
    
    def plot_stability_performance_correlation(self, cancer_type, xai_method, 
                                              performance_correlations, output_dir):
        """绘制特征稳定性与模型性能的相关性图"""
        if not performance_correlations:
            return
        
        n_topk = len(performance_correlations)
        if n_topk == 0:
            return
        
        n_cols = 2
        n_rows = (n_topk + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        if n_topk == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (k, corr_data) in enumerate(sorted(performance_correlations.items())):
            ax = axes[idx]
            
            experiment_df = pd.DataFrame(corr_data['experiment_data'])
            cindex_values = experiment_df['cindex'].values
            stability_values = experiment_df['avg_kuncheva'].values
            
            ax.scatter(stability_values, cindex_values, alpha=0.6, s=50, 
                      color='steelblue', edgecolors='black', linewidth=0.5)
            
            if len(stability_values) > 2:
                z = np.polyfit(stability_values, cindex_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(stability_values.min(), stability_values.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
            
            ax.set_xlabel('Feature Selection Stability (Avg Kuncheva)', fontsize=11)
            ax.set_ylabel('C-index', fontsize=11)
            ax.set_title(f'Top-{k} Features\n{cancer_type} - {xai_method} (Bootstrap)', 
                        fontsize=12, fontweight='bold')
            
            spearman_r = corr_data['spearman_r']
            spearman_p = corr_data['spearman_p']
            n_exp = corr_data['n_experiments']
            
            spearman_sig = "**" if spearman_p < 0.01 else ("*" if spearman_p < 0.05 else "ns")
            
            textstr = f'n = {n_exp}\n'
            textstr += f'Spearman ρ = {spearman_r:.3f} {spearman_sig}\n'
            textstr += f'p-value = {spearman_p:.4f}'
            
            props = dict(boxstyle='round', facecolor='#EFEFEF', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
        
        for idx in range(n_topk, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Feature Stability vs Model Performance\n{cancer_type} - {xai_method} (Bootstrap)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'stability_performance_correlation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 保存性能相关性图: stability_performance_correlation.png")
    
    def _analyze_and_save_performance_correlations(self, cancer_type, xai_method, 
                                                    stability_results, performance_data, output_dir):
        """分析并保存性能相关性结果"""
        performance_correlations = {}
        
        if performance_data:
            print(f"   📈 进行性能相关性分析...")
            performance_correlations = self.analyze_stability_performance_correlation(
                cancer_type, xai_method, stability_results, performance_data
            )
            
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
                
                self.plot_stability_performance_correlation(
                    cancer_type, xai_method, performance_correlations, output_dir
                )
        
        return performance_correlations
    
    def analyze_cancer_xai_combination(self, cancer_type, xai_method):
        """分析单个癌症-XAI组合的稳定性"""
        print(f"\n🔬 分析 {cancer_type} - {xai_method}")
        
        rankings = self.load_feature_rankings(cancer_type, xai_method)
        
        if len(rankings) < self.min_rankings_required:
            print(f"⚠️ 数据不足，跳过 {cancer_type}-{xai_method} "
                  f"(只有{len(rankings)}个文件，需要至少{self.min_rankings_required}个)")
            return None
        
        performance_data = self.load_model_performance_data(cancer_type)
        
        stability_results = self._analyze_all_topk_stability(cancer_type, xai_method, rankings)
        
        print(f"   📈 绘制Overlap@k曲线...")
        kuncheva_curve = self.plot_overlap_curves(cancer_type, xai_method, rankings)
        
        output_dir = os.path.join(self.output_dir, cancer_type, xai_method)
        kuncheva_curve.to_csv(os.path.join(output_dir, 'kuncheva_curve_data.csv'), index=False)
        
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
            'kuncheva_curve': kuncheva_curve
        }
    
    def create_summary_report(self, cancer_type, xai_method, analysis_result):
        """创建稳定性分析报告"""
        if analysis_result is None:
            return
        
        output_dir = self._ensure_dir(os.path.join(self.output_dir, cancer_type, xai_method))
        
        report_data = []
        
        for k in self.top_k_values:
            if k in analysis_result['stability_results']:
                result = analysis_result['stability_results'][k]
                summary = result['stability_summary']
                baseline = result['random_baseline']
                
                row = {
                    'cancer_type': cancer_type,
                    'xai_method': xai_method,
                    'top_k': k,
                    'n_rankings': result['n_rankings'],
                    'n_pairs': result['n_pairs'],
                }
                
                # 多阈值稳定核心特征
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
                
                report_data.append(row)
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(os.path.join(output_dir, 'stability_summary_report.csv'), index=False)
            
            self.create_text_report(cancer_type, xai_method, analysis_result, output_dir)
    
    def create_text_report(self, cancer_type, xai_method, analysis_result, output_dir):
        """创建文本格式的分析报告"""
        report_path = os.path.join(output_dir, 'stability_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Bootstrap特征重要性稳定性分析报告\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"癌症类型: {cancer_type}\n")
            f.write(f"XAI方法: {xai_method}\n")
            f.write(f"Bootstrap模型数量: {analysis_result['n_rankings']}\n")
            f.write(f"Bootstrap总次数: {self.num_bootstraps}\n")
            f.write(f"成功率: {analysis_result['n_rankings']/self.num_bootstraps*100:.1f}%\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"主要发现摘要\n")
            f.write(f"{'-'*30}\n")
            
            for k in self.top_k_values:
                if k in analysis_result['stability_results']:
                    result = analysis_result['stability_results'][k]
                    summary = result['stability_summary']
                    
                    f.write(f"\n📊 Top-{k} 稳定性分析:\n")
                    f.write(f"   Kuncheva指数: {summary.get('kuncheva_mean', 0):.3f} ± {summary.get('kuncheva_std', 0):.3f}\n")
                    
                    if 'stable_cores' in result:
                        f.write(f"   稳定核心特征数量:\n")
                        for threshold in sorted(result['stable_cores'].keys(), reverse=True):
                            n_features = len(result['stable_cores'][threshold])
                            f.write(f"     - 阈值≥{int(threshold*100)}%: {n_features}个\n")
                    else:
                        f.write(f"   稳定核心特征数量 (频次≥{int(self._get_default_threshold()*100)}%): {len(result['stable_core'])}\n")
                    
                    baseline = result['random_baseline']
                    kuncheva_improvement = summary.get('kuncheva_mean', 0) - baseline['kuncheva_mean']
                    f.write(f"   超越随机基线 (Kuncheva): +{kuncheva_improvement:.3f}\n")
            
            f.write(f"\n\n详细统计信息\n")
            f.write(f"{'-'*30}\n")
            
            for k in self.top_k_values:
                if k in analysis_result['stability_results']:
                    result = analysis_result['stability_results'][k]
                    summary = result['stability_summary']
                    
                    f.write(f"\n🔍 Top-{k} 详细分析:\n")
                    f.write(f"   稳定性指标统计:\n")
                    for metric in ['kuncheva']:
                        mean_val = summary.get(f'{metric}_mean', np.nan)
                        median_val = summary.get(f'{metric}_median', np.nan)
                        q25_val = summary.get(f'{metric}_q25', np.nan)
                        q75_val = summary.get(f'{metric}_q75', np.nan)
                        
                        if not np.isnan(mean_val):
                            f.write(f"     {metric}: 均值={mean_val:.3f}, 中位数={median_val:.3f}, IQR=[{q25_val:.3f}, {q75_val:.3f}]\n")
                    
                    if 'stable_cores' in result:
                        f.write(f"   稳定核心特征分析:\n")
                        for threshold in sorted(result['stable_cores'].keys(), reverse=True):
                            stable_features = sorted(result['stable_cores'][threshold].items(), 
                                                   key=lambda x: x[1], reverse=True)
                            f.write(f"\n   阈值≥{int(threshold*100)}% (共{len(stable_features)}个特征):\n")
                            
                            for i, (feature, freq) in enumerate(stable_features[:10]):
                                f.write(f"     {i+1:2d}. {feature}: {freq:.2f}\n")
                            
                            if len(stable_features) > 10:
                                f.write(f"     ... (还有{len(stable_features)-10}个特征)\n")
            
            f.write(f"\n\n解读说明\n")
            f.write(f"{'-'*30}\n")
            f.write(f"• Kuncheva指数: >0表示超越随机期望\n")
            f.write(f"• 稳定核心特征: 在一定比例的bootstrap模型中都被选为Top-k的特征\n")
            f.write(f"  - 阈值{self.stability_thresholds}: 分别表示在≥{[int(t*100) for t in self.stability_thresholds]}%的模型中出现\n")
            f.write(f"• Bootstrap: 通过重抽样生成{self.num_bootstraps}个不同的训练集，评估特征选择的稳健性\n")
    
    def run_full_analysis(self, cancer_types=None):
        """运行完整的稳定性分析"""
        print(f"🚀 开始Bootstrap特征稳定性分析")
        print(f"   分析 {len(self.xai_methods)} 种XAI方法")
        print(f"   Bootstrap次数: {self.num_bootstraps}")
        print(f"   Top-k设置: {self.top_k_values}")
        
        if cancer_types is None:
            # 自动检测癌症类型（从results_bootstrap目录）
            bootstrap_dir = os.path.join(self.results_dir, 'results_bootstrap')
            if os.path.exists(bootstrap_dir):
                cancer_types = [d for d in os.listdir(bootstrap_dir) 
                              if os.path.isdir(os.path.join(bootstrap_dir, d))]
                cancer_types.sort()
            else:
                print(f"⚠️ Bootstrap结果目录不存在: {bootstrap_dir}")
                return
        
        print(f"   发现 {len(cancer_types)} 种癌症类型: {', '.join(cancer_types)}")
        
        all_results = []
        cancer_xai_results = {cancer: {} for cancer in cancer_types}
        
        for cancer_type in cancer_types:
            for xai_method in self.xai_methods:
                result = self.analyze_cancer_xai_combination(cancer_type, xai_method)
                
                if result is not None:
                    all_results.append(result)
                    cancer_xai_results[cancer_type][xai_method] = result
                    self.create_summary_report(cancer_type, xai_method, result)
        
        # 跨XAI方法一致性分析
        print(f"\n🔗 开始跨XAI方法一致性分析...")
        cross_xai_results = {}
        
        for cancer_type in cancer_types:
            if len(cancer_xai_results[cancer_type]) >= 2:
                print(f"\n📊 分析 {cancer_type} 跨XAI一致性...")
                
                xai_stability_data = {}
                for xai_method, result in cancer_xai_results[cancer_type].items():
                    if 'stability_results' in result:
                        xai_stability_data[xai_method] = result['stability_results']
                
                if len(xai_stability_data) >= 2:
                    cancer_cross_xai = {}
                    for k in self.top_k_values:
                        consistency = self.analyze_cross_xai_consistency(
                            cancer_type, xai_stability_data, k, threshold=None
                        )
                        if consistency:
                            cancer_cross_xai[k] = consistency
                    
                    if cancer_cross_xai:
                        cross_xai_results[cancer_type] = cancer_cross_xai
                        self.save_cross_xai_results(cancer_type, cancer_cross_xai)
        
        if cross_xai_results:
            self.create_cross_xai_summary_report(cross_xai_results)
        
        if all_results:
            self.create_cross_cancer_summary(all_results)
        
        print(f"\n🎉 Bootstrap稳定性分析完成!")
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
                        'success_rate': n_rankings / self.num_bootstraps,
                        'kuncheva_mean': summary.get('kuncheva_mean', np.nan),
                        'kuncheva_std': summary.get('kuncheva_std', np.nan),
                        'n_stable_core': len(stability_result['stable_core'])
                    }
                    
                    summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.output_dir, 'cross_cancer_stability_summary.csv'), 
                             index=False)
            
            self.create_cross_cancer_plots(summary_df)
            self.create_cross_cancer_text_report(summary_df)
    
    def create_cross_cancer_plots(self, summary_df):
        """创建跨癌症汇总可视化"""
        xai_methods = summary_df['xai_method'].unique()
        n_methods = len(xai_methods)
        
        comparison_topk_values = [100, 200, 300]
        
        for topk_value in comparison_topk_values:
            print(f"   生成 Top-{topk_value} 跨癌症对比图...")
            
            fig, axes = plt.subplots(n_methods, 2, figsize=(14, 6*n_methods))
            if n_methods == 1:
                axes = axes.reshape(1, -1)
            
            for i, xai_method in enumerate(xai_methods):
                method_data = summary_df[summary_df['xai_method'] == xai_method]
                topk_data = method_data[method_data['top_k'] == topk_value]
                
                if not topk_data.empty:
                    cancer_order = topk_data.sort_values('kuncheva_mean')['cancer_type']
                    
                    # Kuncheva对比
                    axes[i,0].bar(range(len(cancer_order)), 
                                 topk_data.set_index('cancer_type').loc[cancer_order]['kuncheva_mean'],
                                 yerr=topk_data.set_index('cancer_type').loc[cancer_order]['kuncheva_std'],
                                 capsize=3, alpha=0.7, color='#E74C3C')
                    axes[i,0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                    axes[i,0].set_xticks(range(len(cancer_order)))
                    axes[i,0].set_xticklabels(cancer_order, rotation=45, ha='right')
                    axes[i,0].set_ylabel('Kuncheva Index')
                    axes[i,0].set_title(f'{xai_method} - Top-{topk_value} Kuncheva Stability (Bootstrap)')
                    
                    # 稳定核心特征数量对比
                    # 稳定核心特征数量对比（折线图，显示所有阈值）
                    # 准备数据：每个阈值下每种癌症的稳定核心特征数量
                    thresholds_pct = [int(t*100) for t in self.stability_thresholds]  # [70, 75, 80, 85, 90]
                    threshold_labels = [f'{t}%' for t in thresholds_pct]
                    
                    # 获取所有癌症类型
                    all_cancers = sorted(topk_data['cancer_type'].unique())
                    
                    # 为每种癌症绘制折线
                    # 使用指定的颜色列表循环配色，配合线型区分15种癌症
                    colors = ['#8E44AD', '#1D80C1', '#27AE60', '#F1C40F',
                            '#E67E22', '#E74C3C', '#1ABC9C', '#D3C0A3']
                    linestyles = ['-', '--', ':', '-.']
                    num_linestyles = len(linestyles)
                    
                    for idx, cancer in enumerate(all_cancers):
                        cancer_row = topk_data[topk_data['cancer_type'] == cancer].iloc[0]
                        
                        # 提取各阈值的稳定核心特征数量
                        n_stable_values = []
                        for threshold_pct in thresholds_pct:
                            col_name = f'n_stable_core_{threshold_pct}pct'
                            if col_name in cancer_row:
                                n_stable_values.append(cancer_row[col_name])
                            else:
                                n_stable_values.append(0)
                        
                        # 绘制折线（使用指定的颜色列表循环，配合线型区分）
                        # 颜色按索引循环使用8种颜色
                        color = colors[idx % len(colors)]
                        # 线型按颜色组切换：每8种颜色换一种线型
                        linestyle_idx = idx // len(colors)
                        linestyle = linestyles[linestyle_idx % num_linestyles]
                        axes[i,1].plot(threshold_labels, n_stable_values, 
                                      marker='o', markersize=3, linewidth=1.5, 
                                      alpha=0.8, color=color, linestyle=linestyle, label=cancer)
                    
                    axes[i,1].set_xlabel('Stability Threshold', fontsize=10)
                    axes[i,1].set_ylabel('Number of Stable Core Features', fontsize=10)
                    axes[i,1].set_title(f'{xai_method} - Top-{topk_value} Stable Core Features\nAcross Thresholds', fontsize=11)
                    axes[i,1].grid(True, axis='y', alpha=0.3, linestyle='--')

                     # 调整图例显示：放在图的外部（右侧）
                    n_cancers = len(all_cancers)
                    if n_cancers <= 8:
                        fontsize_legend = 8
                    elif n_cancers <= 15:
                        fontsize_legend = 7
                    else:
                        fontsize_legend = 6
                    
                    # 将图例放在图的外部右侧（单列显示）
                    axes[i,1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                                   ncol=1, framealpha=0.9, fontsize=fontsize_legend,
                                   columnspacing=0.8, handlelength=1.5, handletextpad=0.5)
                    axes[i,1].set_xticks(threshold_labels)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'cross_cancer_stability_comparison_top{topk_value}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_cross_cancer_text_report(self, summary_df):
        """创建跨癌症文本汇总报告"""
        report_path = os.path.join(self.output_dir, 'cross_cancer_summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Bootstrap跨癌症特征稳定性汇总报告\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"分析概述:\n")
            f.write(f"  XAI方法: {', '.join(summary_df['xai_method'].unique())}\n")
            f.write(f"  癌症类型: {', '.join(sorted(summary_df['cancer_type'].unique()))}\n")
            f.write(f"  Bootstrap次数: {self.num_bootstraps}\n")
            f.write(f"  Top-k设置: {sorted(summary_df['top_k'].unique())}\n")
            f.write(f"  生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            comparison_topk_values = [100, 200, 300]
            
            for xai_method in summary_df['xai_method'].unique():
                method_data = summary_df[summary_df['xai_method'] == xai_method]
                
                f.write(f"🔬 {xai_method.upper()} 方法稳定性分析 (Bootstrap)\n")
                f.write(f"{'-'*40}\n")
                
                for topk_value in comparison_topk_values:
                    topk_data = method_data[method_data['top_k'] == topk_value].copy()
                    
                    if not topk_data.empty:
                        topk_data = topk_data.sort_values('kuncheva_mean', ascending=False)
                        
                        f.write(f"\n【Top-{topk_value}特征稳定性排名】 (按Kuncheva指数):\n")
                        for i, (_, row) in enumerate(topk_data.iterrows()):
                            f.write(f"  {i+1:2d}. {row['cancer_type']}: Kuncheva={row['kuncheva_mean']:.3f}±{row['kuncheva_std']:.3f}, 核心特征={row['n_stable_core']}个 "
                                  f"(成功率={row['success_rate']*100:.1f}%)\n")
                        
                        f.write(f"\n  统计摘要:\n")
                        f.write(f"    Kuncheva指数: {topk_data['kuncheva_mean'].mean():.3f} ± {topk_data['kuncheva_mean'].std():.3f} "
                              f"(范围: {topk_data['kuncheva_mean'].min():.3f} - {topk_data['kuncheva_mean'].max():.3f})\n")
                        f.write(f"    稳定核心特征: {topk_data['n_stable_core'].mean():.1f} ± {topk_data['n_stable_core'].std():.1f}个\n")
                        f.write(f"    平均成功率: {topk_data['success_rate'].mean()*100:.1f}%\n")
                
                f.write(f"\n")
            
            f.write(f"\nBootstrap特点:\n")
            f.write(f"{'-'*20}\n")
            f.write(f"• Bootstrap通过有放回抽样创建{self.num_bootstraps}个训练集\n")
            f.write(f"• 每个训练集约包含63.2%的原始样本(平均)\n")
            f.write(f"• Out-of-Bag(OOB)样本用于评估，约36.8%的原始样本\n")
            f.write(f"• 成功率表示成功生成特征排名的bootstrap模型比例\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap特征重要性稳定性分析")
    parser.add_argument('--results_dir', type=str, default='/home/zuoyiyi/SNN/TCGA', 
                       help='结果根目录（包含{xai}_bootstrap_results/子目录）')
    parser.add_argument('--output_dir', type=str, default='/home/zuoyiyi/SNN/TCGA/stability_analysis_bootstrap', 
                       help='稳定性分析结果输出目录')
    parser.add_argument('--num_bootstraps', type=int, default=100, 
                       help='Bootstrap迭代次数（100）')
    args = parser.parse_args()
    
    # 设置XAI方法和癌症类型
    xai_methods = ['shap', 'IG', 'LRP', 'PFI', 'deepshap', 'DeepLIFT']  # deepshap可能失败率高，谨慎添加
    cancer_types = [
        'BLCA', 'BRCA', 'COADREAD', 'GBMLGG', 'HNSC', 'KIRC',
        'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'PAAD', 'SKCM', 'STAD', 'UCEC'
    ]

    print(f"🎯 Bootstrap稳定性分析配置:")
    print(f"   XAI方法: {', '.join(xai_methods)}")
    print(f"   癌症类型: {', '.join(cancer_types)}")
    print(f"   Bootstrap次数: {args.num_bootstraps}")
    print(f"   根目录: {args.results_dir}")
    print(f"   输出目录: {args.output_dir}")

    # 创建分析器
    analyzer = BootstrapFeatureStabilityAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        xai_methods=xai_methods,
        num_bootstraps=args.num_bootstraps
    )
    
    # 运行分析
    analyzer.run_full_analysis(cancer_types=cancer_types)
    
    print(f"\n🎉 Bootstrap综合稳定性分析完成！")
    print(f"\n📂 主要结果文件:")
    print(f"   📁 主目录: {args.output_dir}")
    print(f"   📋 跨癌症汇总: cross_cancer_summary_report.txt")
    print(f"   📊 稳定性数据: cross_cancer_stability_summary.csv")
    print(f"   🔗 跨XAI一致性: cross_xai_consistency_report.txt")
    print(f"\n🔬 各癌症详细结果:")
    print(f"   📁 [CANCER]/[XAI_METHOD]/")
    print(f"      ├── stability_summary_report.csv")
    print(f"      ├── performance_correlations.csv")
    print(f"      ├── stability_performance_correlation.png")
    print(f"      ├── overlap_curves.png")
    print(f"      ├── stable_features_top*.csv")
    print(f"      ├── stable_core_top*_threshold70.csv")
    print(f"      └── stable_core_top*_threshold80.csv")


if __name__ == "__main__":
    main()

