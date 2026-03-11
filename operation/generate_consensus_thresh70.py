
import os
import pandas as pd
from collections import Counter, defaultdict

# --- 配置区域 ---
# 结果根目录
BASE_DIR = '/home/zuoyiyi/SNN/TCGA/stability_analysis_4'
# XAI方法列表（注意大小写需与目录名匹配）
XAI_METHODS = ['shap', 'IG', 'LRP', 'PFI', 'deepshap', 'DeepLIFT']
# 需要处理的 Top-k 值
TOP_K_VALUES = [20, 50, 100, 200, 300]
# 目标阈值
TARGET_THRESHOLD = 0.70
# ----------------

def process_cancer_type(cancer_type):
    cancer_dir = os.path.join(BASE_DIR, cancer_type)
    output_dir = os.path.join(cancer_dir, 'cross_xai_analysis')
    
    if not os.path.exists(cancer_dir):
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"处理癌症类型: {cancer_type}")

    for k in TOP_K_VALUES:
        method_stable_features = {}
        
        # 1. 收集各 XAI 方法在当前 Top-k 下的稳定特征
        for method in XAI_METHODS:
            # 对应的稳定特征文件路径
            feature_file = os.path.join(cancer_dir, method, f'stable_features_top{k}.csv')
            
            if os.path.exists(feature_file):
                try:
                    df = pd.read_csv(feature_file)
                    # 关键步骤：根据 70% 阈值进行筛选
                    # 假设文件中包含 'frequency' 列
                    if 'frequency' in df.columns and 'feature_name' in df.columns:
                        stable_features = df[df['frequency'] >= TARGET_THRESHOLD]['feature_name'].tolist()
                        if stable_features:
                            method_stable_features[method] = set(stable_features)
                except Exception as e:
                    print(f"  [警告] 读取 {method} 失败: {e}")
        
        # 2. 如果收集到的方法少于2个，无法计算共识，跳过
        if len(method_stable_features) < 2:
            continue

        # 3. 计算共识
        feature_counts = Counter()
        feature_methods = defaultdict(list)
        
        for method, features in method_stable_features.items():
            for feature in features:
                feature_counts[feature] += 1
                feature_methods[feature].append(method)
        
        # 4. 排序和构建结果 DataFrame
        # 按被多少个方法选中降序排序
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        result_rows = []
        for feature, count in sorted_features:
            methods = sorted(feature_methods[feature])
            result_rows.append({
                'feature': feature,
                'n_methods': count,
                'selected_by_methods': ', '.join(methods)
            })
            
        # 5. 保存结果文件 1: 详细排序列表
        if result_rows:
            output_df = pd.DataFrame(result_rows)
            filename = f'features_by_consensus_top{k}_thresh{int(TARGET_THRESHOLD*100)}.csv'
            save_path = os.path.join(output_dir, filename)
            output_df.to_csv(save_path, index=False)
            print(f"  ✅ [Top-{k}] 生成详细表: {filename} (共 {len(result_rows)} 个特征)")
            
        # 6. 保存结果文件 2: 纯共识特征列表 (所有方法的交集)
        # 注意：这里计算的是所有有稳定特征的方法的交集
        if method_stable_features:
            all_feature_sets = list(method_stable_features.values())
            consensus_features = set.intersection(*all_feature_sets)
            
            if consensus_features:
                consensus_df = pd.DataFrame({'consensus_feature': list(consensus_features)})
                consensus_filename = f'consensus_features_top{k}_thresh{int(TARGET_THRESHOLD*100)}.csv'
                consensus_path = os.path.join(output_dir, consensus_filename)
                consensus_df.to_csv(consensus_path, index=False)
                print(f"  ✅ [Top-{k}] 生成共识表: {consensus_filename} (共 {len(consensus_features)} 个完全共识特征)")

def main():
    if not os.path.exists(BASE_DIR):
        print(f"错误: 目录 {BASE_DIR} 不存在")
        return

    # 获取目录下所有的癌症类型文件夹
    cancer_types = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    print(f"开始处理 {len(cancer_types)} 个癌症类型...")
    print(f"目标阈值: {TARGET_THRESHOLD * 100}%")
    
    for cancer in sorted(cancer_types):
        process_cancer_type(cancer)
        
    print("\n处理完成！")

if __name__ == "__main__":
    main()
