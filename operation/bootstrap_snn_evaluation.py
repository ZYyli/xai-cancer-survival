"""
bootstrap_snn_evaluation.py  #1628074

用途：
- 对每个癌种进行 B 次 bootstrap 重训练（有放回抽样）
- 每轮训练在 D_boot 上训练模型，在未抽中的 OOB 上评估
- 在 OOB 上计算 C-index（sksurv）
- 汇总每种癌症的中位数/均值/Std/95%CI 并保存结果

注意：
- 请实现 `fit_snn_on_bootstrap(train_df, seed)` 和 `predict_risk_scores(model, df_oob)` 两个函数：
  - fit_snn_on_bootstrap: 接受训练 DataFrame，返回已训练模型
  - predict_risk_scores: 返回 OOB 数据对应的 risk_score numpy array
"""

import torch, os, time, random, math, pickle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
from torch.utils.data import DataLoader
from dataset_survival import RNAseqSurvivalDataset
from core_utils import train, summary_survival_omic
from model_genomic import SNN
import numpy as np
from argparse import Namespace
import pandas as pd

from multiprocessing import Process
import multiprocessing as mp
from sksurv.metrics import concordance_index_censored  # returns (cindex, concordant, permissible, tied_risk)
# OR: from lifelines.utils import concordance_index, but sksurv better handles censoring ties

# ---------------------------
# ========== 配置 ==========
# ---------------------------
B_DEFAULT = 500 #500次boostrap
N_JOBS = 1  # 并行 worker 数量（根据你机器 CPU/GPU 调整）
OUTPUT_ROOT = "/home/zuoyiyi/SNN/TCGA/results_bootstrap_500"
RNG_SEED = 1

# ========== 文件保存配置 - 精简版 ==========
SAVE_CONFIG = {
    'save_models': True,            # ✅ 模型权重 - 复现和可解释性分析
    'save_individual_indices': True, # ✅ 训练集/OOB索引 - 完全复现必需
    'save_result_arrays': True,     # ✅ cindex/pvalue数组 - 统计分析必需
}

# ========== 必须设定全局随机种子保证可复现 ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# ========== 工具函数 ==========
# ---------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def compute_cindex_sksurv(oob_times, oob_events, risk_scores):
    """
    sksurv.metrics.concordance_index_censored 参数：
      event_indicator: bool array, True if event occurred
      event_time: numeric array
      estimate (risk): numeric (higher = worse risk)
    返回值第一个元素是 cindex
    """
    # sksurv expects structured arrays: (event, time) as inputs to the function
    # but concordance_index_censored accepts (event_indicator, event_time, estimate)
    # ensure types:
    event_bool = (1 - np.asarray(oob_events)).astype(bool)
    times = np.asarray(oob_times).astype(float)
    est = np.asarray(risk_scores).astype(float)
    try:
        res = concordance_index_censored(event_bool, times, est)
        return float(res[0])
    except Exception as e:
        # 兜底：若出错可返回 nan
        print("C-index computation error:", e)
        return np.nan

# ---------------------------
# ========== 用户需实现的模型接口 ==========
# ---------------------------

def split_train_val_stratified(df, val_frac=0.1, event_col='censorship', seed=None):
    """
    分层划分训练集和内部验证集
    """
    np.random.seed(seed)
    train_idx, val_idx = [], []
    for lbl in df[event_col].unique():
        idx_lbl = df.index[df[event_col] == lbl].to_numpy()
        n_val = max(1, int(len(idx_lbl) * val_frac))
        idx_lbl = np.random.permutation(idx_lbl)
        val_idx.extend(idx_lbl[:n_val])
        train_idx.extend(idx_lbl[n_val:])
    return train_idx, val_idx

def fit_snn_on_bootstrap(D_boot, seed=None, save_model_path=None, device='cuda:0', val_frac=0.1, cancer_name="unknown"):
    """
    在 D_boot 上训练 SNN 模型
    - 内部划分 train/val 用于早停
    """
    if seed is not None:
        set_seed(seed)

    # 内部分层划分训练集/验证集
    train_idx, val_idx = split_train_val_stratified(D_boot, val_frac=val_frac, event_col='censorship', seed=seed)
    df_train = D_boot.loc[train_idx].reset_index(drop=True)
    df_val   = D_boot.loc[val_idx].reset_index(drop=True)

    # Dataset / DataLoader
    train_dataset = RNAseqSurvivalDataset(df_train, label_col="survival_months", seed=seed)
    val_dataset   = RNAseqSurvivalDataset(df_val, label_col="survival_months", seed=seed)

    g_train = torch.Generator(); g_val = torch.Generator()
    if seed is not None:
        g_train.manual_seed(seed); g_val.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    # 设置模型参数
    args = Namespace(
        omic_input_dim=train_dataset.features.shape[1],
        model_size_omic="small",
        n_classes=4,
        lr=2e-4,
        reg=1e-5,
        lambda_reg=1e-5,
        batch_size=1,
        max_epochs=33,
        gc=32,
        alpha_surv=0.0,
        results_dir=f"/home/zuoyiyi/SNN/TCGA/results_bootstrap/tmp_{cancer_name}_seed{seed}_gpu{device.split(':')[1]}",  # 每个癌症+seed+GPU独立目录
        log_data=False,
        early_stopping=True,
        seed=seed
    )
    os.makedirs(args.results_dir, exist_ok=True)

    # 训练模型：train_loader 训练，val_loader 内部验证用于早停
    # bootstrap场景下不需要test_loader，传入val_loader作为占位符
    _, _ = train(train_loader, val_loader, val_loader, 0, args, seed=seed)
    
    # 从 core_utils.py 可以看到，需要重新初始化模型来获取模型对象
    model = SNN(
        omic_input_dim=args.omic_input_dim,
        model_size_omic=args.model_size_omic,
        n_classes=args.n_classes
    )
    
    # 加载最佳检查点
    best_cindex_ckpt = os.path.join(args.results_dir, f"s_0_best_cindex_checkpoint.pt")
    best_loss_ckpt = os.path.join(args.results_dir, f"s_0_best_loss_checkpoint.pt")
    
    if os.path.exists(best_cindex_ckpt):
        model.load_state_dict(torch.load(best_cindex_ckpt, map_location=device))
    elif os.path.exists(best_loss_ckpt):
        model.load_state_dict(torch.load(best_loss_ckpt, map_location=device))
    else:
        print("Warning: No checkpoint found, using random initialized model")
    model.to(device)
    
    # 保存模型
    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    # 自动清理临时训练目录
    import shutil
    try:
        if os.path.exists(args.results_dir):
            shutil.rmtree(args.results_dir)
            print(f"✅ 已清理临时目录: {args.results_dir}")
    except Exception as e:
        print(f"⚠️ 清理临时目录失败: {e}")

    return model

def predict_risk_scores(model, df, device='cuda:0'):
    """
    给定 model 和新数据 df，返回风险分数（越大风险越高）
    """
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()  # 切换到 eval 模式

    dataset = RNAseqSurvivalDataset(
        data=df,
        label_col="survival_months",  # 和你训练时保持一致
        seed=1
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

    # 手动计算风险分数，确保设备一致性
    all_risk_scores = []
    
    with torch.no_grad():
        for batch_idx, (x, label, time, c) in enumerate(loader):
            # 确保所有tensor都在同一设备上
            x = x.to(device)
            
            # 模型预测
            h = model(x_omic=x)
            
            # 计算风险分数（与core_utils.py中一致）
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1)
            
            all_risk_scores.append(risk.cpu().numpy())
    
    risk_scores = np.concatenate(all_risk_scores)
    return risk_scores

# ---------------------------
# ========== 单次 Bootstrap 迭代（用于并行）==========
# ---------------------------

def single_bootstrap_iteration(D_df, seed, stratify=True, time_col='survival_months', event_col='censorship', label_col='label', model_dir=None, device='cuda:0', cancer_name="unknown"):
    """
    - 从 D_df (原始整个癌种数据) 做有放回抽样 -> train_idx
    - 得到 OOB idx -> oob_df
    - 训练模型（fit_snn_on_bootstrap）
    - 用模型对 oob_df 做预测 risk_scores
    - 在 OOB 上计算 c-index
    返回 dict { 'cindex':... }
    """
    n = len(D_df)
    if n == 0:
        return {'cindex': np.nan}

    # 分层抽样（按 event 保持比例）— 提升稳定性，尤其事件稀少时
    if stratify and label_col in D_df.columns:
        train_idx = []
        for lbl in D_df[label_col].unique():
            idx_lbl = D_df.index[D_df[label_col] == lbl].to_numpy()
            draw_lbl = np.random.choice(idx_lbl, size=len(idx_lbl), replace=True)
            train_idx.extend(draw_lbl)
        train_idx = np.unique(train_idx)
    else:
        train_idx = np.unique(np.random.choice(D_df.index.to_numpy(), size=n, replace=True))


    idx_all = D_df.index.to_numpy()
    oob_indices = np.setdiff1d(idx_all, train_idx, assume_unique=True)

    # 诊断 OOB 比例
    oob_ratio = len(oob_indices) / n
    print(f"[Bootstrap] 样本总数={n}, OOB={len(oob_indices)} ({oob_ratio:.2%})")
    # 如果 OOB 过少，重采样
    min_oob_ratio = 0.1  # 至少 10% 的样本落到 OOB
    if oob_ratio < min_oob_ratio:
        print(f"[Bootstrap] OOB 比例过低 ({oob_ratio:.2%})，跳过该次抽样")
        return single_bootstrap_iteration(D_df, seed=seed, stratify=stratify, time_col=time_col, event_col=event_col, label_col=label_col, model_dir=model_dir, device=device, cancer_name=cancer_name)

    D_boot = D_df.loc[train_idx].reset_index(drop=True)
    D_oob  = D_df.loc[oob_indices].reset_index(drop=True)

    save_path = None
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"bootstrap_model_seed{seed}.pt")

    model = fit_snn_on_bootstrap(D_boot, seed=seed, save_model_path=save_path, device=device, val_frac=0.1, cancer_name=cancer_name)

    # 预测 OOB 的 risk scores
    try:
        risk_oob = predict_risk_scores(model, D_oob, device=device)
    except Exception as e:
        print("predict_risk_scores error:", e)
        return {'cindex': np.nan}

    # 计算 c-index（sksurv）
    cidx = compute_cindex_sksurv(D_oob[time_col].values, D_oob[event_col].values, risk_oob)

    #每次 bootstrap 结束后手动释放模型
    del model
    torch.cuda.empty_cache()

    # 始终返回必需信息用于复现和可解释性分析
    result = {
        'cindex': cidx, 
        'seed': seed, 
        'n_oob': len(oob_indices),
        'train_idx': train_idx, 
        'oob_idx': oob_indices, 
        'risk_scores': risk_oob
    }
        
    return result

# ---------------------------
# ========== 主流程：单癌种 Bootstrap ==========
# ---------------------------

def worker(seed, gpu_id, D_df, stratify, time_col, event_col, model_dir, cancer_name):
        torch.cuda.set_device(int(gpu_id))   # 🔹 强制当前进程使用指定 GPU，确保是整数
        return single_bootstrap_iteration(
            D_df,
            seed=seed,
            device=f'cuda:{gpu_id}',
            stratify=stratify,
            time_col=time_col,
            event_col=event_col,
            model_dir=model_dir,
            cancer_name=cancer_name
        )

def gpu_worker_loop(gpu_id, seeds_subset, D_df, stratify, time_col, event_col, label_col, model_dir, out_dir, cancer_name, result_queue):
    """
    在单个 GPU 上顺序执行 seeds_subset 中的 bootstrap 任务。
    结果通过 result_queue 返回给主进程
    """
    # 在子进程内设置 GPU
    import torch
    torch.cuda.set_device(int(gpu_id))  # 确保gpu_id是整数类型
    print(f"[GPU worker] PID {os.getpid()} -> using cuda:{gpu_id}, seeds: {seeds_subset}")

    for seed in seeds_subset:
        try:
            res = single_bootstrap_iteration(
                D_df,
                seed=seed,
                stratify=stratify,
                time_col=time_col,
                event_col=event_col,
                label_col=label_col,
                model_dir=model_dir,
                device=f'cuda:{gpu_id}',
                cancer_name=cancer_name
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] seed {seed} failed with exception: {e}")
            res = {'cindex': np.nan, 'seed': seed,
                   'train_idx': [], 'oob_idx': [], 'risk_scores': np.array([]),
                   'n_oob': 0}
        
        # 将结果放入队列，供主进程收集
        result_queue.put(res)
        print(f"[GPU {gpu_id}] seed {seed} completed, result sent to queue")

def run_bootstrap_for_cancer(cancer_name, D_df, B, n_jobs=1, output_root=OUTPUT_ROOT, 
                             stratify=True, seed_base=RNG_SEED, time_col='survival_months', 
                             event_col='censorship', label_col='label'):
    """
    对单个癌种执行 B 次 bootstrap
    按 GPU 启动固定子进程，每个子进程顺序跑一组 seed（避免 GPU 争用）
    返回 summary dict 并保存详细结果到磁盘
    """
    out_dir = os.path.join(output_root, cancer_name)
    ensure_dir(out_dir)

    # 模型保存目录
    model_dir = os.path.join(out_dir, "models")
    ensure_dir(model_dir)

    # 指定可用 GPU（你可以按需修改）
    gpu_ids = [0, 1]  # <-- 根据机器调整
    n_gpus = len(gpu_ids)
    if n_gpus == 0:
        raise RuntimeError("No GPUs declared in gpu_ids")

    # seeds 列表
    seeds = [int(seed_base + i) for i in range(B)]
    # 将 seeds 均匀分配给每个 GPU
    seed_sublists = np.array_split(seeds, n_gpus)

    # 创建结果队列用于进程间通信
    from multiprocessing import Queue
    result_queue = Queue()

    # 启动每张 GPU 的进程
    processes = []
    for gpu_id, seeds_sub in zip(gpu_ids, seed_sublists):
        seeds_sub = [int(s) for s in seeds_sub]  # 确保seeds也是整数
        if len(seeds_sub) == 0:
            continue
        p = Process(
            target=gpu_worker_loop,
            args=(int(gpu_id), seeds_sub, D_df, stratify, time_col, event_col, label_col, model_dir, out_dir, cancer_name, result_queue)
        )
        p.start()
        processes.append(p)
        print(f"[Main] started process PID={p.pid} for GPU {gpu_id} with {len(seeds_sub)} seeds")

    # 从队列中收集结果
    results = []
    total_expected = len(seeds)
    
    print(f"[Main] Collecting {total_expected} results from queue...")
    for i in range(total_expected):
        try:
            result = result_queue.get(timeout=3600)  # 1小时超时
            results.append(result)
            print(f"[Main] Collected result {i+1}/{total_expected} for seed {result.get('seed', 'unknown')}")
        except Exception as e:
            print(f"[Main] Failed to get result {i+1}: {e}")
            # 添加占位结果
            results.append({'cindex': np.nan, 'seed': -1, 'train_idx': [], 'oob_idx': [], 'risk_scores': np.array([]), 'n_oob': 0})

    # 等待全部进程完成
    for p in processes:
        p.join()
        print(f"[Main] process PID={p.pid} finished, exitcode={p.exitcode}")
    
    # 按seed排序结果
    results.sort(key=lambda x: x.get('seed', -1))

    # 结果已经通过队列收集完成

    # 提取 arrays
    c_list = np.array([r['cindex'] for r in results], dtype=float)

    # 过滤 nan
    c_valid = c_list[~np.isnan(c_list)]

    def summarize(arr):
        if len(arr) == 0:
            return {'median': np.nan, 'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'n': 0}
        med = float(np.nanmedian(arr))
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        lower, upper = np.percentile(arr, [2.5, 97.5])
        return {'median': med, 'mean': mean, 'std': std, 'ci_lower': float(lower), 'ci_upper': float(upper), 'n': len(arr)}

    c_sum = summarize(c_valid)

    summary = {
        'cindex_summary': c_sum,
        'B_requested': B,
        'B_effective_cindex': int(c_sum['n'])
    }

    # 根据配置保存详细数组与汇总
    if SAVE_CONFIG['save_result_arrays']:
        np.save(os.path.join(out_dir, 'cindex_array.npy'), c_list)

    # 写入可读 CSV
    summary_flat = {
        'Cancer': cancer_name,
        'B_requested': B,
        'c_median': c_sum['median'], 'c_mean': c_sum['mean'], 'c_std': c_sum['std'],
        'c_ci_lower': c_sum['ci_lower'], 'c_ci_upper': c_sum['ci_upper'], 'c_n': c_sum['n']
    }
    df_summary = pd.DataFrame([summary_flat])
    df_summary.to_csv(os.path.join(out_dir, 'summary_metrics.csv'), index=False)

    # 处理详细结果，选代表性模型
    results_detailed = [r for r in results if not np.isnan(r['cindex'])]

    # 根据配置决定是否保存 seed + train_idx/oob_idx + risk_scores
    if SAVE_CONFIG['save_individual_indices']:
        # 保存所有seeds到一个文件 - 更简洁
        all_seeds = [r['seed'] for r in results_detailed]
        np.save(os.path.join(model_dir, 'all_seeds.npy'), all_seeds)
        
        for r in results_detailed:
            seed = r['seed']
            # 保存训练集和OOB集索引 - 用于完全复现
            np.save(os.path.join(model_dir, f"train_idx_seed{seed}.npy"), r.get('train_idx', []))
            np.save(os.path.join(model_dir, f"oob_idx_seed{seed}.npy"), r.get('oob_idx', []))
            # 保存OOB风险分数 - 用于可解释性分析
            np.save(os.path.join(model_dir, f"risk_oob_seed{seed}.npy"), r.get('risk_scores', np.array([])))

    # 最终清理：删除所有该癌种的临时训练目录
    import shutil
    import glob
    temp_pattern = f"/home/zuoyiyi/SNN/TCGA/results_bootstrap/tmp_{cancer_name}_seed*_gpu*"
    try:
        temp_dirs = glob.glob(temp_pattern)
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        if temp_dirs:
            print(f"🧹 已清理 {len(temp_dirs)} 个临时训练目录")
    except Exception as e:
        print(f"⚠️ 批量清理临时目录失败: {e}")

    return summary, c_list

# ---------------------------
# ========== 多癌种批处理入口 ==========
# ---------------------------

def run_all_cancers(cancer_dfs: dict, B, n_jobs_per_cancer=1, n_jobs_cancers=2, out_root=OUTPUT_ROOT):
    """
    cancer_dfs: dict {cancer_name: DataFrame}
    n_jobs_per_cancer: 并行运行单癌种内部的 worker 数量（bootstrap 并行）
    n_jobs_cancers: 同时并行运行的癌种数量（for joblib nesting consider reduce）
    建议：优先并行化“癌种层面”（每个癌种独立），内部每个癌种使用较小的并行数
    """
    ensure_dir(out_root)
    all_results = {}
    # 逐癌种串行或并行（这里用简单的串行循环以避免 nested loky 问题）
    for cancer_name, df in cancer_dfs.items():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing cancer: {cancer_name} | n_samples={len(df)}")
        summary, c_arr = run_bootstrap_for_cancer(
            cancer_name,
            df,
            B=B,
            n_jobs=n_jobs_per_cancer,
            output_root=out_root
        )
        all_results[cancer_name] = summary
    # 将所有summary合并为表
    rows = []
    for c, s in all_results.items():
        cs = s['cindex_summary']
        rows.append({
            'Cancer': c,
            'B_requested': s['B_requested'],
            'c_median': cs['median'], 'c_mean': cs['mean'], 'c_std': cs['std'],
            'c_ci_lower': cs['ci_lower'], 'c_ci_upper': cs['ci_upper']
        })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(out_root, 'all_cancer_summaries.csv'), index=False)
    
    # 全局最终清理：确保所有临时目录都被删除
    import shutil
    import glob
    temp_pattern = f"{out_root}/tmp_*"
    try:
        temp_dirs = glob.glob(temp_pattern)
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        if temp_dirs:
            print(f"🧹 全局清理完成，删除了 {len(temp_dirs)} 个残留临时目录")
    except Exception as e:
        print(f"⚠️ 全局清理失败: {e}")
    
    return df_all

# ---------------------------
# ========== 使用说明（示例）=========
# ---------------------------

if __name__ == "__main__":
    # 示例：如何准备 cancer_dfs 字典（每个 DataFrame 至少包含 ['survival_months','censorship']）
    # 注意：替换成你本地的数据加载方式，确保 index 唯一
    # 格式示例:
    # cancer_df = pd.read_csv("coadread_full_table.csv")  # 包含 features + 'survival_months' + 'censorship'
    # cancer_dfs = {'COADREAD': cancer_df, ...}
    import multiprocessing as mp
    mp.set_start_method("spawn") 

    cancer_dfs = {}

    # --------- TODO: 在这里加载你的 15 个癌种数据集到 cancer_dfs ----------
    # 假设你把每个癌种CSV放在 data_root/<cancer>.csv
    data_root = "/home/zuoyiyi/SNN/TCGA/datasets_csv/preprocess_cancer_single"
    for fname in os.listdir(data_root):
        if fname.endswith(".csv"):
            cname = fname.replace("_preprocessed.csv","")
            df = pd.read_csv(os.path.join(data_root, fname), index_col=None)
            
            # 验证必需的列是否存在
            required_columns = ['case_id', 'survival_months', 'censorship', 'disc_label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"警告：{cname} 缺少必需的列: {missing_columns}")
                continue
                
            # 确保有RNA-seq特征列
            rnaseq_cols = [col for col in df.columns if '_rnaseq' in col]
            if len(rnaseq_cols) == 0:
                print(f"警告：{cname} 没有找到RNA-seq特征列（列名应包含'_rnaseq'）")
                continue
                
            cancer_dfs[cname] = df

    # 运行（示例参数）
    B = B_DEFAULT
    n_jobs_per_cancer = N_JOBS  # bootstrap 并行 worker 数
    out_root = OUTPUT_ROOT
    ensure_dir(out_root)

    # 运行所有癌种
    df_results = run_all_cancers(cancer_dfs, B=B, n_jobs_per_cancer=n_jobs_per_cancer, out_root=out_root)
    print("All done. Results saved to:", out_root)