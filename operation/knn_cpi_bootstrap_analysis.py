import argparse
import os
import pickle
import warnings
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lifelines import CoxPHFitter
from scipy.stats import pearsonr, spearmanr
from sksurv.metrics import concordance_index_censored
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

from dataset_survival import RNAseqSurvivalDataset
from model_risk import SNN_RISK

warnings.filterwarnings("ignore")


def load_cindex_from_array(results_dir: str) -> Optional[np.ndarray]:
    """从 bootstrap 训练/评估输出中加载 cindex 数组（若存在）。"""
    try:
        cindex_file = os.path.join(results_dir, "cindex_array.npy")
        return np.load(cindex_file)
    except Exception as e:
        print(f"读取cindex数组失败: {e}")
        return None


def load_model(model_path: str, input_dim: int, device: torch.device) -> torch.nn.Module:
    """加载 bootstrap 训练好的 SNN_RISK 模型。"""
    model = SNN_RISK(omic_input_dim=input_dim, model_size_omic="small", n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def compute_model_output(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """模型推理输出（与 bootstrap PFI 脚本一致：直接取 model(X) 的输出）。"""
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X_tensor).detach().cpu().numpy().reshape(-1)
    return pred


def _build_knn_neighbors_indices(
    X: np.ndarray,
    k: int = 20,
    variance_threshold: float = 0.85,
    random_state: int = 0,
) -> Tuple[np.ndarray, int]:
    """动态 PCA(解释>=variance_threshold 方差) + KNN 建树，返回邻居索引矩阵。"""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")

    pca_full = PCA(svd_solver="full", random_state=random_state)
    pca_full.fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(1, min(n_components, X.shape[1]))
    print(
        f"[KNN-CPI] PCA 自动选择维度 n_components={n_components} (>= {int(variance_threshold * 100)}% variance)"
    )

    X_reduced = pca_full.transform(X)[:, :n_components]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_reduced)
    indices = nn.kneighbors(X_reduced, return_distance=False)

    self_mask = indices[:, 0] == np.arange(indices.shape[0])
    if bool(np.all(self_mask)):
        neighbors_indices = indices[:, 1 : k + 1]
    else:
        neighbors_indices = indices[:, :k]

    if neighbors_indices.shape != (X.shape[0], k):
        raise ValueError(f"neighbors_indices shape mismatch: {neighbors_indices.shape} vs {(X.shape[0], k)}")

    return neighbors_indices, n_components


def vectorized_conditional_permute(
    X: np.ndarray,
    feature_idx: int,
    neighbors_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """向量化 KNN 条件置换：对每个样本从其近邻中随机抽一个邻居并拷贝该邻居的目标特征值。"""
    n_samples = X.shape[0]
    k = neighbors_indices.shape[1]

    neighbor_pos = rng.integers(low=0, high=k, size=n_samples)
    chosen_neighbors = neighbors_indices[np.arange(n_samples), neighbor_pos]

    X_perm = X.copy()
    X_perm[:, feature_idx] = X[chosen_neighbors, feature_idx]
    return X_perm


def default_compute_drop_fn(
    preds: np.ndarray,
    original_c_index: float,
    y_oob: Tuple[np.ndarray, np.ndarray],
) -> float:
    """drop = original - permuted, 指标为 sksurv 的 C-index。"""
    event, time = y_oob
    permuted_score = concordance_index_censored(event.astype(bool), time, preds)[0]
    return float(original_c_index - float(permuted_score))


def evaluate_knn_cpi(
    X: np.ndarray,
    model: torch.nn.Module,
    original_c_index: float,
    compute_drop_fn: Callable[[np.ndarray, float, Tuple[np.ndarray, np.ndarray]], float],
    y_oob: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    k: int = 20,
    num_repeats: int = 5,
    variance_threshold: float = 0.85,
    seed: int = 0,
) -> np.ndarray:
    """向量化 KNN-CPI：对每个特征进行 num_repeats 次条件置换，取 drop 平均值。"""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")

    rng = np.random.default_rng(seed)
    neighbors_indices, _ = _build_knn_neighbors_indices(
        X=X,
        k=k,
        variance_threshold=variance_threshold,
        random_state=seed,
    )

    n_features = X.shape[1]
    cpi_scores = np.zeros(n_features, dtype=float)

    for feature_idx in range(n_features):
        drops = []
        for _ in range(num_repeats):
            X_perm = vectorized_conditional_permute(
                X=X,
                feature_idx=feature_idx,
                neighbors_indices=neighbors_indices,
                rng=rng,
            )
            preds = compute_model_output(model, X_perm, device)
            drops.append(float(compute_drop_fn(preds, original_c_index, y_oob)))
        cpi_scores[feature_idx] = float(np.mean(drops))

    return cpi_scores


def perform_cox_analysis(
    X_data,
    data_df: pd.DataFrame,
    top_100_indices: np.ndarray,
    feature_names: list,
):
    """对 Top100 特征做单因素 Cox（批量），返回 Cox 结果列表与计数（raw 与 FDR）。"""
    cox_results = []
    protective_count = 0
    risk_count = 0

    if "survival_months" not in data_df.columns or "censorship" not in data_df.columns:
        print("警告：无法找到 survival_months/censorship，跳过 Cox 分析")
        return [], 0, 0, 0, 0

    survival_time = data_df["survival_months"].values
    censorship = data_df["censorship"].values
    event = 1 - censorship

    for idx in top_100_indices:
        feature_name = feature_names[idx]
        try:
            if isinstance(X_data, pd.DataFrame):
                feature_values = X_data.iloc[:, idx].values
            else:
                feature_values = X_data[:, idx]

            cox_data = pd.DataFrame({"T": survival_time, "E": event, "feature": feature_values}).dropna()
            if len(cox_data) < 10:
                continue

            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col="T", event_col="E")

            coef = float(cph.params_["feature"])
            hr = float(np.exp(coef))
            p_value = float(cph.summary.loc["feature", "p"])

            ci_cols = cph.confidence_intervals_.columns.tolist()
            if "95% lower-bound" in ci_cols and "95% upper-bound" in ci_cols:
                ci_lower = float(np.exp(cph.confidence_intervals_.loc["feature", "95% lower-bound"]))
                ci_upper = float(np.exp(cph.confidence_intervals_.loc["feature", "95% upper-bound"]))
            elif "95% CI lower" in ci_cols and "95% CI upper" in ci_cols:
                ci_lower = float(np.exp(cph.confidence_intervals_.loc["feature", "95% CI lower"]))
                ci_upper = float(np.exp(cph.confidence_intervals_.loc["feature", "95% CI upper"]))
            else:
                ci_lower = float(np.exp(cph.confidence_intervals_.loc["feature", ci_cols[0]]))
                ci_upper = float(np.exp(cph.confidence_intervals_.loc["feature", ci_cols[1]]))

            cox_result = {
                "gene": feature_name,
                "coef": coef,
                "hr": hr,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_samples": int(len(cox_data)),
                "n_events": int(cox_data["E"].sum()),
            }

            if p_value < 0.05:
                if coef > 0:
                    cox_result["type"] = "risk"
                    risk_count += 1
                elif coef < 0:
                    cox_result["type"] = "protective"
                    protective_count += 1
                else:
                    cox_result["type"] = "neutral"
            else:
                cox_result["type"] = "not_significant"

            cox_results.append(cox_result)

        except Exception:
            continue

    if not cox_results:
        return [], 0, 0, 0, 0

    p_values = [r["p_value"] for r in cox_results]
    reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    protective_count_fdr = 0
    risk_count_fdr = 0

    for result, adj_p, sig in zip(cox_results, p_adj, reject):
        result["p_adj"] = float(adj_p)
        result["significant_fdr"] = bool(sig)

        if bool(sig):
            if result["coef"] > 0:
                result["type_fdr"] = "risk"
                risk_count_fdr += 1
            elif result["coef"] < 0:
                result["type_fdr"] = "protective"
                protective_count_fdr += 1
            else:
                result["type_fdr"] = "neutral"
        else:
            result["type_fdr"] = "not_significant"

    return cox_results, protective_count_fdr, risk_count_fdr, protective_count, risk_count


def compute_knn_cpi_and_cox_for_seed(bootstrap_seed: int, args, num_features: int):
    """单个 bootstrap seed：OOB 上做 KNN-CPI 并保存 2000 特征排序；Top100 做 Cox 并返回统计。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== 分析 Bootstrap Seed {bootstrap_seed} ===")

    model_path = os.path.join(args.cancer_results_dir, "models", f"bootstrap_model_seed{bootstrap_seed}.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    data_csv = os.path.join(args.csv_path, f"{args.cancer_upper}_preprocessed.csv")
    if not os.path.exists(data_csv):
        print(f"数据文件不存在: {data_csv}")
        return None

    data_df = pd.read_csv(data_csv)
    dataset = RNAseqSurvivalDataset(data_df, label_col="survival_months", seed=args.seed)
    X_full = dataset.features

    if isinstance(X_full, pd.DataFrame):
        feature_names = X_full.columns.tolist()
        X_full_np = X_full.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_full.shape[1])]
        X_full_np = X_full.numpy()

    train_idx_file = os.path.join(args.cancer_results_dir, "models", f"train_idx_seed{bootstrap_seed}.npy")
    oob_idx_file = os.path.join(args.cancer_results_dir, "models", f"oob_idx_seed{bootstrap_seed}.npy")

    if not os.path.exists(train_idx_file) or not os.path.exists(oob_idx_file):
        print(f"索引文件不存在: {train_idx_file} or {oob_idx_file}")
        return None

    oob_indices = np.load(oob_idx_file)

    model = load_model(model_path, input_dim=X_full_np.shape[1], device=device)

    X_oob = X_full_np[oob_indices]

    survival_time = data_df["survival_months"].values[oob_indices]
    censorship = data_df["censorship"].values[oob_indices]
    event = 1 - censorship
    y_oob = (event, survival_time)

    preds_baseline = compute_model_output(model, X_oob, device)
    baseline_cindex = concordance_index_censored(event.astype(bool), survival_time, preds_baseline)[0]

    print("计算 KNN-CPI 值...")
    importance_scores = evaluate_knn_cpi(
        X=X_oob,
        model=model,
        original_c_index=float(baseline_cindex),
        compute_drop_fn=default_compute_drop_fn,
        y_oob=y_oob,
        device=device,
        k=args.knn_k,
        num_repeats=args.num_repeats,
        variance_threshold=0.85,
        seed=int(args.seed + bootstrap_seed),
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_score": importance_scores,
        }
    )
    feature_importance_df = feature_importance_df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    feature_importance_df["rank"] = np.arange(1, len(feature_importance_df) + 1)

    importance_dir = os.path.join(args.knn_cpi_dir, args.cancer, "knn_cpi_feature_importance")
    os.makedirs(importance_dir, exist_ok=True)
    importance_file = os.path.join(importance_dir, f"seed{bootstrap_seed}_knn_cpi_ranking.csv")
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"✅ KNN-CPI特征重要性排序已保存: {importance_file}")

    top_100_indices = np.argsort(importance_scores)[-100:][::-1]

    print("开始 Cox 分析 (Top100)...")
    cox_results, cox_protective_count_fdr, cox_risk_count_fdr, cox_protective_count_raw, cox_risk_count_raw = (
        perform_cox_analysis(X_full, data_df, top_100_indices, feature_names)
    )

    cox_prognostic_count_fdr = int(cox_protective_count_fdr + cox_risk_count_fdr)
    cox_prognostic_count_raw = int(cox_protective_count_raw + cox_risk_count_raw)

    cindex_array = load_cindex_from_array(args.cancer_results_dir)
    if cindex_array is not None and bootstrap_seed <= len(cindex_array):
        cindex = float(cindex_array[bootstrap_seed - 1])
    else:
        cindex = float(baseline_cindex)

    return {
        "bootstrap_seed": int(bootstrap_seed),
        "cindex": float(cindex),
        "baseline_cindex_oob": float(baseline_cindex),
        "cox_prognostic_factors_fdr": int(cox_prognostic_count_fdr),
        "cox_protective_factors_fdr": int(cox_protective_count_fdr),
        "cox_risk_factors_fdr": int(cox_risk_count_fdr),
        "cox_prognostic_factors_raw": int(cox_prognostic_count_raw),
        "cox_protective_factors_raw": int(cox_protective_count_raw),
        "cox_risk_factors_raw": int(cox_risk_count_raw),
        "cox_results": cox_results,
        "feature_names": feature_names,
        "importance_scores": importance_scores,
        "analysis_method": "Bootstrap_KNN_CPI_plus_Cox_Analysis",
    }


def create_bootstrap_correlation_plots(cancer: str, df: pd.DataFrame, results_dir: str) -> None:
    """相关性散点图（仅 Cox 统计口径，raw p）。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def plot_with_fit(ax, x, y, xlabel, ylabel, title, color):
        ax.scatter(x, y, alpha=0.7, color=color, s=50)
        r, p = pearsonr(x, y)

        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(x_line, p_fit(x_line), "r--", alpha=0.8, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")

        if p < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p:.3f}"

        if p < 0.05:
            sig_text = "*"
            if p < 0.01:
                sig_text = "**"
            if p < 0.001:
                sig_text = "***"
        else:
            sig_text = "ns"

        stats_text = f"r = {r:.3f}\n{p_text}\n{sig_text}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
            fontsize=10,
            fontweight="bold",
        )

    plot_with_fit(
        axes[0],
        df["cindex"],
        df["cox_prognostic_factors_raw"],
        "C-index",
        "Cox Prognostic Factors (Raw p)",
        f"{cancer}: C-index vs Cox Prognostic Factors (Bootstrap)",
        "green",
    )

    plot_with_fit(
        axes[1],
        df["cindex"],
        df["cox_risk_factors_raw"],
        "C-index",
        "Cox Risk Factors (Raw p)",
        f"{cancer}: C-index vs Cox Risk Factors (Bootstrap)",
        "orange",
    )

    plot_with_fit(
        axes[2],
        df["cindex"],
        df["cox_protective_factors_raw"],
        "C-index",
        "Cox Protective Factors (Raw p)",
        f"{cancer}: C-index vs Cox Protective Factors (Bootstrap)",
        "cyan",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{cancer}_bootstrap_correlation_plots.png"), dpi=300, bbox_inches="tight")
    plt.close()


def analyze_bootstrap_cancer(args, num_features: int, num_bootstraps: int = 300):
    all_results = []

    for bootstrap_seed in range(1, num_bootstraps + 1):
        result = compute_knn_cpi_and_cox_for_seed(bootstrap_seed, args, num_features)
        if result is not None:
            all_results.append(result)
        else:
            print(f"⚠️ Bootstrap seed {bootstrap_seed} 分析失败，跳过")

    if not all_results:
        print(f"癌症类型 {args.cancer} 没有成功分析的bootstrap模型")
        return None

    cindices = [r["cindex"] for r in all_results]

    cox_prognostic_counts_fdr = [r["cox_prognostic_factors_fdr"] for r in all_results]
    cox_risk_counts_fdr = [r["cox_risk_factors_fdr"] for r in all_results]
    cox_protective_counts_fdr = [r["cox_protective_factors_fdr"] for r in all_results]

    cox_prognostic_counts_raw = [r["cox_prognostic_factors_raw"] for r in all_results]
    cox_risk_counts_raw = [r["cox_risk_factors_raw"] for r in all_results]
    cox_protective_counts_raw = [r["cox_protective_factors_raw"] for r in all_results]

    stats_summary = {
        "cancer": args.cancer,
        "n_bootstrap_models": int(len(all_results)),
        "cindex_mean": float(np.mean(cindices)),
        "cindex_std": float(np.std(cindices)),
        "cindex_min": float(np.min(cindices)),
        "cindex_max": float(np.max(cindices)),
        "cox_prognostic_factors_fdr_mean": float(np.mean(cox_prognostic_counts_fdr)),
        "cox_prognostic_factors_fdr_std": float(np.std(cox_prognostic_counts_fdr)),
        "cox_risk_factors_fdr_mean": float(np.mean(cox_risk_counts_fdr)),
        "cox_risk_factors_fdr_std": float(np.std(cox_risk_counts_fdr)),
        "cox_protective_factors_fdr_mean": float(np.mean(cox_protective_counts_fdr)),
        "cox_protective_factors_fdr_std": float(np.std(cox_protective_counts_fdr)),
        "cox_prognostic_factors_raw_mean": float(np.mean(cox_prognostic_counts_raw)),
        "cox_prognostic_factors_raw_std": float(np.std(cox_prognostic_counts_raw)),
        "cox_risk_factors_raw_mean": float(np.mean(cox_risk_counts_raw)),
        "cox_risk_factors_raw_std": float(np.std(cox_risk_counts_raw)),
        "cox_protective_factors_raw_mean": float(np.mean(cox_protective_counts_raw)),
        "cox_protective_factors_raw_std": float(np.std(cox_protective_counts_raw)),
    }

    if len(cindices) > 2:
        corr_cox_prog_raw, p_cox_prog_raw = pearsonr(cindices, cox_prognostic_counts_raw)
        spearman_cox_prog_raw, sp_cox_prog_raw = spearmanr(cindices, cox_prognostic_counts_raw)
        corr_cox_prog_fdr, p_cox_prog_fdr = pearsonr(cindices, cox_prognostic_counts_fdr)

        stats_summary.update(
            {
                "corr_cindex_cox_prognostic_raw_pearson": float(corr_cox_prog_raw),
                "corr_cindex_cox_prognostic_raw_pearson_p": float(p_cox_prog_raw),
                "corr_cindex_cox_prognostic_raw_spearman": float(spearman_cox_prog_raw),
                "corr_cindex_cox_prognostic_raw_spearman_p": float(sp_cox_prog_raw),
                "corr_cindex_cox_prognostic_fdr_pearson": float(corr_cox_prog_fdr),
                "corr_cindex_cox_prognostic_fdr_pearson_p": float(p_cox_prog_fdr),
            }
        )

    results_dir = os.path.join(args.knn_cpi_dir, args.cancer)
    os.makedirs(results_dir, exist_ok=True)

    detailed_results_df = pd.DataFrame(
        [
            {
                "bootstrap_seed": r["bootstrap_seed"],
                "cindex": r["cindex"],
                "baseline_cindex_oob": r["baseline_cindex_oob"],
                "cox_prognostic_factors_fdr": r["cox_prognostic_factors_fdr"],
                "cox_risk_factors_fdr": r["cox_risk_factors_fdr"],
                "cox_protective_factors_fdr": r["cox_protective_factors_fdr"],
                "cox_prognostic_factors_raw": r["cox_prognostic_factors_raw"],
                "cox_risk_factors_raw": r["cox_risk_factors_raw"],
                "cox_protective_factors_raw": r["cox_protective_factors_raw"],
            }
            for r in all_results
        ]
    )

    detailed_results_df.to_csv(os.path.join(results_dir, f"{args.cancer}_bootstrap_detailed_results.csv"), index=False)

    with open(os.path.join(results_dir, f"{args.cancer}_bootstrap_complete_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    detailed_results_df[[
        "bootstrap_seed",
        "cindex",
        "baseline_cindex_oob",
        "cox_prognostic_factors_fdr",
        "cox_risk_factors_fdr",
        "cox_protective_factors_fdr",
        "cox_prognostic_factors_raw",
        "cox_risk_factors_raw",
        "cox_protective_factors_raw",
    ]].to_csv(os.path.join(results_dir, f"{args.cancer}_bootstrap_cox_analysis_summary.csv"), index=False)

    all_cox_details = []
    for r in all_results:
        for cox_result in r.get("cox_results", []) or []:
            all_cox_details.append(
                {
                    "bootstrap_seed": r["bootstrap_seed"],
                    "cindex": r["cindex"],
                    **cox_result,
                }
            )

    if len(all_cox_details) > 0:
        pd.DataFrame(all_cox_details).to_csv(
            os.path.join(results_dir, f"{args.cancer}_bootstrap_cox_detailed_results.csv"), index=False
        )

    if len(all_results) > 2:
        create_bootstrap_correlation_plots(args.cancer, detailed_results_df, results_dir)

    pd.DataFrame([stats_summary]).to_csv(
        os.path.join(results_dir, f"{args.cancer}_bootstrap_summary.csv"), index=False
    )

    print(f"✅ 结果已保存到: {results_dir}")
    return stats_summary


def main(args):
    cancer_list = sorted(os.listdir(args.results_dir))
    os.makedirs(args.knn_cpi_dir, exist_ok=True)

    all_cancer_summaries = []

    num_bootstraps = int(getattr(args, "num_bootstrap", 100))

    for cancer in cancer_list:
        if not os.path.isdir(os.path.join(args.results_dir, cancer)):
            continue

        args.cancer = cancer
        args.cancer_upper = cancer.upper()
        args.cancer_results_dir = os.path.join(args.results_dir, cancer)

        num_features = 2000
        print(f"\n{'=' * 60}")
        print(f"正在处理癌症类型: {cancer} (Bootstrap KNN-CPI)")
        print(f"{'=' * 60}")

        summary = analyze_bootstrap_cancer(args, num_features=num_features, num_bootstraps=num_bootstraps)
        if summary is not None:
            all_cancer_summaries.append(summary)

    if len(all_cancer_summaries) > 0:
        all_summaries_df = pd.DataFrame(all_cancer_summaries)
        summary_file = os.path.join(args.knn_cpi_dir, "all_cancers_bootstrap_summary.csv")
        all_summaries_df.to_csv(summary_file, index=False)
        print(f"\n✅ 所有癌症汇总结果已保存到: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对Bootstrap模型进行KNN-CPI可解释性分析 + Top100单因素Cox分析")
    parser.add_argument("--csv_path", type=str, required=True, help="Bootstrap数据路径 (preprocess_cancer_single)")
    parser.add_argument("--results_dir", type=str, required=True, help="Bootstrap结果目录 (results_bootstrap)")
    parser.add_argument("--knn_cpi_dir", type=str, required=True, help="KNN-CPI分析结果保存目录")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--num_bootstrap", type=int, default=100, help="Bootstrap迭代次数 (默认100)")
    parser.add_argument("--knn_k", type=int, default=20, help="KNN-CPI: KNN邻居数 (默认20)")
    parser.add_argument("--num_repeats", type=int, default=5, help="KNN-CPI: 每特征条件置换次数 (默认5)")

    args = parser.parse_args()
    main(args)
