import argparse
import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pickle

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

from model_genomic import SNN
from dataset_survival import RNAseqSurvivalDataset

import warnings

warnings.filterwarnings("ignore")


def load_cindex_from_results(results_file: str) -> Optional[float]:
    """从结果文件中加载 cindex"""
    try:
        with open(results_file, "rb") as f:
            results = pickle.load(f)
            return float(results["test_cindex"])
    except Exception as e:
        print(f"读取结果文件失败: {e}")
        return None


def load_model(model_path: str, input_dim: int, device: torch.device) -> SNN:
    """加载已训练好的 SNN 模型"""
    model = SNN(omic_input_dim=input_dim, model_size_omic="small", n_classes=4)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def compute_risk_score(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """预测风险分数（与训练/评估一致：risk = -sum(survival)）"""
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_omic=X_tensor)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk


def _build_knn_neighbors_indices(
    X: np.ndarray,
    k: int = 20,
    variance_threshold: float = 0.85,
    random_state: int = 0,
) -> Tuple[np.ndarray, int]:
    """动态 PCA(解释85%方差) + KNN 建树，返回邻居索引矩阵。

    参数:
    - X: 标准化后的基因表达矩阵, shape [N, F]
    - k: 每个样本使用的邻居数（不含自身）

    返回:
    - neighbors_indices: shape [N, k]
    - n_components: 自动选择的 PCA 维度
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")

    # 1) 动态 PCA：选择累计解释方差达到 85% 的维度
    pca_full = PCA(svd_solver="full", random_state=random_state)
    pca_full.fit(X)

    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(1, min(n_components, X.shape[1]))
    print(f"[KNN-CPI] PCA 自动选择维度 n_components={n_components} (>= {int(variance_threshold * 100)}% variance)")

    X_reduced = pca_full.transform(X)[:, :n_components]

    # 2) 在降维后的空间建 KNN 树
    # 为避免近邻包含自身，查询 k+1 并在后处理时去掉自身（若存在）
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_reduced)
    indices = nn.kneighbors(X_reduced, return_distance=False)

    # 去掉自身（若第一列就是自身）
    self_mask = indices[:, 0] == np.arange(indices.shape[0])
    if bool(np.all(self_mask)):
        neighbors_indices = indices[:, 1 : k + 1]
    else:
        # 兜底：如果有些行不满足，仍然取前 k 个作为邻居
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
    """向量化的 KNN 条件置换。

    对每个样本 i：
    - 从其 K 个近邻里随机抽取 1 个邻居 j
    - 将 X[j, feature_idx] 赋值给 X_perm[i, feature_idx]

    约束：严禁在样本维度使用 for-loop。

    参数:
    - X: shape [N, F]
    - feature_idx: 目标基因/特征索引
    - neighbors_indices: shape [N, K]
    - rng: numpy 随机数生成器
    """
    n_samples = X.shape[0]
    k = neighbors_indices.shape[1]

    # 为每个样本随机选择一个近邻位置 [0, K)
    neighbor_pos = rng.integers(low=0, high=k, size=n_samples)

    # 高级索引：为每个样本选出一个具体邻居的样本索引
    chosen_neighbors = neighbors_indices[np.arange(n_samples), neighbor_pos]

    X_perm = X.copy()
    X_perm[:, feature_idx] = X[chosen_neighbors, feature_idx]
    return X_perm


def evaluate_knn_cpi(
    X: np.ndarray,
    model: torch.nn.Module,
    original_c_index: float,
    compute_drop_fn: Callable[[np.ndarray, float, np.ndarray], float],
    y_val: np.ndarray,
    device: torch.device,
    k: int = 20,
    num_repeats: int = 5,
    variance_threshold: float = 0.85,
    seed: int = 0,
) -> np.ndarray:
    """向量化加速的 KNN-CPI 评估。

    重要说明：
    - X 必须为标准化后的表达矩阵（numpy array），shape [N_samples, 2000]
    - 该函数对每个 feature 进行 num_repeats 次条件置换，并计算性能下降

    参数:
    - X: 标准化后的表达矩阵, shape [N, F]
    - model: PyTorch 模型（SNN）
    - original_c_index: 原始数据上的性能基线（例如 baseline C-index）
    - compute_drop_fn: 回调函数，用于从预测结果计算性能下降
        - 签名建议：compute_drop_fn(preds, original_c_index, y_val) -> drop
    - y_val: 生存信息数组/结构（由 compute_drop_fn 自行解释）
    - device: torch device
    - k: KNN 近邻数（默认 20）
    - num_repeats: 每个特征条件置换次数（默认 5）

    返回:
    - cpi_scores: shape [F,] 的 numpy array
    """
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

            preds = compute_risk_score(model, X_perm, device)
            drop = float(compute_drop_fn(preds, original_c_index, y_val))
            drops.append(drop)

        cpi_scores[feature_idx] = float(np.mean(drops))

    return cpi_scores


def default_compute_drop_fn(risk_scores: np.ndarray, original_c_index: float, y_val: pd.DataFrame) -> float:
    """默认的性能下降计算：drop = original_c_index - cindex_perm"""
    cindex_perm = concordance_index(y_val["survival_months"], -risk_scores, y_val["censorship"])
    return float(original_c_index - cindex_perm)


def perform_cox_analysis(
    X_val: pd.DataFrame,
    val_df: pd.DataFrame,
    top_100_indices: np.ndarray,
    feature_names: list,
):
    """对选定特征进行批量单因素 Cox 分析（与其他脚本一致）。"""
    cox_results = []
    protective_count = 0
    risk_count = 0

    survival_time = val_df["survival_months"].values
    censorship = val_df["censorship"].values
    event = 1 - censorship

    skipped_features = []

    for idx in top_100_indices:
        feature_name = feature_names[idx]

        try:
            if isinstance(X_val, pd.DataFrame):
                feature_values = X_val.iloc[:, idx].values
            else:
                feature_values = X_val[:, idx]

            cox_data = pd.DataFrame({
                "T": survival_time,
                "E": event,
                "feature": feature_values,
            })

            original_samples = len(cox_data)
            cox_data = cox_data.dropna()
            samples_after_dropna = len(cox_data)

            if len(cox_data) < 10:
                skipped_features.append({
                    "feature": feature_name,
                    "reason": f"样本不足: {samples_after_dropna}/10 (原始:{original_samples})",
                })
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

        except Exception as e:
            skipped_features.append({
                "feature": feature_name,
                "reason": f"Cox分析失败: {str(e)}",
            })
            continue

    if cox_results:
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

    return [], 0, 0, 0, 0


def compute_knn_cpi_and_prognostic_factors(repeat: int, fold: int, args, num_features: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== 分析 Repeat {repeat} Fold {fold} ===")

    model_path = os.path.join(args.cancer_results_dir, f"repeat{repeat}_s_{fold}_final_test_model.pt")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None

    results_file = os.path.join(args.cancer_results_dir, f"repeat{repeat}_fold{fold}_results.pkl")
    cindex = load_cindex_from_results(results_file)
    if cindex is None:
        print(f"无法获取cindex，跳过 repeat{repeat}_fold{fold}")
        return None

    val_csv = os.path.join(args.csv_path, f"{args.cancer_lower}_{fold}_val.csv")
    if not os.path.exists(val_csv):
        print(f"数据文件不存在: {val_csv}")
        return None

    val_df = pd.read_csv(val_csv)
    val_dataset = RNAseqSurvivalDataset(val_df, label_col="survival_months", seed=args.seed)
    X_val = val_dataset.features

    if isinstance(X_val, pd.DataFrame):
        feature_names = X_val.columns.tolist()
        X_val_np = X_val.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
        X_val_np = X_val.numpy()

    train_csv = os.path.join(args.csv_path, f"{args.cancer_lower}_{fold}_train.csv")
    if not os.path.exists(train_csv):
        print(f"训练数据文件不存在: {train_csv}")
        return None

    train_df = pd.read_csv(train_csv)

    print("🔧 合并训练集和验证集用于Cox分析，提高样本量和统计功效")
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_dataset = RNAseqSurvivalDataset(combined_df, label_col="survival_months", seed=args.seed)
    X_combined = combined_dataset.features

    model = load_model(model_path, input_dim=X_val_np.shape[1], device=device)

    y_val = pd.DataFrame({
        "survival_months": val_dataset.times,
        "censorship": val_dataset.censorship,
    }, index=val_dataset.case_ids)

    risk_scores = compute_risk_score(model, X_val_np, device)
    baseline_cindex = concordance_index(y_val["survival_months"], -risk_scores, y_val["censorship"])

    print("计算 KNN-CPI 值...")
    importance = evaluate_knn_cpi(
        X=X_val_np,
        model=model,
        original_c_index=float(baseline_cindex),
        compute_drop_fn=default_compute_drop_fn,
        y_val=y_val,
        device=device,
        k=args.knn_k,
        num_repeats=args.num_repeats,
        variance_threshold=0.85,
        seed=args.seed,
    )

    print("保存完整特征重要性排序...")
    feature_importance_df = pd.DataFrame({
        "feature_name": feature_names,
        "importance_score": importance,
        "rank": range(1, len(feature_names) + 1),
    })
    feature_importance_df = feature_importance_df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    feature_importance_df["rank"] = range(1, len(feature_importance_df) + 1)

    importance_dir = os.path.join(args.knn_cpi_dir, args.cancer, "knn_cpi_feature_importance")
    os.makedirs(importance_dir, exist_ok=True)
    importance_file = os.path.join(importance_dir, f"repeat{repeat}_fold{fold}_knn_cpi_feature_importance_ranking.csv")
    feature_importance_df.to_csv(importance_file, index=False)

    # Top 100 特征索引（用于下游 Cox 分析）
    top_100_indices = np.argsort(importance)[-100:][::-1]

    print("开始Cox分析...")
    cox_results, cox_protective_count_fdr, cox_risk_count_fdr, cox_protective_count_raw, cox_risk_count_raw = perform_cox_analysis(
        X_combined, combined_df, top_100_indices, feature_names
    )

    cox_prognostic_count_fdr = int(cox_protective_count_fdr + cox_risk_count_fdr)
    cox_prognostic_count_raw = int(cox_protective_count_raw + cox_risk_count_raw)

    return {
        "repeat": repeat,
        "fold": fold,
        "cindex": float(cindex),
        "cox_prognostic_factors_fdr": int(cox_prognostic_count_fdr),
        "cox_protective_factors_fdr": int(cox_protective_count_fdr),
        "cox_risk_factors_fdr": int(cox_risk_count_fdr),
        "cox_prognostic_factors_raw": int(cox_prognostic_count_raw),
        "cox_protective_factors_raw": int(cox_protective_count_raw),
        "cox_risk_factors_raw": int(cox_risk_count_raw),
        "cox_results": cox_results,
        "feature_names": feature_names,
        "importance_scores": importance,
        "analysis_method": "KNN_CPI_plus_Cox_Analysis",
    }


def analyze_cancer_type(args, num_features: int, num_repeats: int = 10, num_folds: int = 5):
    all_results = []

    for repeat in range(num_repeats):
        for fold in range(num_folds):
            try:
                result = compute_knn_cpi_and_prognostic_factors(repeat, fold, args, num_features)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ {args.cancer} Repeat {repeat} Fold {fold} 分析失败: {str(e)}")
                continue

    if not all_results:
        print(f"癌症类型 {args.cancer} 没有成功分析的实验")
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
        "n_experiments": int(len(all_results)),
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

    results_dir = os.path.join(args.knn_cpi_dir, args.cancer)
    os.makedirs(results_dir, exist_ok=True)

    detailed_results_df = pd.DataFrame([
        {
            "repeat": r["repeat"],
            "fold": r["fold"],
            "cindex": r["cindex"],
            "cox_prognostic_factors_fdr": r["cox_prognostic_factors_fdr"],
            "cox_risk_factors_fdr": r["cox_risk_factors_fdr"],
            "cox_protective_factors_fdr": r["cox_protective_factors_fdr"],
            "cox_prognostic_factors_raw": r["cox_prognostic_factors_raw"],
            "cox_risk_factors_raw": r["cox_risk_factors_raw"],
            "cox_protective_factors_raw": r["cox_protective_factors_raw"],
        }
        for r in all_results
    ])

    detailed_results_df.to_csv(os.path.join(results_dir, f"{args.cancer}_detailed_results.csv"), index=False)

    with open(os.path.join(results_dir, f"{args.cancer}_complete_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    cox_df = detailed_results_df[[
        "repeat",
        "fold",
        "cindex",
        "cox_prognostic_factors_fdr",
        "cox_risk_factors_fdr",
        "cox_protective_factors_fdr",
        "cox_prognostic_factors_raw",
        "cox_risk_factors_raw",
        "cox_protective_factors_raw",
    ]].copy()
    cox_df.to_csv(os.path.join(results_dir, f"{args.cancer}_cox_analysis_summary.csv"), index=False)

    all_cox_details = []
    for r in all_results:
        if "cox_results" in r and r["cox_results"]:
            for cox_result in r["cox_results"]:
                cox_detail = {
                    "repeat": r["repeat"],
                    "fold": r["fold"],
                    "cindex": r["cindex"],
                    **cox_result,
                }
                all_cox_details.append(cox_detail)

    if all_cox_details:
        cox_details_df = pd.DataFrame(all_cox_details)
        cox_details_df.to_csv(os.path.join(results_dir, f"{args.cancer}_cox_detailed_results.csv"), index=False)

    return stats_summary


def main(args):
    print("🔥 综合分析框架: KNN-CPI + Cox回归")

    cancer_list = sorted(os.listdir(args.results_dir))
    os.makedirs(args.knn_cpi_dir, exist_ok=True)

    all_cancer_summaries = []

    for cancer in cancer_list:
        if not os.path.isdir(os.path.join(args.results_dir, cancer)):
            continue

        args.cancer = cancer
        args.cancer_lower = cancer.lower()
        args.cancer_results_dir = os.path.join(args.results_dir, cancer)

        num_features = 2000

        summary = analyze_cancer_type(args, num_features=num_features)
        if summary is not None:
            all_cancer_summaries.append(summary)

    if all_cancer_summaries:
        summary_df = pd.DataFrame(all_cancer_summaries)
        summary_df.to_csv(os.path.join(args.knn_cpi_dir, "all_cancers_summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单独分析每个实验的KNN-CPI重要性并进行Cox回归分析")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV数据路径")
    parser.add_argument("--results_dir", type=str, required=True, help="Nested CV结果目录")
    parser.add_argument("--knn_cpi_dir", type=str, required=True, help="KNN-CPI分析结果保存目录")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--knn_k", type=int, default=20, help="KNN邻居数")
    parser.add_argument("--num_repeats", type=int, default=5, help="每个特征的条件置换重复次数")

    args = parser.parse_args()
    main(args)
