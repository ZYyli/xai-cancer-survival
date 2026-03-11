import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from dataset_survival import RNAseqSurvivalDataset
from model_genomic import SNN


ArrayLike = Union[np.ndarray, torch.Tensor]
YValType = Union[pd.DataFrame, Mapping[str, Any]]
ComputeCIndexFn = Callable[[torch.nn.Module, torch.Tensor, YValType], float]


def default_compute_cindex_fn_snn(model: torch.nn.Module, X_tensor: torch.Tensor, y_val: YValType) -> float:
    """默认 C-index 计算函数（与 operation/core_utils.py 口径一致）。

    适用模型：`operation/model_genomic.py` 中的 `SNN`（forward 输出 logits）。

    风险分数定义（与训练/评估一致）：
    - logits = model(x_omic=X)
    - hazards = sigmoid(logits)
    - survival = cumprod(1 - hazards, dim=1)
    - risk = -sum(survival, dim=1)

    C-index 计算（与 core_utils.py 一致使用 sksurv）：
    - event_bool = (1 - censorship).astype(bool)
    - time = survival_months
    - estimate = risk（值越大风险越高）

    y_val 支持：
    - pd.DataFrame：必须包含 `survival_months` 与 `censorship`
    - dict-like：包含同名 key
    """
    if isinstance(y_val, pd.DataFrame):
        if "survival_months" not in y_val.columns or "censorship" not in y_val.columns:
            raise ValueError("y_val DataFrame must contain columns: survival_months, censorship")
        survival_months = y_val["survival_months"].to_numpy(dtype=float)
        censorship = y_val["censorship"].to_numpy(dtype=float)
    else:
        if "survival_months" not in y_val or "censorship" not in y_val:
            raise ValueError("y_val dict-like must contain keys: survival_months, censorship")
        survival_months = np.asarray(y_val["survival_months"], dtype=float)
        censorship = np.asarray(y_val["censorship"], dtype=float)

    if survival_months.shape[0] != X_tensor.shape[0] or censorship.shape[0] != X_tensor.shape[0]:
        raise ValueError(
            f"y_val length mismatch with X_tensor: len(time)={survival_months.shape[0]}, "
            f"len(censor)={censorship.shape[0]}, n_samples={X_tensor.shape[0]}"
        )

    model.eval()
    with torch.no_grad():
        logits = model(x_omic=X_tensor)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = (-torch.sum(survival, dim=1)).detach().cpu().numpy().reshape(-1)

    event_bool = (1 - censorship).astype(bool)
    cindex = concordance_index_censored(event_bool, survival_months, risk)[0]
    return float(cindex)


def _to_numpy_2d(X: ArrayLike) -> np.ndarray:
    """将输入 X 转为 numpy 2D 矩阵。

    说明：
    - 仅做数据格式转换，不改变数值。
    - 不做标准化/归一化假设。
    """
    if isinstance(X, np.ndarray):
        X_np = X
    elif torch.is_tensor(X):
        X_np = X.detach().cpu().numpy()
    else:
        raise TypeError(f"X_val must be np.ndarray or torch.Tensor, got {type(X)}")

    if X_np.ndim != 2:
        raise ValueError(f"X_val must be 2D array with shape [N, F], got shape={X_np.shape}")

    return X_np


def _validate_rankings(rankings: Dict[str, np.ndarray], n_features: int) -> None:
    """检查每个方法的 ranking 是否为长度 n_features 的一维索引数组。"""
    for method, r in rankings.items():
        if not isinstance(r, np.ndarray):
            raise TypeError(f"xai_rankings['{method}'] must be np.ndarray, got {type(r)}")
        if r.ndim != 1:
            raise ValueError(f"xai_rankings['{method}'] must be 1D array, got shape={r.shape}")
        if len(r) != n_features:
            raise ValueError(
                f"xai_rankings['{method}'] length mismatch: got {len(r)}, expected {n_features}"
            )
        if np.min(r) < 0 or np.max(r) >= n_features:
            raise ValueError(
                f"xai_rankings['{method}'] has invalid indices: min={np.min(r)}, max={np.max(r)}, n_features={n_features}"
            )


def _normalize_k_steps(k_steps: Sequence[int], n_features: int) -> List[int]:
    """规范化 k_steps：去重、排序、裁剪到 [0, n_features]。"""
    uniq = sorted(set(int(k) for k in k_steps))
    out = []
    for k in uniq:
        if k < 0:
            continue
        if k > n_features:
            k = n_features
        out.append(k)
    if len(out) == 0:
        raise ValueError("k_steps becomes empty after normalization")
    if out[0] != 0:
        out = [0] + out
    return out


def _mask_topk_with_mean(X: np.ndarray, topk_indices: np.ndarray, feature_means: np.ndarray) -> np.ndarray:
    """将 topk 特征列替换为全局均值（向量化操作，无逐样本循环）。

    注意：
    - 这里的“删除/屏蔽”并不是把特征置 0，而是把该列替换为全局均值。
    - 这会保留该特征的总体分布中心，但抹除个体差异信息。
    """
    if topk_indices.size == 0:
        return X

    X_masked = X.copy()
    X_masked[:, topk_indices] = feature_means[topk_indices]
    return X_masked


def evaluate_deletion_curve(
    model: torch.nn.Module,
    X_val: ArrayLike,
    y_val: YValType,
    xai_rankings: Dict[str, np.ndarray],
    compute_cindex_fn: ComputeCIndexFn,
    k_steps: Sequence[int],
    device: Optional[torch.device] = None,
    random_seed: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算 Deletion Curve 与 AOPC（Area Over the Perturbation Curve）。

    核心思想：
    - 对每个方法给出的特征重要性排名，从 Top-K 开始逐步“屏蔽”特征。
    - 屏蔽方式：把被选中的特征整列替换为其在 X_val 中的全局均值（不是置 0）。

    参数:
    - model: 训练好的 PyTorch 模型（外部保证 eval 模式更好，但函数内部也会调用 model.eval()）
    - X_val: 验证集特征矩阵，shape [N, 2000]
    - y_val: 生存标签（DataFrame 或 dict，由 compute_cindex_fn 自行解释）
    - xai_rankings: {method_name: indices_sorted_desc}, 其中 indices_sorted_desc 长度=F
    - compute_cindex_fn: 回调函数，签名 (model, X_tensor, y_val) -> cindex
    - k_steps: 逐步删除的 K 列表，例如 [0,10,20,50,...,2000]
    - device: torch.device；若为 None，则优先使用 model 参数所在 device，否则 cpu
    - random_seed: Random 对照组的随机种子

    返回:
    - results_df: 每一行对应一个 (method, k)
      列：method, k, cindex
    - aopc_df: 每一行对应一个 method
      列：method, baseline_cindex, aopc
    """
    # 将输入统一为 numpy，便于后续做“按列均值替换”的向量化屏蔽
    X_np = _to_numpy_2d(X_val)
    n_samples, n_features = X_np.shape

    rankings: Dict[str, np.ndarray] = dict(xai_rankings)

    _validate_rankings(rankings, n_features=n_features)

    # 自动加入 Random 对照组（不覆盖用户同名 key）
    if "Random" not in rankings:
        rng = np.random.default_rng(int(random_seed))
        rankings["Random"] = rng.permutation(n_features).astype(int)

    k_steps_norm = _normalize_k_steps(k_steps, n_features=n_features)

    # 预计算每个特征在验证集的“全局均值”，用于 mean masking
    feature_means = X_np.mean(axis=0)

    # 确定 device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    # 先算 baseline (k=0) —— 对所有方法共享同一个 baseline
    X0_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline_cindex = float(compute_cindex_fn(model, X0_tensor, y_val))

    rows = []

    for method, ranking in rankings.items():
        for k in k_steps_norm:
            if k == 0:
                cindex_k = baseline_cindex
            else:
                topk = ranking[:k]
                # 核心屏蔽逻辑：把 top-k 特征列替换为其全局均值（向量化，无逐样本 for 循环）
                X_masked = _mask_topk_with_mean(X_np, topk_indices=topk, feature_means=feature_means)
                X_tensor = torch.tensor(X_masked, dtype=torch.float32, device=device)
                with torch.no_grad():
                    cindex_k = float(compute_cindex_fn(model, X_tensor, y_val))

            rows.append({"method": str(method), "k": int(k), "cindex": float(cindex_k)})

    results_df = pd.DataFrame(rows)

    # AOPC: mean over k_steps (包含 k=0 时 drop=0，不影响)
    aopc_rows = []
    for method in sorted(results_df["method"].unique()):
        df_m = results_df[results_df["method"] == method].sort_values("k")
        drops = baseline_cindex - df_m["cindex"].values.astype(float)
        aopc = float(np.mean(drops))
        aopc_rows.append({"method": method, "baseline_cindex": baseline_cindex, "aopc": aopc})

    aopc_df = pd.DataFrame(aopc_rows)
    return results_df, aopc_df


def plot_deletion_curve(
    results_df: pd.DataFrame,
    aopc_df: pd.DataFrame,
    output_path: str,
    title: str = "Deletion Curve (Mean Masking)",
) -> None:
    """绘制 Deletion Curve，并在图例中标注 AOPC。"""
    required_cols = {"method", "k", "cindex"}
    if not required_cols.issubset(set(results_df.columns)):
        raise ValueError(f"results_df missing required columns: {required_cols - set(results_df.columns)}")

    if "method" not in aopc_df.columns or "aopc" not in aopc_df.columns:
        raise ValueError("aopc_df must contain columns: method, aopc")

    aopc_map = aopc_df.set_index("method")["aopc"].to_dict()

    plt.figure(figsize=(10, 6))

    methods = sorted(results_df["method"].unique())
    cmap = plt.get_cmap("tab10")

    for i, method in enumerate(methods):
        df_m = results_df[results_df["method"] == method].sort_values("k")
        ks = df_m["k"].values.astype(int)
        cs = df_m["cindex"].values.astype(float)
        aopc = float(aopc_map.get(method, np.nan))
        label = f"{method} (AOPC={aopc:.4f})" if np.isfinite(aopc) else f"{method}"
        plt.plot(ks, cs, marker="o", linewidth=2, markersize=4, color=cmap(i % 10), label=label)

    plt.xlabel("删除/屏蔽的特征数量 (k)")
    plt.ylabel("C-index")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _read_val_csv(csv_dir: Path, cancer: str, fold: int, seed: int) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """读取验证集 CSV，并返回 (X_val_np, y_val_df, feature_names)。"""
    cancer_lower = str(cancer).lower()
    val_csv = csv_dir / f"{cancer_lower}_{fold}_val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    val_df = pd.read_csv(val_csv)
    dataset = RNAseqSurvivalDataset(val_df, label_col="survival_months", seed=int(seed))
    X_val = dataset.features

    if isinstance(X_val, pd.DataFrame):
        feature_names = X_val.columns.astype(str).tolist()
        X_val_np = X_val.values
    else:
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
        X_val_np = X_val.detach().cpu().numpy() if torch.is_tensor(X_val) else np.asarray(X_val)

    y_val_df = val_df[["survival_months", "censorship"]].copy()
    return X_val_np, y_val_df, feature_names


def _load_snn_model(model_path: Path, input_dim: int, device: torch.device) -> torch.nn.Module:
    """加载 model_genomic.SNN 权重（兼容 state_dict 或 {'model_state_dict': ...} 结构）。"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = SNN(omic_input_dim=int(input_dim), model_size_omic="small", n_classes=4).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _read_ranking_csv_as_indices(ranking_csv: Path, feature_to_index: Dict[str, int], n_features: int) -> np.ndarray:
    """读取 ranking CSV（feature_name, importance_score, rank），并映射为特征索引数组（从重要性高到低）。"""
    if not ranking_csv.exists():
        raise FileNotFoundError(f"Ranking CSV not found: {ranking_csv}")

    df = pd.read_csv(ranking_csv)
    if "feature_name" not in df.columns:
        raise ValueError(f"Invalid ranking CSV columns: {list(df.columns)}")

    feat_names = df["feature_name"].astype(str).tolist()
    missing = [f for f in feat_names if f not in feature_to_index]
    if len(missing) > 0:
        raise ValueError(
            f"Ranking CSV contains features not found in validation features (show first 10): {missing[:10]}"
        )

    indices = np.array([int(feature_to_index[f]) for f in feat_names], dtype=int)
    if len(indices) != n_features:
        raise ValueError(f"Ranking length mismatch: got {len(indices)}, expected {n_features}")

    return indices


def _default_xai_ranking_paths(xai_base_dir: Path, cancer: str, repeat: int, fold: int) -> Dict[str, Path]:
    """根据项目约定，构建 7 个 XAI 的 ranking CSV 路径。

    目录命名来自你的工程现状：
    - IG_results_2
    - DeepLIFT_results_2
    - deepshap_results_2 (DeepSHAP)
    - shap_results_2 (GradientSHAP)
    - PFI_results_2
    - LRP_results_2
    - KNN_CPI_results_2
    """
    cancer_upper = str(cancer).upper()
    r = int(repeat)
    f = int(fold)

    return {
        "IG": xai_base_dir / "IG_results_2" / cancer_upper / "ig_feature_importance" / f"repeat{r}_fold{f}_ig_feature_importance_ranking.csv",
        "DeepLIFT": xai_base_dir
        / "DeepLIFT_results_2"
        / cancer_upper
        / "deeplift_feature_importance"
        / f"repeat{r}_fold{f}_deeplift_feature_importance_ranking.csv",
        "DeepSHAP": xai_base_dir
        / "deepshap_results_2"
        / cancer_upper
        / "deepshap_feature_importance"
        / f"repeat{r}_fold{f}_deepshap_feature_importance_ranking.csv",
        "GradientSHAP": xai_base_dir
        / "shap_results_2"
        / cancer_upper
        / "shap_feature_importance"
        / f"repeat{r}_fold{f}_shap_feature_importance_ranking.csv",
        "PFI": xai_base_dir
        / "PFI_results_2"
        / cancer_upper
        / "pfi_feature_importance"
        / f"repeat{r}_fold{f}_pfi_feature_importance_ranking.csv",
        "LRP": xai_base_dir
        / "LRP_results_2"
        / cancer_upper
        / "lrp_feature_importance"
        / f"repeat{r}_fold{f}_lrp_feature_importance_ranking.csv",
        "KNN_CPI": xai_base_dir
        / "KNN_CPI_results_2"
        / cancer_upper
        / "knn_cpi_feature_importance"
        / f"repeat{r}_fold{f}_knn_cpi_feature_importance_ranking.csv",
    }


def _list_available_cancers(results_dir: Path) -> List[str]:
    """从 results_dir 下列出可用癌种目录名（大写）。"""
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")
    cancers = []
    for p in results_dir.iterdir():
        if p.is_dir():
            cancers.append(p.name.upper())
    return sorted(cancers)


def _evaluate_one_model(
    *,
    cancer: str,
    fold: int,
    repeat: int,
    csv_dir: Path,
    results_dir: Path,
    xai_base_dir: Path,
    model_path_override: Optional[Path],
    device: torch.device,
    k_steps: List[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """评估单个 (cancer, repeat, fold) 的 deletion curve & AOPC，并返回两张表。"""
    X_val_np, y_val_df, feature_names = _read_val_csv(csv_dir=csv_dir, cancer=cancer, fold=fold, seed=int(seed))
    n_features = int(X_val_np.shape[1])
    feature_to_index = {str(n): int(i) for i, n in enumerate(feature_names)}

    if model_path_override is not None:
        model_path = model_path_override
    else:
        model_path = results_dir / cancer / f"repeat{repeat}_s_{fold}_final_test_model.pt"

    model = _load_snn_model(model_path=model_path, input_dim=n_features, device=device)

    xai_paths = _default_xai_ranking_paths(xai_base_dir=xai_base_dir, cancer=cancer, repeat=repeat, fold=fold)
    xai_rankings: Dict[str, np.ndarray] = {}
    for method, path in xai_paths.items():
        xai_rankings[method] = _read_ranking_csv_as_indices(
            path,
            feature_to_index=feature_to_index,
            n_features=n_features,
        )

    results_df, aopc_df = evaluate_deletion_curve(
        model=model,
        X_val=X_val_np,
        y_val=y_val_df,
        xai_rankings=xai_rankings,
        compute_cindex_fn=default_compute_cindex_fn_snn,
        k_steps=k_steps,
        device=device,
        random_seed=int(seed),
    )

    results_df = results_df.copy()
    results_df.insert(0, "cancer", str(cancer).upper())
    results_df.insert(1, "repeat", int(repeat))
    results_df.insert(2, "fold", int(fold))

    aopc_df = aopc_df.copy()
    aopc_df.insert(0, "cancer", str(cancer).upper())
    aopc_df.insert(1, "repeat", int(repeat))
    aopc_df.insert(2, "fold", int(fold))

    return results_df, aopc_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XAI faithfulness via Deletion Curve & AOPC")
    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Project root directory. Other default paths are resolved relative to this.",
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default="",
        help="Single cancer type, e.g., BLCA. If empty and --all_cancers is set, will run all cancers.",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold id for single run, e.g., 0-4")
    parser.add_argument("--repeat", type=int, default=0, help="Repeat id for single run, e.g., 0-9")
    parser.add_argument(
        "--all_cancers",
        action="store_true",
        help="Run all cancers found under results_dir (batch mode)",
    )
    parser.add_argument(
        "--repeats",
        type=str,
        default="0-9",
        help="Repeat range for batch mode, e.g. 0-9",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0-4",
        help="Fold range for batch mode, e.g. 0-4",
    )

    parser.add_argument(
        "--csv_dir",
        type=str,
        default="",
        help="Validation CSV directory. If empty, use {project_root}/datasets_csv/preprocess_1 (contains {cancer_lower}_{fold}_val.csv)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Trained model directory root. If empty, use {project_root}/results_2 (contains {CANCER}/repeat{repeat}_s_{fold}_final_test_model.pt)",
    )
    parser.add_argument(
        "--xai_base_dir",
        type=str,
        default="",
        help="Base directory containing *_results_2 folders. If empty, use {project_root}",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional explicit model path (.pt). If empty, use results_dir/{CANCER}/repeat{repeat}_s_{fold}_final_test_model.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. If empty, use {project_root}/faithfulness_results",
    )

    parser.add_argument(
        "--k_steps",
        type=str,
        default="0,10,20,50,100,200,500,1000,1500,2000",
        help="Comma-separated k steps, e.g. 0,10,20,50,100,200,500,1000,1500,2000",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for Random baseline ranking")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu or cuda:0")

    args = parser.parse_args()

    def _parse_range(r: str) -> List[int]:
        s = str(r).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(s)]

    project_root = Path(args.project_root)
    csv_dir = Path(args.csv_dir) if str(args.csv_dir).strip() != "" else (project_root / "datasets_csv" / "preprocess_1")
    results_dir = Path(args.results_dir) if str(args.results_dir).strip() != "" else (project_root / "results_2")
    xai_base_dir = Path(args.xai_base_dir) if str(args.xai_base_dir).strip() != "" else project_root
    output_root = Path(args.output_dir) if str(args.output_dir).strip() != "" else (project_root / "faithfulness_results")
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"[Device] Using: {device} (requested: {args.device})")

    k_steps = [int(x) for x in str(args.k_steps).split(",") if str(x).strip() != ""]

    model_path_override = Path(args.model_path) if str(args.model_path).strip() != "" else None

    # Batch mode: all cancers / all repeat-fold
    if bool(args.all_cancers):
        cancers = _list_available_cancers(results_dir)
        repeats = _parse_range(args.repeats)
        folds = _parse_range(args.folds)

        all_curve_rows = []
        all_aopc_rows = []

        for cancer in cancers:
            for repeat in repeats:
                for fold in folds:
                    try:
                        curve_df, aopc_df = _evaluate_one_model(
                            cancer=cancer,
                            fold=fold,
                            repeat=repeat,
                            csv_dir=csv_dir,
                            results_dir=results_dir,
                            xai_base_dir=xai_base_dir,
                            model_path_override=None,
                            device=device,
                            k_steps=k_steps,
                            seed=int(args.seed),
                        )
                        all_curve_rows.append(curve_df)
                        all_aopc_rows.append(aopc_df)
                    except Exception as e:
                        print(f"[SKIP] cancer={cancer} repeat={repeat} fold={fold}: {e}")
                        continue

        if len(all_curve_rows) == 0 or len(all_aopc_rows) == 0:
            raise RuntimeError("No successful evaluations. Please check paths and inputs.")

        all_curve_df = pd.concat(all_curve_rows, ignore_index=True)
        all_aopc_df = pd.concat(all_aopc_rows, ignore_index=True)

        curve_csv = output_root / "all_models_deletion_curve.csv"
        aopc_csv = output_root / "all_models_aopc.csv"

        all_curve_df.to_csv(curve_csv, index=False)
        all_aopc_df.to_csv(aopc_csv, index=False)

        print(f"Saved: {curve_csv}")
        print(f"Saved: {aopc_csv}")
        return

    # Single mode
    cancer = str(args.cancer).upper()
    if cancer == "":
        raise ValueError("Please provide --cancer for single run, or use --all_cancers for batch run")

    fold = int(args.fold)
    repeat = int(args.repeat)

    curve_df, aopc_df = _evaluate_one_model(
        cancer=cancer,
        fold=fold,
        repeat=repeat,
        csv_dir=csv_dir,
        results_dir=results_dir,
        xai_base_dir=xai_base_dir,
        model_path_override=model_path_override,
        device=device,
        k_steps=k_steps,
        seed=int(args.seed),
    )

    output_dir = output_root / cancer / f"repeat{repeat}_fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "deletion_curve_results.csv"
    aopc_csv = output_dir / "aopc_summary.csv"
    fig_path = output_dir / "deletion_curve.png"

    curve_df.to_csv(results_csv, index=False)
    aopc_df.to_csv(aopc_csv, index=False)
    plot_deletion_curve(results_df=curve_df[["method", "k", "cindex"]], aopc_df=aopc_df[["method", "baseline_cindex", "aopc"]], output_path=str(fig_path))

    print(f"Saved: {results_csv}")
    print(f"Saved: {aopc_csv}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
