#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

CANCER = "BRCA"
METRIC_COL = "total_hit_same_any"
N_SAMPLED_MODELS = 50

XAI_DISPLAY_NAMES = {
    "shap": "G-SHAP",
    "deepshap": "D-SHAP"
}

CATEGORY_COLORS = {
    'Gradient': '#e6cf88',      # 黄色
    'Perturbation': '#e6889f',  # 粉红
    'Propagation': '#88cee6'    # 蓝色
}

XAI_COLORS = {
    "IG": "#F49E39",
    "shap": "#D3C0A3",
    "deepshap": "#9271B1",
    "LRP": "#A6DAEF",
    "DeepLIFT": "#66c1a4",
    "PFI": "#E26472"
}


def _xai_color(xai: str) -> str:
    """根据 XAI 方法名返回预设颜色；若未命中则返回灰色。"""
    x = str(xai)
    return XAI_COLORS.get(x, XAI_COLORS.get(x.lower(), "#888888"))

def _xai_label(xai: str) -> str:
    """将 XAI 方法名映射为展示用名称（如 shap -> G-SHAP）。"""
    x = str(xai)
    return XAI_DISPLAY_NAMES.get(x, XAI_DISPLAY_NAMES.get(x.lower(), x))

def _set_plot_style() -> None:
    """设置全局绘图字体大小（与 Fig1 风格保持一致）。"""
    # 核心设置：将 PDF 字体类型设为 42 (TrueType)，这样 AI 就能识别为文字
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 可选：强制使用 Arial 字体（BBRC 推荐）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["figure.titlesize"] = 8


def _p_to_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = int(p.size)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def friedman_test_paired(wide: pd.DataFrame) -> tuple[float, float, float, int, int]:
    wide = wide.dropna(axis=0, how="any")
    n_blocks = int(wide.shape[0])
    n_groups = int(wide.shape[1])
    if n_blocks == 0 or n_groups < 2:
        return np.nan, np.nan, np.nan, n_blocks, n_groups
    args = [wide[c].values for c in wide.columns]
    stat, p = friedmanchisquare(*args)
    kendalls_w = float(stat) / float(n_blocks * (n_groups - 1)) if n_blocks > 0 and n_groups > 1 else np.nan
    return float(stat), float(p), float(kendalls_w), n_blocks, n_groups


def pairwise_wilcoxon_paired(wide: pd.DataFrame) -> pd.DataFrame:
    from itertools import combinations

    wide = wide.dropna(axis=0, how="any")
    groups = list(wide.columns)
    rows = []
    for g1, g2 in combinations(groups, 2):
        x = wide[g1].values
        y = wide[g2].values
        try:
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
            stat = float(res.statistic) if res.statistic is not None else np.nan
            p_value = float(res.pvalue)
        except Exception:
            stat = np.nan
            p_value = np.nan
        rows.append(
            {
                "Group_1": str(g1),
                "Group_2": str(g2),
                "Statistic": stat,
                "p_value": p_value,
                "n_blocks": int(len(x)),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["p_fdr"] = _bh_fdr(out["p_value"].values)
        out["sig_fdr"] = out["p_fdr"] < 0.05
    return out


def write_statistical_tests_csv(
    output_path: Path,
    overall_row: dict,
    pairwise_df: pd.DataFrame,
    overall_title: str,
    pairwise_title: str,
) -> None:
    lines = []
    lines.append(f"# {overall_title}\n")
    lines.append(pd.DataFrame([overall_row]).to_csv(index=False))
    lines.append("\n")
    lines.append(f"# {pairwise_title}\n")
    if pairwise_df is None:
        pairwise_df = pd.DataFrame()
    lines.append(pairwise_df.to_csv(index=False))
    with open(output_path, "w") as f:
        f.writelines(lines)

def style_axis(ax) -> None:
    """统一坐标轴外观：黑色边框、加粗线宽、黑色刻度。"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    ax.tick_params(axis="both", colors="black", width=1.2)

def _apply_y_grid(ax) -> None:
    """给当前坐标轴添加 y 轴网格线，并将网格放在数据下方。"""
    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)

def _draw_boxplot_and_scatter(
    ax,
    data_arrays: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    box_alpha: float,
    scatter_alpha: float,
    scatter_size: float,
    jitter_sd: float = 0.04,
) -> None:
    """绘制 matplotlib boxplot 并叠加 jitter scatter（用于 Fig1 / Fig2 子图，保持一致风格）。"""
    positions = list(range(len(labels)))
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1)
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(float(box_alpha))
        patch.set_zorder(1)
    for k in ["whiskers", "caps", "medians", "fliers"]:
        for artist in bp.get(k, []):
            artist.set_zorder(2)

    for i, (y, color) in enumerate(zip(data_arrays, colors)):
        y = np.asarray(y)
        y = y[~np.isnan(y)]
        x = np.random.normal(i, float(jitter_sd), size=len(y))
        ax.scatter(x, y, alpha=float(scatter_alpha), s=float(scatter_size), color=color, zorder=3)

def _save_fig_png_pdf(fig, out_png: Path, dpi: int = 600) -> None:
    """同时保存 PNG 和 PDF（PDF 文件名与 PNG 同名，仅后缀不同）。"""
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_png).replace(".png", ".pdf"), bbox_inches="tight")

def _read_csv_required(path: Path) -> pd.DataFrame:
    """读取 CSV；若文件不存在则抛出 FileNotFoundError。"""
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)

def _xai_dir(output_dir: Path, cancer: str, xai: str) -> Path:
    """返回某个 cancer + xai 的输出目录（用于读取 model_statistics）。"""
    return output_dir / cancer / xai

def load_model_statistics(output_dir: Path, cancer: str, xai: str) -> pd.DataFrame:
    """加载 model_statistics.csv（每个模型/重复实验的统计汇总）。"""
    path = _xai_dir(output_dir, cancer, xai) / "model_statistics.csv"
    df = _read_csv_required(path)
    return df


def load_gene_validation(output_dir: Path, cancer: str, xai: str) -> pd.DataFrame:
    path = _xai_dir(output_dir, cancer, xai) / "gene_validation.csv"
    df = _read_csv_required(path)
    return df


def _extract_metric(ms: pd.DataFrame, xai: str, metric_col: str) -> pd.DataFrame:
    required = ["model_idx", metric_col]
    missing = [c for c in required if c not in ms.columns]
    if missing:
        raise ValueError(f"model_statistics.csv missing columns: {missing}")
    out = ms[["model_idx", metric_col]].copy()
    out["xai_method"] = str(xai)
    out["model_idx"] = out["model_idx"].astype(int)
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    return out


def _build_sampled_long_df(
    output_dir: Path,
    cancer: str,
    xai_methods: list[str],
    metric_col: str,
    n_models: int,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    rows = []
    for xai in xai_methods:
        ms = load_model_statistics(output_dir, cancer, xai)
        rows.append(_extract_metric(ms, xai=xai, metric_col=metric_col))

    df = pd.concat(rows, ignore_index=True)
    wide = df.pivot_table(index="model_idx", columns="xai_method", values=metric_col, aggfunc="median")
    wide = wide.reindex(columns=xai_methods)
    wide = wide.dropna(axis=0, how="any")

    available = sorted(wide.index.to_list())
    # 对齐 LGG 风格：使用配对可用的 replicate（默认应为 50 个），并保持确定性顺序
    sampled = available[: int(n_models)]

    wide_s = wide.loc[sampled]
    df_long = (
        wide_s.reset_index()
        .melt(id_vars=["model_idx"], var_name="xai_method", value_name=metric_col)
    )
    return df_long, xai_methods, sampled


def _to_repeat_fold(model_idx: int) -> tuple[int, int, str]:
    r = int(model_idx) // 5
    f = int(model_idx) % 5
    return r, f, f"{r}_{f}"


def build_plot_data_csv(
    df_long: pd.DataFrame,
    cancer: str,
    metric_col: str,
) -> pd.DataFrame:
    df = df_long.copy()
    df["Method"] = df["xai_method"].map(lambda x: _xai_label(str(x)))
    df["Cancer"] = str(cancer)
    tmp = df["model_idx"].apply(_to_repeat_fold)
    df["repeat"] = tmp.apply(lambda t: int(t[0]))
    df["fold"] = tmp.apply(lambda t: int(t[1]))
    df["replicate_id"] = tmp.apply(lambda t: str(t[2]))
    df["N_Factors"] = pd.to_numeric(df[metric_col], errors="coerce").astype(float)
    out = df[["Cancer", "Method", "repeat", "fold", "N_Factors", "replicate_id"]].copy()
    return out


def summarize_across_models_by_method(plot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in plot_df.groupby("Method", sort=False):
        vals = pd.to_numeric(sub["N_Factors"], errors="coerce").dropna().astype(float).values
        if vals.size == 0:
            continue
        q25, q75 = np.quantile(vals, [0.25, 0.75])
        rows.append(
            {
                "Method": str(method),
                "n_models": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "min": float(np.min(vals)),
                "q25": float(q25),
                "median": float(np.median(vals)),
                "q75": float(q75),
                "max": float(np.max(vals)),
                "IQR": float(q75 - q25),
            }
        )
    return pd.DataFrame(rows)


def summarize_across_cancers_by_xai(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for xai, sub in df.groupby("xai_method", sort=False):
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().astype(float).values
        if vals.size == 0:
            continue
        q25 = _nanquantile_safe(pd.Series(vals), 0.25)
        q75 = _nanquantile_safe(pd.Series(vals), 0.75)
        rows.append(
            {
                "xai_method": str(xai),
                "n_cancers": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "min": float(np.min(vals)),
                "q25": float(q25) if q25 is not None else np.nan,
                "median": float(np.median(vals)),
                "q75": float(q75) if q75 is not None else np.nan,
                "max": float(np.max(vals)),
                "IQR": float(q75 - q25) if (q25 is not None and q75 is not None and np.isfinite(q25) and np.isfinite(q75)) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_xai_boxplot_across_cancers(
    df: pd.DataFrame,
    value_col: str,
    xai_order: list[str],
    pairwise_df: pd.DataFrame,
    champion_label: str,
    out_png: Path,
    title: str,
    y_label: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(2.35, 2.6))
    draw_xai_boxplot_across_cancers(
        ax=ax,
        df=df,
        value_col=value_col,
        xai_order=xai_order,
        pairwise_df=pairwise_df,
        champion_label=champion_label,
        title=title,
        y_label=y_label,
    )
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def build_category_radar_plot_data_from_summaries(
    db_supported_summary_csv: Path,
    prognostic_summary_csv: Path,
    stability_summary_csv: Path,
    categories: list[str],
) -> pd.DataFrame:
    df_db = _read_csv_required(Path(db_supported_summary_csv)).copy()
    df_prog = _read_csv_required(Path(prognostic_summary_csv)).copy()
    df_stab = _read_csv_required(Path(stability_summary_csv)).copy()

    if "Category" not in df_db.columns or "median" not in df_db.columns:
        raise ValueError(f"Invalid DB-supported category summary CSV columns: {list(df_db.columns)}")
    if "Category" not in df_prog.columns or "median" not in df_prog.columns:
        raise ValueError(f"Invalid prognostic category summary CSV columns: {list(df_prog.columns)}")
    if "XAI_Category" not in df_stab.columns or "median" not in df_stab.columns:
        raise ValueError(f"Invalid stability category summary CSV columns: {list(df_stab.columns)}")

    categories = [str(c) for c in categories]
    metrics = [
        ("Prognostic factors", df_prog, "Category"),
        ("DB-supported hits", df_db, "Category"),
        ("Stability (Kuncheva)", df_stab, "XAI_Category"),
    ]

    rows = []
    for metric_name, df, cat_col in metrics:
        tmp = df[[cat_col, "median"]].copy()
        tmp[cat_col] = tmp[cat_col].astype(str)
        tmp["median"] = pd.to_numeric(tmp["median"], errors="coerce").astype(float)
        med_map = tmp.set_index(cat_col)["median"].to_dict()

        def _candidate_category_keys(cat: str) -> list[str]:
            cat = str(cat)
            keys = [cat]
            if cat.endswith("-based"):
                keys.append(cat[: -len("-based")])
            return keys

        def _lookup_median(cat: str) -> float:
            for k in _candidate_category_keys(cat):
                v = med_map.get(k, np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                if np.isfinite(v):
                    return v
            return np.nan

        missing = [c for c in categories if not np.isfinite(_lookup_median(c))]
        if len(missing) > 0:
            raise ValueError(f"Missing median values for metric '{metric_name}' categories: {missing}")

        vals = pd.Series({c: float(_lookup_median(c)) for c in categories}, dtype=float)
        vmin = float(np.nanmin(vals.values)) if np.isfinite(vals.values).any() else np.nan
        vmax = float(np.nanmax(vals.values)) if np.isfinite(vals.values).any() else np.nan
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            scores = pd.Series(0.5, index=vals.index, dtype=float)
        else:
            scores = (vals - vmin) / (vmax - vmin)

        for c in categories:
            rows.append(
                {
                    "xai_category": str(c),
                    "metric": str(metric_name),
                    "raw_value": float(vals.loc[c]),
                    "score": float(scores.loc[c]),
                }
            )

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["rank_metric"] = (
            out.groupby("metric")["raw_value"].rank(method="dense", ascending=False).astype(int)
        )
    composite = (
        out.groupby("xai_category", as_index=False)
        .agg(
            composite_score_mean=("score", "mean"),
            composite_score_sum=("score", "sum"),
        )
        .copy()
    )
    composite["rank_overall"] = (
        composite["composite_score_mean"].rank(method="dense", ascending=False).astype(int)
    )
    out = out.merge(composite, on="xai_category", how="left")
    return out


def plot_radar_per_category(
    radar_df: pd.DataFrame,
    category_order: list[str],
    metric_order: list[str],
    out_png: Path,
    title: str,
    eps: float = 0.05,
) -> None:
    _set_plot_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.4),
        subplot_kw={"polar": True},
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    n_metrics = int(len(metric_order))
    angles = np.linspace(0.0, 2.0 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0 or eps >= 1.0:
        eps = 0.0

    category_colors = CATEGORY_COLORS

    for ax, cat in zip(axes, category_order):
        sub = radar_df[radar_df["xai_category"].astype(str) == str(cat)].copy()
        val_map = sub.set_index("metric")["score"].to_dict()
        vals = np.array([float(val_map.get(str(m), np.nan)) for m in metric_order], dtype=float)
        vals_display = eps + vals * (1.0 - eps)
        vals_closed = np.concatenate([vals_display, vals_display[:1]])

        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(metric_order, fontsize=10)
        ax.set_yticks([eps, eps + 0.5 * (1.0 - eps), 1.0])
        ax.set_yticklabels(["", "", ""], fontsize=9)
        ax.set_ylim(0.0, 1.0)

        color = category_colors.get(str(cat), "#888888")
        ax.plot(angles_closed, vals_closed, color=color, linewidth=1.0)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)
        ax.set_title(str(cat), fontweight="bold", pad=14)

    for ax in axes[len(category_order) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontweight="bold")
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def plot_radar_overlay_xai(
    radar_df: pd.DataFrame,
    xai_order: list[str],
    metric_order: list[str],
    out_png: Path,
    title: str,
    eps: float = 0.05,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0), subplot_kw={"polar": True})

    n_metrics = int(len(metric_order))
    angles = np.linspace(0.0, 2.0 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0 or eps >= 1.0:
        eps = 0.0

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_order, fontsize=10)
    ax.set_yticks([eps, eps + 0.5 * (1.0 - eps), 1.0])
    ax.set_yticklabels(["", "", ""], fontsize=9)
    ax.set_ylim(0.0, 1.0)

    for xai in xai_order:
        sub = radar_df[radar_df["xai_method"].astype(str) == str(xai)].copy()
        val_map = sub.set_index("metric")["score"].to_dict()
        vals = np.array([float(val_map.get(str(m), np.nan)) for m in metric_order], dtype=float)
        vals_display = eps + vals * (1.0 - eps)
        vals_closed = np.concatenate([vals_display, vals_display[:1]])
        color = _xai_color(str(xai))
        ax.plot(
            angles_closed,
            vals_closed,
            color=color,
            linewidth=1.0,
            label=_xai_label(str(xai)),
        )
        ax.fill(
            angles_closed,
            vals_closed,
            color=color,
            alpha=0.15,
            zorder=1,
        )

    ax.set_title(title, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=8,
        title="XAI",
        title_fontsize=8,
    )
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#CCCCCC")
            leg.get_frame().set_linewidth(1.0)
        except Exception:
            pass
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def plot_radar_overlay_category(
    radar_df: pd.DataFrame,
    category_order: list[str],
    metric_order: list[str],
    out_png: Path,
    title: str,
    eps: float = 0.05,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0), subplot_kw={"polar": True})

    n_metrics = int(len(metric_order))
    angles = np.linspace(0.0, 2.0 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0 or eps >= 1.0:
        eps = 0.0

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(metric_order, fontsize=10)
    ax.set_yticks([eps, eps + 0.5 * (1.0 - eps), 1.0])
    ax.set_yticklabels(["", "", ""], fontsize=9)
    ax.set_ylim(0.0, 1.0)

    category_colors = CATEGORY_COLORS

    for cat in category_order:
        sub = radar_df[radar_df["xai_category"].astype(str) == str(cat)].copy()
        val_map = sub.set_index("metric")["score"].to_dict()
        vals = np.array([float(val_map.get(str(m), np.nan)) for m in metric_order], dtype=float)
        vals_display = eps + vals * (1.0 - eps)
        vals_closed = np.concatenate([vals_display, vals_display[:1]])
        color = category_colors.get(str(cat), "#888888")
        ax.plot(
            angles_closed,
            vals_closed,
            color=color,
            linewidth=1.0,
            label=str(cat),
        )
        ax.fill(
            angles_closed,
            vals_closed,
            color=color,
            alpha=0.15,
            zorder=1,
        )

    ax.set_title(title, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=8,
        title="Category",
        title_fontsize=8,
    )
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#CCCCCC")
            leg.get_frame().set_linewidth(1.0)
        except Exception:
            pass
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def _map_external_method_to_internal_xai(method: str) -> str:
    m = str(method)
    m_norm = m.strip()
    m_up = m_norm.upper()
    if m_up in {"G-SHAP", "G_SHAP", "GSHAP"}:
        return "shap"
    if m_up in {"D-SHAP", "D_SHAP", "DSHAP"}:
        return "deepshap"
    return m_norm


def build_radar_plot_data_from_summaries(
    db_supported_summary_csv: Path,
    prognostic_summary_csv: Path,
    stability_summary_csv: Path,
    xai_methods: list[str],
) -> pd.DataFrame:
    df_db = _read_csv_required(Path(db_supported_summary_csv)).copy()
    df_prog = _read_csv_required(Path(prognostic_summary_csv)).copy()
    df_stab = _read_csv_required(Path(stability_summary_csv)).copy()

    if "Method" not in df_db.columns or "median" not in df_db.columns:
        raise ValueError(f"Invalid DB-supported summary CSV columns: {list(df_db.columns)}")
    if "Method" not in df_prog.columns or "median" not in df_prog.columns:
        raise ValueError(f"Invalid prognostic summary CSV columns: {list(df_prog.columns)}")
    if "XAI_Method" not in df_stab.columns or "median" not in df_stab.columns:
        raise ValueError(f"Invalid stability summary CSV columns: {list(df_stab.columns)}")

    xai_methods = [str(m) for m in xai_methods]
    metrics = [
        ("Prognostic factors", df_prog, "Method"),
        ("DB-supported hits", df_db, "Method"),
        ("Stability (Kuncheva)", df_stab, "XAI_Method"),
    ]

    rows = []
    for metric_name, df, method_col in metrics:
        tmp = df[[method_col, "median"]].copy()
        tmp[method_col] = tmp[method_col].astype(str).map(_map_external_method_to_internal_xai)
        tmp["median"] = pd.to_numeric(tmp["median"], errors="coerce").astype(float)
        med_map = tmp.set_index(method_col)["median"].to_dict()

        missing = [m for m in xai_methods if not np.isfinite(float(med_map.get(m, np.nan)))]
        if len(missing) > 0:
            raise ValueError(
                f"Missing median values for metric '{metric_name}' in {str(df)}: {missing}"
            )

        vals = pd.Series({m: float(med_map.get(m, np.nan)) for m in xai_methods}, dtype=float)
        vmin = float(np.nanmin(vals.values)) if np.isfinite(vals.values).any() else np.nan
        vmax = float(np.nanmax(vals.values)) if np.isfinite(vals.values).any() else np.nan
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            scores = pd.Series(0.5, index=vals.index, dtype=float)
        else:
            scores = (vals - vmin) / (vmax - vmin)

        for m in xai_methods:
            rows.append(
                {
                    "xai_method": str(m),
                    "metric": str(metric_name),
                    "raw_value": float(vals.loc[m]),
                    "score": float(scores.loc[m]),
                }
            )

    out = pd.DataFrame(rows)
    out["xai_label"] = out["xai_method"].map(lambda x: _xai_label(str(x)))
    if len(out) > 0:
        out["rank_metric"] = (
            out.groupby("metric")["raw_value"].rank(method="dense", ascending=False).astype(int)
        )

    composite = (
        out.groupby("xai_method", as_index=False)
        .agg(
            composite_score_mean=("score", "mean"),
            composite_score_sum=("score", "sum"),
        )
        .copy()
    )
    composite["rank_overall"] = (
        composite["composite_score_mean"].rank(method="dense", ascending=False).astype(int)
    )
    out = out.merge(composite, on="xai_method", how="left")
    return out


def plot_radar_per_xai(
    radar_df: pd.DataFrame,
    xai_order: list[str],
    metric_order: list[str],
    out_png: Path,
    title: str,
    eps: float = 0.05,
) -> None:
    _set_plot_style()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 7.4),
        subplot_kw={"polar": True},
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    n_metrics = int(len(metric_order))
    angles = np.linspace(0.0, 2.0 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0 or eps >= 1.0:
        eps = 0.0

    for ax, xai in zip(axes, xai_order):
        sub = radar_df[radar_df["xai_method"].astype(str) == str(xai)].copy()
        val_map = sub.set_index("metric")["score"].to_dict()
        vals = np.array([float(val_map.get(str(m), np.nan)) for m in metric_order], dtype=float)
        vals_display = eps + vals * (1.0 - eps)
        vals_closed = np.concatenate([vals_display, vals_display[:1]])

        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(metric_order, fontsize=10)
        ax.set_yticks([eps, eps + 0.5 * (1.0 - eps), 1.0])
        ax.set_yticklabels(["", "", ""], fontsize=9)
        ax.set_ylim(0.0, 1.0)

        color = _xai_color(str(xai))
        ax.plot(angles_closed, vals_closed, color=color, linewidth=1.0)
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)
        ax.set_title(_xai_label(str(xai)), fontweight="bold", pad=14)

    for ax in axes[len(xai_order) :]:
        ax.set_visible(False)

    fig.suptitle(title, fontweight="bold")
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def build_breadth_depth_bubble_plot_data(
    df_breadth: pd.DataFrame,
    df_depth: pd.DataFrame,
    xai_order: list[str],
    size_min: float = 70.0,
    size_max: float = 900.0,
    gamma: float = 1.5,
) -> pd.DataFrame:
    b = df_breadth.groupby("xai_method", sort=False)["Breadth_U"].median().rename("value").reset_index()
    b["metric"] = "Breadth (U)"
    d = df_depth.groupby("xai_method", sort=False)["Depth_H_over_U"].median().rename("value").reset_index()
    d["metric"] = "Depth (H/U)"

    out = pd.concat([b, d], ignore_index=True)
    out["xai_method"] = out["xai_method"].astype(str)

    order_map = {str(x): i for i, x in enumerate([str(x) for x in xai_order])}
    out["x"] = out["xai_method"].map(lambda m: order_map.get(str(m), np.nan)).astype(float)
    y_map = {"Breadth (U)": 1.0, "Depth (H/U)": 0.0}
    out["y"] = out["metric"].map(lambda s: float(y_map.get(str(s), np.nan))).astype(float)
    out["xai_label"] = out["xai_method"].map(lambda m: _xai_label(str(m)))

    sizes = []
    for metric, sub in out.groupby("metric", sort=False):
        vals = pd.to_numeric(sub["value"], errors="coerce").astype(float)
        vmin = float(np.nanmin(vals.values)) if np.isfinite(vals.values).any() else np.nan
        vmax = float(np.nanmax(vals.values)) if np.isfinite(vals.values).any() else np.nan
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            norm = np.full(len(sub), 0.5, dtype=float)
        else:
            norm = (vals.values - vmin) / (vmax - vmin)
        g = float(gamma)
        if np.isfinite(g) and g > 0:
            norm = np.power(norm, g)
        s = float(size_min) + norm * (float(size_max) - float(size_min))
        sizes.append(pd.Series(s, index=sub.index))
    out["bubble_size"] = pd.concat(sizes).sort_index().values

    out = out.sort_values(["x", "y"], kind="mergesort").reset_index(drop=True)
    return out


def plot_breadth_depth_bubble(
    bubble_df: pd.DataFrame,
    xai_order: list[str],
    out_png: Path,
    title: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.8, 3.9))

    if bubble_df is None or len(bubble_df) == 0:
        ax.set_title(title, fontweight="bold")
        style_axis(ax)
        plt.tight_layout()
        _save_fig_png_pdf(fig, out_png)
        plt.close(fig)
        return

    colors = {
        "Breadth (U)": "#fdbb2d",
        "Depth (H/U)": "#1ABC9C",
    }
    for metric, sub in bubble_df.groupby("metric", sort=False):
        ax.scatter(
            sub["x"].values,
            sub["y"].values,
            s=sub["bubble_size"].values,
            color=colors.get(str(metric), "#888888"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=str(metric),
            zorder=3,
        )

    ax.set_xticks(np.arange(len(xai_order)))
    ax.set_xticklabels([_xai_label(str(m)) for m in xai_order], rotation=0, fontweight="bold")
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Depth (H/U)", "Breadth (U)"], fontweight="bold")
    ax.set_xlim(-0.6, float(len(xai_order) - 1) + 0.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_title(title, fontweight="bold")
    ax.grid(False)
    ax.legend(
        title="Metric",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#CCCCCC")
            leg.get_frame().set_linewidth(1.0)
        except Exception:
            pass
    style_axis(ax)
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def draw_xai_boxplot_across_cancers(
    ax,
    df: pd.DataFrame,
    value_col: str,
    xai_order: list[str],
    pairwise_df: pd.DataFrame,
    champion_label: str,
    title: str,
    y_label: str,
) -> None:
    data_arrays = [
        pd.to_numeric(df.loc[df["xai_method"].astype(str) == str(xai), value_col], errors="coerce").dropna().astype(float).values
        for xai in xai_order
    ]
    colors = [_xai_color(str(xai)) for xai in xai_order]
    labels = [_xai_label(str(xai)) for xai in xai_order]
    _draw_boxplot_and_scatter(
        ax,
        data_arrays=data_arrays,
        labels=labels,
        colors=colors,
        box_alpha=0.7,
        scatter_alpha=0.9,
        scatter_size=6,
        jitter_sd=0.06,
    )

    add_champion_significance_stars(ax, pairwise_df, labels, champion_label)
    _apply_y_grid(ax)
    ax.set_title(title)
    ax.set_xticks(range(len(xai_order)))
    ax.set_xticklabels([_xai_label(x) for x in xai_order], rotation=30, ha="center")
    ax.set_xlabel("XAI Method")
    ax.set_ylabel(y_label)
    style_axis(ax)


def add_champion_significance_stars(ax, pairwise_df: pd.DataFrame, order: list[str], champion: str) -> None:
    if pairwise_df is None or len(pairwise_df) == 0:
        return
    y_low, y_high = ax.get_ylim()
    y_range = y_high - y_low
    if y_range <= 0:
        y_range = 1.0
    try:
        y_data_max = float(ax.dataLim.y1)
    except Exception:
        y_data_max = float(y_high)
    y = y_data_max + 0.05 * y_range

    def _get_p(g1: str, g2: str) -> float:
        mask = ((pairwise_df["Group_1"] == g1) & (pairwise_df["Group_2"] == g2)) | (
            (pairwise_df["Group_1"] == g2) & (pairwise_df["Group_2"] == g1)
        )
        row = pairwise_df.loc[mask]
        if len(row) == 0:
            return np.nan
        return float(row.iloc[0].get("p_fdr", row.iloc[0].get("p_value", np.nan)))

    for i, g in enumerate(order):
        if str(g) == str(champion):
            continue
        p_val = _get_p(str(champion), str(g))
        lab = _p_to_stars(p_val)
        if not lab:
            continue
        ax.text(i, y, lab, ha="center", va="bottom", fontsize=8)
    if y > y_high:
        ax.set_ylim(y_low, y + 0.06 * y_range)


def plot_lgg_style_boxplot(
    df_long: pd.DataFrame,
    metric_col: str,
    xai_order: list[str],
    pairwise_df: pd.DataFrame,
    champion: str,
    out_png: Path,
    title: str,
    y_label: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    palette = {str(k): _xai_color(str(k)) for k in xai_order}

    # violin（背景层）
    sns.violinplot(
        data=df_long,
        x="xai_method",
        y=metric_col,
        order=xai_order,
        hue="xai_method",
        palette=palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1,
        ax=ax,
    )
    for coll in ax.collections:
        try:
            coll.set_alpha(0.7)
            coll.set_zorder(0)
        except Exception:
            pass

    data_arrays = [
        df_long[df_long["xai_method"].astype(str) == str(g)][metric_col].dropna().values
        for g in xai_order
    ]
    colors = [_xai_color(str(g)) for g in xai_order]
    _draw_boxplot_and_scatter(
        ax,
        data_arrays=data_arrays,
        labels=[str(g) for g in xai_order],
        colors=colors,
        box_alpha=0.7,
        scatter_alpha=1,
        scatter_size=6,
        jitter_sd=0.05,
    )

    # 显著性星号（champion vs others；使用 pairwise_df 的 FDR p 值）
    add_champion_significance_stars(ax, pairwise_df, [_xai_label(m) for m in xai_order], champion)

    _apply_y_grid(ax)
    #ax.set_title(title, fontweight="bold")
    #ax.set_xlabel("XAI", fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(xai_order)))
    ax.set_xticklabels([_xai_label(x) for x in xai_order], rotation=30, ha="center")
    style_axis(ax)
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def _paired_model_indices_for_cancer(
    output_dir: Path,
    cancer: str,
    xai_methods: list[str],
    n_models: int,
) -> list[int]:
    rows = []
    for xai in xai_methods:
        ms = load_model_statistics(output_dir, cancer, xai)
        if "model_idx" not in ms.columns:
            raise ValueError(f"model_statistics.csv missing columns: ['model_idx']")
        tmp = ms[["model_idx"]].copy()
        tmp["xai_method"] = str(xai)
        tmp["model_idx"] = tmp["model_idx"].astype(int)
        tmp["available"] = 1
        rows.append(tmp)

    df = pd.concat(rows, ignore_index=True)
    wide = df.pivot_table(index="model_idx", columns="xai_method", values="available", aggfunc="max")
    wide = wide.reindex(columns=xai_methods)
    wide = wide.dropna(axis=0, how="any")
    available = sorted(wide.index.to_list())
    return available[: int(n_models)]


def _nanquantile_safe(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").astype(float).values
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, float(q)))


def _compute_u_h_depth_for_models(gv: pd.DataFrame, model_indices: list[int]) -> pd.DataFrame:
    required = ["model_idx", "gene", "n_same_any"]
    missing = [c for c in required if c not in gv.columns]
    if missing:
        raise ValueError(f"gene_validation.csv missing columns: {missing}")

    wanted = set(int(i) for i in model_indices)
    sub = gv.loc[gv["model_idx"].astype(int).isin(wanted), ["model_idx", "gene", "n_same_any"]].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=["U", "H", "Depth"]).astype(float)

    sub["model_idx"] = sub["model_idx"].astype(int)
    sub["n_same_any"] = pd.to_numeric(sub["n_same_any"], errors="coerce").fillna(0.0).astype(float)
    sub = sub.drop_duplicates(subset=["model_idx", "gene"], keep="first")

    sub["supported"] = sub["n_same_any"] >= 1.0
    u = sub.groupby("model_idx", sort=False)["supported"].sum().astype(float)
    h = sub.groupby("model_idx", sort=False)["n_same_any"].sum().astype(float)
    out = pd.DataFrame({"U": u, "H": h})
    out["Depth"] = out["H"] / out["U"]
    out.loc[out["U"] <= 0, "Depth"] = np.nan
    return out


def _compute_discovery_proportions_for_models(gv: pd.DataFrame, model_indices: list[int]) -> tuple[pd.DataFrame, dict]:
    required = ["model_idx", "gene"]
    missing = [c for c in required if c not in gv.columns]
    if missing:
        raise ValueError(f"gene_validation.csv missing columns: {missing}")

    wanted = set(int(i) for i in model_indices)
    class_cols = [
        "oncokb_class",
        "dgidb_class",
        "opentargets_class",
        "cancermine_class",
    ]
    has_class_cols = all(c in gv.columns for c in class_cols)
    if not has_class_cols:
        raise ValueError(
            "gene_validation.csv missing required *_class columns for Fig7 (oncokb_class, dgidb_class, opentargets_class, cancermine_class)"
        )
    sub = gv.loc[
        gv["model_idx"].astype(int).isin(wanted),
        ["model_idx", "gene"] + class_cols,
    ].copy()
    if len(sub) == 0:
        empty = pd.DataFrame(columns=["p_known", "p_transfer", "p_novel"]).astype(float)
        return empty, {"conflict_rows": 0, "unknown_rows": 0}

    sub["model_idx"] = sub["model_idx"].astype(int)

    n_genes = sub.groupby("model_idx", sort=False)["gene"].size().astype(float)

    ot = sub["opentargets_class"].astype(str).str.lower()
    cm = sub["cancermine_class"].astype(str).str.lower()

    is_same_any = lambda s: s.isin(["same_only", "same_and_other"])
    is_other_only = lambda s: s.isin(["other_only"])

    known_flag_raw = is_same_any(ot) | is_same_any(cm)
    transfer_flag_raw = is_other_only(ot) | is_other_only(cm)
    conflict_flag = known_flag_raw & transfer_flag_raw

    known_flag = known_flag_raw
    transfer_flag = (~known_flag_raw) & transfer_flag_raw
    novel_flag = (~known_flag_raw) & (~transfer_flag_raw)
    unknown_flag = np.zeros(len(sub), dtype=bool)

    known_n = known_flag.groupby(sub["model_idx"], sort=False).sum().astype(float)
    transfer_n = transfer_flag.groupby(sub["model_idx"], sort=False).sum().astype(float)
    novel_n = novel_flag.groupby(sub["model_idx"], sort=False).sum().astype(float)

    denom = n_genes.replace(0, np.nan)
    out = pd.DataFrame(index=n_genes.index)
    out["p_known"] = known_n / denom
    out["p_transfer"] = transfer_n / denom
    out["p_novel"] = novel_n / denom
    out = out.reindex([int(i) for i in model_indices])

    stats = {"conflict_rows": int(conflict_flag.sum()), "unknown_rows": int(np.sum(unknown_flag))}
    return out, stats


def _compute_non_cancer_specific_evidence_for_models(gv: pd.DataFrame, model_indices: list[int]) -> tuple[pd.DataFrame, dict]:
    required = ["model_idx", "gene", "oncokb_class", "dgidb_class"]
    missing = [c for c in required if c not in gv.columns]
    if missing:
        raise ValueError(f"gene_validation.csv missing columns for non-cancer-specific evidence: {missing}")

    wanted = set(int(i) for i in model_indices)
    sub = gv.loc[gv["model_idx"].astype(int).isin(wanted), ["model_idx", "gene", "oncokb_class", "dgidb_class"]].copy()
    if len(sub) == 0:
        empty = pd.DataFrame(columns=["p_non_cancer_evidence"]).astype(float)
        return empty, {}

    sub["model_idx"] = sub["model_idx"].astype(int)
    n_genes = sub.groupby("model_idx", sort=False)["gene"].size().astype(float)

    ok_cls = sub["oncokb_class"].astype(str).str.lower()
    dg_cls = sub["dgidb_class"].astype(str).str.lower()
    valid = ["same_only", "same_and_other", "other_only"]
    ok = ok_cls.isin(valid)
    dg = dg_cls.isin(valid)
    supported = ok | dg
    supported_n = supported.groupby(sub["model_idx"], sort=False).sum().astype(float)

    denom = n_genes.replace(0, np.nan)
    out = pd.DataFrame(index=n_genes.index)
    out["p_non_cancer_evidence"] = supported_n / denom
    out = out.reindex([int(i) for i in model_indices])
    return out, {}


def build_discovery_preference_plot_data(
    output_dir: Path,
    cancers: list[str],
    xai_methods: list[str],
    n_models: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cancer in cancers:
        sampled = _paired_model_indices_for_cancer(output_dir, cancer=cancer, xai_methods=xai_methods, n_models=n_models)
        for xai in xai_methods:
            gv = load_gene_validation(output_dir, cancer=cancer, xai=xai)
            per_model, stats = _compute_discovery_proportions_for_models(gv, model_indices=[int(i) for i in sampled])
            print(
                f"[Fig7A] {cancer}-{xai}: conflict={stats.get('conflict_rows', 0)}; unknown={stats.get('unknown_rows', 0)}"
            )
            for model_idx, r in per_model.iterrows():
                rows.append(
                    {
                        "Cancer": str(cancer),
                        "xai_method": str(xai),
                        "model_idx": int(model_idx),
                        "p_known": float(r.get("p_known", np.nan)),
                        "p_transfer": float(r.get("p_transfer", np.nan)),
                        "p_novel": float(r.get("p_novel", np.nan)),
                    }
                )

    long_df = pd.DataFrame(rows)
    points_df = (
        long_df.groupby(["Cancer", "xai_method"], as_index=False)
        .agg(
            n_models=("model_idx", "nunique"),
            median_p_known=("p_known", "median"),
            q25_p_known=("p_known", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_known=("p_known", lambda s: _nanquantile_safe(s, 0.75)),
            median_p_transfer=("p_transfer", "median"),
            q25_p_transfer=("p_transfer", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_transfer=("p_transfer", lambda s: _nanquantile_safe(s, 0.75)),
            median_p_novel=("p_novel", "median"),
            q25_p_novel=("p_novel", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_novel=("p_novel", lambda s: _nanquantile_safe(s, 0.75)),
        )
    )
    return long_df, points_df


def build_non_cancer_specific_evidence_plot_data(
    output_dir: Path,
    cancers: list[str],
    xai_methods: list[str],
    n_models: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cancer in cancers:
        sampled = _paired_model_indices_for_cancer(output_dir, cancer=cancer, xai_methods=xai_methods, n_models=n_models)
        for xai in xai_methods:
            gv = load_gene_validation(output_dir, cancer=cancer, xai=xai)
            per_model, _ = _compute_non_cancer_specific_evidence_for_models(gv, model_indices=[int(i) for i in sampled])
            for model_idx, r in per_model.iterrows():
                rows.append(
                    {
                        "Cancer": str(cancer),
                        "xai_method": str(xai),
                        "model_idx": int(model_idx),
                        "p_non_cancer_evidence": float(r.get("p_non_cancer_evidence", np.nan)),
                    }
                )

    long_df = pd.DataFrame(rows)
    points_df = (
        long_df.groupby(["Cancer", "xai_method"], as_index=False)
        .agg(
            n_models=("model_idx", "nunique"),
            median_p_non_cancer_evidence=("p_non_cancer_evidence", "median"),
            q25_p_non_cancer_evidence=("p_non_cancer_evidence", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_non_cancer_evidence=("p_non_cancer_evidence", lambda s: _nanquantile_safe(s, 0.75)),
        )
    )
    return long_df, points_df


def plot_non_cancer_specific_evidence_bar(
    summary_xai: pd.DataFrame,
    xai_order: list[str],
    out_png: Path,
    title: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.3))

    if summary_xai is None or len(summary_xai) == 0:
        ax.set_title(title, fontweight="bold")
        style_axis(ax)
        plt.tight_layout()
        _save_fig_png_pdf(fig, out_png)
        plt.close(fig)
        return

    sub = summary_xai.set_index("xai_method").reindex([str(x) for x in xai_order])
    p_sup = pd.to_numeric(sub["median_p_non_cancer_evidence"], errors="coerce").astype(float).values
    p_sup = np.nan_to_num(p_sup, nan=0.0)
    p_sup = np.clip(p_sup, 0.0, 1.0)
    p_no = 1.0 - p_sup

    x = np.arange(len(xai_order))
    labels = [_xai_label(str(m)) for m in xai_order]
    c_supported = "#1a2a6c"
    c_not = "#b21f1f"

    ax.bar(x, p_sup, color=c_supported, edgecolor="black", linewidth=1, label="Supported")
    ax.bar(x, p_no, bottom=p_sup, color=c_not, edgecolor="black", linewidth=1, label="Not Supported")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linestyle="-", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(
        title="Evidence",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=8,
        title_fontsize=8
    )
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#CCCCCC")
            leg.get_frame().set_linewidth(1.0)
        except Exception:
            pass
    style_axis(ax)
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def plot_discovery_preference_stacked_bar(
    summary_xai: pd.DataFrame,
    xai_order: list[str],
    out_png: Path,
    title: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(3, 2.3))

    if summary_xai is None or len(summary_xai) == 0:
        ax.set_title(title, fontweight="bold")
        style_axis(ax)
        plt.tight_layout()
        _save_fig_png_pdf(fig, out_png)
        plt.close(fig)
        return

    sub = summary_xai.set_index("xai_method").reindex([str(x) for x in xai_order])
    pk = pd.to_numeric(sub["median_p_known"], errors="coerce").fillna(0.0).astype(float).values
    pt = pd.to_numeric(sub["median_p_transfer"], errors="coerce").fillna(0.0).astype(float).values
    pn = pd.to_numeric(sub["median_p_novel"], errors="coerce").fillna(0.0).astype(float).values

    s = pk + pt + pn
    s[s <= 0] = 1.0
    pk = pk / s
    pt = pt / s
    pn = pn / s

    x = np.arange(len(xai_order))
    labels = [_xai_label(str(m)) for m in xai_order]
    c_known = "#1a2a6c"
    c_transfer = "#D3C0A3"
    c_novel = "#b21f1f"

    ax.bar(x, pk, color=c_known, edgecolor="black", linewidth=1, label="Known")
    ax.bar(x, pt, bottom=pk, color=c_transfer, edgecolor="black", linewidth=1, label="Transfer")
    ax.bar(x, pn, bottom=pk + pt, color=c_novel, edgecolor="black", linewidth=1, label="Novel")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linestyle="-", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(
        title="Discovery class",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=8,            # legend 文本
        title_fontsize=8
    )
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("#CCCCCC")
            leg.get_frame().set_linewidth(1.0)
        except Exception:
            pass
    style_axis(ax)
    plt.tight_layout()
    _save_fig_png_pdf(fig, out_png)
    plt.close(fig)


def build_breadth_depth_plot_data(
    output_dir: Path,
    cancers: list[str],
    xai_methods: list[str],
    n_models: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cancer in cancers:
        sampled = _paired_model_indices_for_cancer(output_dir, cancer=cancer, xai_methods=xai_methods, n_models=n_models)
        for xai in xai_methods:
            gv = load_gene_validation(output_dir, cancer=cancer, xai=xai)
            metrics = _compute_u_h_depth_for_models(gv, model_indices=[int(i) for i in sampled])
            metrics = metrics.reindex([int(i) for i in sampled])
            for model_idx, r in metrics.iterrows():
                rows.append(
                    {
                        "Cancer": str(cancer),
                        "xai_method": str(xai),
                        "model_idx": int(model_idx),
                        "U": float(r.get("U", np.nan)),
                        "H": float(r.get("H", np.nan)),
                        "Depth": float(r.get("Depth", np.nan)),
                    }
                )

    long_df = pd.DataFrame(rows)

    points_df = (
        long_df.groupby(["Cancer", "xai_method"], as_index=False)
        .agg(
            median_U=("U", "median"),
            q25_U=("U", lambda s: _nanquantile_safe(s, 0.25)),
            q75_U=("U", lambda s: _nanquantile_safe(s, 0.75)),
            median_Depth=("Depth", "median"),
            q25_Depth=("Depth", lambda s: _nanquantile_safe(s, 0.25)),
            q75_Depth=("Depth", lambda s: _nanquantile_safe(s, 0.75)),
            n_models=("model_idx", "nunique"),
        )
    )

    return long_df, points_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-models", type=int, default=N_SAMPLED_MODELS)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import OUTPUT_DIR, XAI_METHODS, CANCER_TYPES, TCGA_DIR

    xai_methods = list(XAI_METHODS)
    out_dir = Path(args.out_dir) if args.out_dir else (OUTPUT_DIR / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_long, xai_order, sampled = _build_sampled_long_df(
        OUTPUT_DIR,
        cancer=CANCER,
        xai_methods=xai_methods,
        metric_col=METRIC_COL,
        n_models=int(args.n_models),
    )

    # 排序（与 LGG 图一致：按中位数从低到高排序）
    med = df_long.groupby("xai_method")[METRIC_COL].median().to_dict()
    xai_order = sorted(list(xai_order), key=lambda m: (float(med.get(m, np.inf)), str(m)))

    # 生成与 LGG 一致的 plot_data CSV（Cancer/Method/repeat/fold/N_Factors/replicate_id）
    plot_df = build_plot_data_csv(df_long, cancer=CANCER, metric_col=METRIC_COL)
    plot_df_out = plot_df.rename(columns={"N_Factors": str(METRIC_COL)})
    plot_df_out.to_csv(out_dir / f"supplementary_fig5_plot_data_{CANCER}_models.csv", index=False)

    # summary（与 LGG 一致的列名）
    summary_df = summarize_across_models_by_method(plot_df)
    summary_df.to_csv(out_dir / "supplementary_fig5_summary_across_models_by_method.csv", index=False)

    # Friedman + pairwise Wilcoxon（与 LGG 一致的 statistical_tests_{CANCER}_XAI_methods.csv 格式）
    wide_for_test = plot_df.pivot_table(index="replicate_id", columns="Method", values="N_Factors", aggfunc="median")
    wide_for_test = wide_for_test.reindex(columns=[_xai_label(m) for m in xai_order])
    stat, p_val, w, n_blocks, n_groups = friedman_test_paired(wide_for_test)
    pw = pairwise_wilcoxon_paired(wide_for_test)
    if pw is not None and len(pw) > 0:
        pw = pw[["Group_1", "Group_2", "Statistic", "p_value", "n_blocks", "p_fdr", "sig_fdr"]]

    overall_row = {
        "Test": "Friedman (paired by repeat+fold)",
        "Statistic": stat,
        "p_value": p_val,
        "Effect_Size_Kendalls_W": w,
        "n_replicates": n_blocks,
        "Significant": "Yes" if (p_val is not None and float(p_val) < 0.05) else "No",
    }
    stat_path = out_dir / f"statistical_tests_{CANCER}_XAI_methods.csv"
    write_statistical_tests_csv(
        stat_path,
        overall_row=overall_row,
        pairwise_df=pw,
        overall_title="Overall Test: Friedman Test (paired by repeat+fold)",
        pairwise_title="Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction",
    )

    champion = str(wide_for_test.median(axis=0, skipna=True).idxmax()) if wide_for_test.shape[1] else ""

    out_png = out_dir / f"fig5_{CANCER}_XAI_databases_supported_boxplot.png"
    plot_lgg_style_boxplot(
        df_long,
        metric_col=METRIC_COL,
        xai_order=xai_order,
        pairwise_df=pw,
        champion=champion,
        out_png=out_png,
        title=f"Total Supported Hits by XAI Method\n{CANCER} Cancer Type ({len(sampled)} Sampled Models)",
        y_label="Total Supported Hits",
    )

    cancers = list(CANCER_TYPES)
    long_uhr, points_uhr = build_breadth_depth_plot_data(
        OUTPUT_DIR,
        cancers=cancers,
        xai_methods=xai_methods,
        n_models=int(args.n_models),
    )

    long_uhr.to_csv(out_dir / "supplementary_fig6_plot_data_breadth_depth_models.csv", index=False)
    points_uhr.to_csv(out_dir / "supplementary_fig6_median_by_cancer_method.csv", index=False)

    df_breadth = points_uhr[["Cancer", "xai_method", "median_U"]].copy()
    df_breadth.rename(columns={"median_U": "Breadth_U"}, inplace=True)
    df_depth = points_uhr[["Cancer", "xai_method", "median_Depth"]].copy()
    df_depth.rename(columns={"median_Depth": "Depth_H_over_U"}, inplace=True)

    med_u = df_breadth.groupby("xai_method")["Breadth_U"].median().to_dict()
    xai_order_u = sorted(list(xai_methods), key=lambda m: (float(med_u.get(m, np.inf)), str(m)))
    med_d = df_depth.groupby("xai_method")["Depth_H_over_U"].median().to_dict()
    xai_order_d = sorted(list(xai_methods), key=lambda m: (float(med_d.get(m, np.inf)), str(m)))

    summarize_across_cancers_by_xai(df_breadth, value_col="Breadth_U").to_csv(
        out_dir / "supplementary_fig6a_summary_across_cancers_by_xai.csv", index=False
    )
    wide_u = df_breadth.pivot_table(index="Cancer", columns="xai_method", values="Breadth_U", aggfunc="median")
    wide_u = wide_u.reindex(columns=xai_order_u)
    wide_u.columns = [_xai_label(str(c)) for c in wide_u.columns]
    stat_u, p_u, w_u, n_blocks_u, _ = friedman_test_paired(wide_u)
    pw_u = pairwise_wilcoxon_paired(wide_u)
    if pw_u is not None and len(pw_u) > 0:
        pw_u = pw_u[["Group_1", "Group_2", "Statistic", "p_value", "n_blocks", "p_fdr", "sig_fdr"]]
    overall_u = {
        "Test": "Friedman (paired by cancer)",
        "Statistic": stat_u,
        "p_value": p_u,
        "Effect_Size_Kendalls_W": w_u,
        "n_cancers": n_blocks_u,
        "Significant": "Yes" if (p_u is not None and float(p_u) < 0.05) else "No",
    }
    write_statistical_tests_csv(
        out_dir / "statistical_tests_fig6a_breadth_U_XAI_methods.csv",
        overall_row=overall_u,
        pairwise_df=pw_u,
        overall_title="Overall Test: Friedman Test (paired by cancer)",
        pairwise_title="Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction",
    )
    champion_u = str(wide_u.median(axis=0, skipna=True).idxmax()) if wide_u.shape[1] else ""
    plot_xai_boxplot_across_cancers(
        df_breadth,
        value_col="Breadth_U",
        xai_order=xai_order_u,
        pairwise_df=pw_u,
        champion_label=champion_u if champion_u else "",
        out_png=out_dir / "fig6a_boxplot_xai_breadth_U.png",
        title="Breadth (U) Across Cancer Types",
        y_label="Supported genes (U)",
    )

    summarize_across_cancers_by_xai(df_depth, value_col="Depth_H_over_U").to_csv(
        out_dir / "supplementary_fig6b_summary_across_cancers_by_xai.csv", index=False
    )
    wide_d = df_depth.pivot_table(index="Cancer", columns="xai_method", values="Depth_H_over_U", aggfunc="median")
    wide_d = wide_d.reindex(columns=xai_order_d)
    wide_d.columns = [_xai_label(str(c)) for c in wide_d.columns]
    stat_d, p_d, w_d, n_blocks_d, _ = friedman_test_paired(wide_d)
    pw_d = pairwise_wilcoxon_paired(wide_d)
    if pw_d is not None and len(pw_d) > 0:
        pw_d = pw_d[["Group_1", "Group_2", "Statistic", "p_value", "n_blocks", "p_fdr", "sig_fdr"]]
    overall_d = {
        "Test": "Friedman (paired by cancer)",
        "Statistic": stat_d,
        "p_value": p_d,
        "Effect_Size_Kendalls_W": w_d,
        "n_cancers": n_blocks_d,
        "Significant": "Yes" if (p_d is not None and float(p_d) < 0.05) else "No",
    }
    write_statistical_tests_csv(
        out_dir / "statistical_tests_fig6b_depth_H_over_U_XAI_methods.csv",
        overall_row=overall_d,
        pairwise_df=pw_d,
        overall_title="Overall Test: Friedman Test (paired by cancer)",
        pairwise_title="Pairwise Comparisons: Wilcoxon Signed-Rank Test (paired) with FDR Correction",
    )
    champion_d = str(wide_d.median(axis=0, skipna=True).idxmax()) if wide_d.shape[1] else ""
    plot_xai_boxplot_across_cancers(
        df_depth,
        value_col="Depth_H_over_U",
        xai_order=xai_order_d,
        pairwise_df=pw_d,
        champion_label=champion_d if champion_d else "",
        out_png=out_dir / "fig6b_boxplot_xai_depth_H_over_U.png",
        title="Depth (H/U) Across Cancer Types",
        y_label="DB hits per supported gene (H/U)",
    )

    bubble_df = build_breadth_depth_bubble_plot_data(df_breadth=df_breadth, df_depth=df_depth, xai_order=xai_order_u)
    bubble_df.to_csv(out_dir / "supplementary_fig6c_bubble_plot_data_breadth_depth_by_xai.csv", index=False)
    plot_breadth_depth_bubble(
        bubble_df,
        xai_order=xai_order_u,
        out_png=out_dir / "fig6c_bubble_breadth_depth_by_xai.png",
        title="Breadth (U) and Depth (H/U) Across Cancer Types",
    )

    long_pref, points_pref = build_discovery_preference_plot_data(
        OUTPUT_DIR,
        cancers=cancers,
        xai_methods=xai_methods,
        n_models=int(args.n_models),
    )
    long_pref.to_csv(out_dir / "supplementary_fig7a_plot_data_cancer_specific_discovery_models.csv", index=False)
    points_pref.to_csv(out_dir / "supplementary_fig7a_proportions_median_by_cancer_method.csv", index=False)

    summary_pref = (
        points_pref.groupby("xai_method", as_index=False)
        .agg(
            n_cancers=("Cancer", "nunique"),
            median_p_known=("median_p_known", "median"),
            q25_p_known=("median_p_known", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_known=("median_p_known", lambda s: _nanquantile_safe(s, 0.75)),
            median_p_transfer=("median_p_transfer", "median"),
            q25_p_transfer=("median_p_transfer", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_transfer=("median_p_transfer", lambda s: _nanquantile_safe(s, 0.75)),
            median_p_novel=("median_p_novel", "median"),
            q25_p_novel=("median_p_novel", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_novel=("median_p_novel", lambda s: _nanquantile_safe(s, 0.75)),
        )
    )
    summary_pref.to_csv(out_dir / "supplementary_fig7a_summary_across_cancers_by_xai.csv", index=False)

    med_known = summary_pref.set_index("xai_method")["median_p_known"].to_dict()
    xai_order_fig7 = sorted(list(xai_methods), key=lambda m: (float(med_known.get(m, np.inf)), str(m)))
    plot_discovery_preference_stacked_bar(
        summary_pref,
        xai_order=xai_order_fig7,
        out_png=out_dir / "fig7a_cancer_specific_100pct_stacked_bar_xai_discovery_preference.png",
        title="Cancer-specific Discovery Preference",
    )

    long_nc, points_nc = build_non_cancer_specific_evidence_plot_data(
        OUTPUT_DIR,
        cancers=cancers,
        xai_methods=xai_methods,
        n_models=int(args.n_models),
    )
    long_nc.to_csv(out_dir / "supplementary_fig7b_plot_data_non_cancer_specific_models.csv", index=False)
    points_nc.to_csv(out_dir / "supplementary_fig7b_proportions_median_by_cancer_method.csv", index=False)

    summary_nc = (
        points_nc.groupby("xai_method", as_index=False)
        .agg(
            n_cancers=("Cancer", "nunique"),
            median_p_non_cancer_evidence=("median_p_non_cancer_evidence", "median"),
            q25_p_non_cancer_evidence=("median_p_non_cancer_evidence", lambda s: _nanquantile_safe(s, 0.25)),
            q75_p_non_cancer_evidence=("median_p_non_cancer_evidence", lambda s: _nanquantile_safe(s, 0.75)),
        )
    )
    summary_nc.to_csv(out_dir / "supplementary_fig7b_summary_across_cancers_by_xai.csv", index=False)

    med_nc = summary_nc.set_index("xai_method")["median_p_non_cancer_evidence"].to_dict()
    xai_order_fig7b = sorted(list(xai_methods), key=lambda m: (float(med_nc.get(m, np.inf)), str(m)))
    plot_non_cancer_specific_evidence_bar(
        summary_nc,
        xai_order=xai_order_fig7b,
        out_png=out_dir / "fig7b_bar_non_cancer_specific_evidence.png",
        title="Non-cancer-specific Evidence",
    )

    radar_plot_df = build_radar_plot_data_from_summaries(
        db_supported_summary_csv=TCGA_DIR
        / "biological_plausibility"
        / "outputs"
        / "visualizations"
        / "supplementary_fig1_summary_across_cancers_by_method.csv",
        prognostic_summary_csv=TCGA_DIR
        / "Prognostic_comparison_plots"
        / "supplementary_fig1_summary_across_cancers_by_method.csv",
        stability_summary_csv=TCGA_DIR
        / "stability_comparison_plots_nestedcv"
        / "visualization"
        / "supplementary_fig1_summary_across_cancers_by_method_top100_kuncheva_median.csv",
        xai_methods=xai_methods,
    )
    radar_plot_df.to_csv(out_dir / "supplementary_fig8_radar_plot_data_xai_3metrics_minmax.csv", index=False)
    plot_radar_per_xai(
        radar_df=radar_plot_df,
        xai_order=xai_methods,
        metric_order=["Prognostic factors", "DB-supported hits", "Stability (Kuncheva)"],
        out_png=out_dir / "fig8_radar_minmax_prognostic_db_stability.png",
        title="Min-max Normalized Composite Evaluation",
        eps=0.05,
    )
    plot_radar_overlay_xai(
        radar_df=radar_plot_df,
        xai_order=xai_methods,
        metric_order=["Prognostic factors", "DB-supported hits", "Stability (Kuncheva)"],
        out_png=out_dir / "fig8_radar_minmax_prognostic_db_stability_overlay_xai.png",
        title="Min-max Normalized Composite Evaluation",
        eps=0.05,
    )

    category_order = ["Perturbation-based", "Gradient-based", "Propagation-based"]
    radar_cat_df = build_category_radar_plot_data_from_summaries(
        db_supported_summary_csv=TCGA_DIR
        / "biological_plausibility"
        / "outputs"
        / "visualizations"
        / "supplementary_fig2_summary_across_cancers_by_category.csv",
        prognostic_summary_csv=TCGA_DIR
        / "Prognostic_comparison_plots"
        / "supplementary_fig2_summary_across_cancers_by_category.csv",
        stability_summary_csv=TCGA_DIR
        / "stability_comparison_plots_nestedcv"
        / "visualization"
        / "supplementary_fig2_summary_across_cancers_by_category_top100_kuncheva_median.csv",
        categories=category_order,
    )
    radar_cat_df.to_csv(out_dir / "supplementary_fig8_radar_plot_data_xai_categories_3metrics_minmax.csv", index=False)
    plot_radar_per_category(
        radar_df=radar_cat_df,
        category_order=category_order,
        metric_order=["Prognostic factors", "DB-supported hits", "Stability (Kuncheva)"],
        out_png=out_dir / "fig8_radar_minmax_categories_prognostic_db_stability.png",
        title="Min-max Normalized Composite Evaluation by XAI Category",
        eps=0.05,
    )
    plot_radar_overlay_category(
        radar_df=radar_cat_df,
        category_order=category_order,
        metric_order=["Prognostic factors", "DB-supported hits", "Stability (Kuncheva)"],
        out_png=out_dir / "fig8_radar_minmax_categories_prognostic_db_stability_overlay_categories.png",
        title="Min-max Normalized Composite Evaluation by XAI Category",
        eps=0.05,
    )

if __name__ == "__main__":
    main()