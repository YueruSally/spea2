import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import os

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = [f"S{i}" for i in range(12)]
ALGORITHMS = {"spea2": "spea2", "nsga2": "nsga2", "moead": "moead"}

COMPARISONS = [
    ("nsga2", "spea2", "NSGA-II", "SPEA2"),
    ("nsga2", "moead", "NSGA-II", "MOEA/D"),
]

N_RUNS = 30
REF_POINT = np.array([1.0, 1.0, 1.0])
OUTPUT_FILE = os.path.join(BASE_DIR, "wilcoxon_results.xlsx")


def load_front_hist(algo, scenario):
    fname = f"{algo}_{scenario.lower()}_front_hist.json"
    fpath = os.path.join(BASE_DIR, fname)
    with open(fpath, "r") as f:
        return json.load(f)


def load_all_fronts(scenario):
    result = {}
    for algo_key, algo_name in ALGORITHMS.items():
        hist = load_front_hist(algo_name, scenario)
        run_fronts = []
        for run in hist:
            last_gen = run[-1] if isinstance(run, list) else run
            run_fronts.append(np.array(last_gen))
        result[algo_key] = run_fronts
    return result


def compute_global_bounds(all_fronts):
    all_pts = []
    for run_fronts in all_fronts.values():
        for front in run_fronts:
            all_pts.append(front)
    all_pts = np.vstack(all_pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def normalize_front(front, global_min, global_max):
    denom = global_max - global_min
    denom[denom == 0] = 1.0
    return (front - global_min) / denom


def get_nondominated_front(points):
    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        diff = points - points[i]
        all_leq = np.all(diff <= 0, axis=1)
        any_lt = np.any(diff < 0, axis=1)
        dominators = all_leq & any_lt
        dominators[i] = False
        if np.any(dominators):
            is_dominated[i] = True
            continue
        all_geq = np.all(diff >= 0, axis=1)
        any_gt = np.any(diff > 0, axis=1)
        dominated_by_i = all_geq & any_gt
        dominated_by_i[i] = False
        is_dominated |= dominated_by_i
    return points[~is_dominated]


def get_unified_reference_front(all_fronts, global_min, global_max):
    all_pts = []
    for run_fronts in all_fronts.values():
        for front in run_fronts:
            all_pts.append(front)
    all_pts = np.vstack(all_pts)
    all_pts_norm = normalize_front(all_pts, global_min, global_max)
    return get_nondominated_front(all_pts_norm)


def compute_hv(front_norm, ref_point):
    mask = np.all(front_norm < ref_point, axis=1)
    front_valid = front_norm[mask]
    if len(front_valid) == 0:
        return 0.0
    indicator = HV(ref_point=ref_point)
    return indicator(front_valid)


def compute_igd_plus(front_norm, ref_front_norm):
    if len(front_norm) == 0:
        return np.inf
    indicator = IGDPlus(ref_front_norm)
    return indicator(front_norm)


def compute_spacing(front_norm):
    if len(front_norm) < 2:
        return 0.0
    diff = front_norm[:, np.newaxis, :] - front_norm[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists_matrix, np.inf)
    min_dists = dists_matrix.min(axis=1)
    mean_d = np.mean(min_dists)
    return np.sqrt(np.mean((min_dists - mean_d) ** 2))


def compute_metrics_per_run(scenario):
    all_fronts = load_all_fronts(scenario)
    global_min, global_max = compute_global_bounds(all_fronts)
    ref_front_norm = get_unified_reference_front(all_fronts, global_min, global_max)

    records = []
    for algo_key, run_fronts in all_fronts.items():
        for run_idx, front in enumerate(run_fronts):
            front_norm = normalize_front(front, global_min, global_max)
            hv = compute_hv(front_norm, REF_POINT)
            igd = compute_igd_plus(front_norm, ref_front_norm)
            sp = compute_spacing(front_norm)
            records.append({
                "Scenario": scenario,
                "Algorithm": algo_key,
                "Run": run_idx + 1,
                "HV": hv,
                "IGD+": igd,
                "Spacing": sp,
            })
    return pd.DataFrame(records)


def wilcoxon_pairwise(per_run_df, metric):
    """
    Mann-Whitney U 检验仍然基于原始 30 次运行数据(非参数检验,不依赖均值/方差假设),
    但 winner 判定改为比较 mean,以与汇总表展示的 mean ± std 保持一致。
    """
    all_rows = []
    for scenario in SCENARIOS:
        scenario_p_raws = []
        scenario_data = []
        for algo1_key, algo2_key, algo1_label, algo2_label in COMPARISONS:
            vals1 = per_run_df[
                (per_run_df.Scenario == scenario) &
                (per_run_df.Algorithm == algo1_key)
            ][metric].values
            vals2 = per_run_df[
                (per_run_df.Scenario == scenario) &
                (per_run_df.Algorithm == algo2_key)
            ][metric].values
            U, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
            scenario_data.append({
                "Scenario": scenario,
                "Comparison": f"{algo1_label}_vs_{algo2_label}",
                "U": U,
                "p_raw": p,
                "mean_Algo1": np.mean(vals1),
                "std_Algo1": np.std(vals1, ddof=1),
                "mean_Algo2": np.mean(vals2),
                "std_Algo2": np.std(vals2, ddof=1),
                "Algo1": algo1_label,
                "Algo2": algo2_label,
            })
            scenario_p_raws.append(p)
        _, p_adj, _, _ = multipletests(scenario_p_raws, method="holm")
        for i, row in enumerate(scenario_data):
            row["p_adj"] = p_adj[i]
            # HV 越大越好;IGD+ 和 Spacing 越小越好
            if metric == "HV":
                row["winner"] = row["Algo1"] if row["mean_Algo1"] >= row["mean_Algo2"] else row["Algo2"]
            else:
                row["winner"] = row["Algo1"] if row["mean_Algo1"] <= row["mean_Algo2"] else row["Algo2"]
        all_rows.extend(scenario_data)
    return pd.DataFrame(all_rows)


def fmt_mean_std(mean, std, decimals=4):
    """格式化为 'mean ± std' 字符串,便于论文/报告直接使用。"""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    print("Loading data and computing metrics...")
    all_runs = []
    for scenario in SCENARIOS:
        print(f"  {scenario}...")
        df = compute_metrics_per_run(scenario)
        all_runs.append(df)
    per_run_df = pd.concat(all_runs, ignore_index=True)

    # ── 算法对比汇总表(mean ± std)──
    algo_labels = [("spea2", "SPEA2"), ("nsga2", "NSGA-II"), ("moead", "MOEA/D")]
    comparison_sheets = {}        # 给人看的 'mean ± std' 字符串表
    comparison_sheets_raw = {}    # 给后续分析用的纯数值表(mean / std 分两列)

    for metric in ["HV", "IGD+", "Spacing"]:
        rows_str = []
        rows_raw = []
        for scenario in SCENARIOS:
            row_str = {"Scenario": scenario}
            row_raw = {"Scenario": scenario}
            means = {}
            for algo_key, algo_label in algo_labels:
                vals = per_run_df[
                    (per_run_df.Scenario == scenario) &
                    (per_run_df.Algorithm == algo_key)
                ][metric].values
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1))  # ddof=1 表示样本标准差,30 次运行用样本 std
                row_str[algo_label] = fmt_mean_std(m, s, decimals=4)
                row_raw[f"{algo_label}_mean"] = round(m, 6)
                row_raw[f"{algo_label}_std"] = round(s, 6)
                means[algo_label] = m
            # HV 越大越好;IGD+ / Spacing 越小越好
            if metric == "HV":
                best = max(means, key=means.get)
            else:
                best = min(means, key=means.get)
            row_str["Best"] = best
            row_raw["Best"] = best
            rows_str.append(row_str)
            rows_raw.append(row_raw)
        comparison_sheets[metric] = pd.DataFrame(rows_str)
        comparison_sheets_raw[metric] = pd.DataFrame(rows_raw)

    # ── Wilcoxon → WTL ──
    metric_sheets = {}
    for metric in ["HV", "IGD+", "Spacing"]:
        metric_sheets[metric] = wilcoxon_pairwise(per_run_df, metric)

    wtl_rows = []
    for metric in ["HV", "IGD+", "Spacing"]:
        ldf = metric_sheets[metric]
        for opponent_label in ["SPEA2", "MOEA/D"]:
            sub = ldf[ldf["Comparison"] == f"NSGA-II_vs_{opponent_label}"]
            wins = ties = losses = 0
            for _, rec in sub.iterrows():
                if rec["p_adj"] >= 0.05:
                    ties += 1
                elif rec["winner"] == "NSGA-II":
                    wins += 1
                else:
                    losses += 1
            wtl_rows.append({
                "Metric": metric,
                "Comparison": f"NSGA-II vs {opponent_label}",
                "Wins": wins,
                "Ties": ties,
                "Losses": losses,
            })
    wtl_sheet = pd.DataFrame(wtl_rows)

    # ── 输出 ──
    print(f"Writing results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # 给人看的合并版(mean ± std)
        for metric in ["HV", "IGD+", "Spacing"]:
            comparison_sheets[metric].to_excel(
                writer, sheet_name=f"{metric}_compare", index=False)
        # 给后续画图/分析用的纯数值版
        for metric in ["HV", "IGD+", "Spacing"]:
            sheet_name = f"{metric}_raw".replace("+", "p")  # IGD+ 在 sheet 名里替换一下
            comparison_sheets_raw[metric].to_excel(
                writer, sheet_name=sheet_name, index=False)
        # Wilcoxon 明细
        for metric in ["HV", "IGD+", "Spacing"]:
            sheet_name = f"{metric}_wilcoxon".replace("+", "p")
            metric_sheets[metric].to_excel(writer, sheet_name=sheet_name, index=False)
        # WTL 总结
        wtl_sheet.to_excel(writer, sheet_name="NSGA2_WTL", index=False)
    print("Done!")


if __name__ == "__main__":
    main()