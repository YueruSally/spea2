
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = [f"S{i}" for i in range(12)]
ALGORITHMS = {"spea2": "spea2", "nsga2": "nsga2", "moead": "moead"}

COMPARISONS = [
    ("spea2", "nsga2", "SPEA2", "NSGA-II"),
    ("spea2", "moead", "SPEA2", "MOEA/D"),
    ("nsga2", "moead", "NSGA-II", "MOEA/D"),
]

N_RUNS = 30  # 修正: 实际是 30 次 run, 不是 20
REF_POINT = np.array([1.0, 1.0, 1.0])  # 修正: 归一化后统一参考点
OUTPUT_FILE = os.path.join(BASE_DIR, "wilcoxon_results.xlsx")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_front_hist(algo, scenario):
    fname = f"{algo}_{scenario.lower()}_front_hist.json"
    fpath = os.path.join(BASE_DIR, fname)
    with open(fpath, "r") as f:
        return json.load(f)


def load_all_fronts(scenario):
    """加载三个算法在同一场景下的所有 run 的最终 Pareto 前沿"""
    result = {}
    for algo_key, algo_name in ALGORITHMS.items():
        hist = load_front_hist(algo_name, scenario)
        run_fronts = []
        for run in hist:
            last_gen = run[-1] if isinstance(run, list) else run
            run_fronts.append(np.array(last_gen))
        result[algo_key] = run_fronts
    return result


# ─────────────────────────────────────────────
# 修正1: 归一化
# ─────────────────────────────────────────────
def compute_global_bounds(all_fronts):
    """合并三个算法所有 run 的所有点, 计算每维的全局 min/max"""
    all_pts = []
    for run_fronts in all_fronts.values():
        for front in run_fronts:
            all_pts.append(front)
    all_pts = np.vstack(all_pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def normalize_front(front, global_min, global_max):
    """将前沿点归一化到 [0, 1]"""
    denom = global_max - global_min
    denom[denom == 0] = 1.0  # 避免除零
    return (front - global_min) / denom


# ─────────────────────────────────────────────
# 修正4: 非支配过滤 (向量化)
# ─────────────────────────────────────────────
def get_nondominated_front(points):
    """向量化非支配排序, 返回非支配点集"""
    n = len(points)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        # 向量化: 一次比较 points[i] 与所有其他点
        diff = points - points[i]
        # j dominates i: all(points[j] <= points[i]) and any(points[j] < points[i])
        all_leq = np.all(diff <= 0, axis=1)
        any_lt = np.any(diff < 0, axis=1)
        dominators = all_leq & any_lt
        dominators[i] = False
        if np.any(dominators):
            is_dominated[i] = True
            continue
        # i dominates j: 标记被 i 支配的点
        all_geq = np.all(diff >= 0, axis=1)
        any_gt = np.any(diff > 0, axis=1)
        dominated_by_i = all_geq & any_gt
        dominated_by_i[i] = False
        is_dominated |= dominated_by_i
    return points[~is_dominated]


def get_unified_reference_front(all_fronts, global_min, global_max):
    """三个算法所有 run 合并 → 归一化 → 取非支配前沿"""
    all_pts = []
    for run_fronts in all_fronts.values():
        for front in run_fronts:
            all_pts.append(front)
    all_pts = np.vstack(all_pts)
    all_pts_norm = normalize_front(all_pts, global_min, global_max)
    return get_nondominated_front(all_pts_norm)


# ─────────────────────────────────────────────
# 修正3: 使用 pymoo 的 HV / IGD+ 标准实现
# ─────────────────────────────────────────────
def compute_hv(front_norm, ref_point):
    """使用 pymoo 计算 Hypervolume"""
    # pymoo HV 要求所有点严格被 ref_point 支配
    mask = np.all(front_norm < ref_point, axis=1)
    front_valid = front_norm[mask]
    if len(front_valid) == 0:
        return 0.0
    indicator = HV(ref_point=ref_point)
    return indicator(front_valid)


def compute_igd_plus(front_norm, ref_front_norm):
    """使用 pymoo 计算 IGD+"""
    if len(front_norm) == 0:
        return np.inf
    indicator = IGDPlus(ref_front_norm)
    return indicator(front_norm)


def compute_spacing(front_norm):
    """Spacing 指标 (不依赖参考点, 直接在归一化空间计算)"""
    if len(front_norm) < 2:
        return 0.0
    # 向量化计算所有点对距离
    diff = front_norm[:, np.newaxis, :] - front_norm[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists_matrix, np.inf)
    min_dists = dists_matrix.min(axis=1)
    mean_d = np.mean(min_dists)
    return np.sqrt(np.mean((min_dists - mean_d) ** 2))


# ─────────────────────────────────────────────
# 计算每个 run 的指标 (整合修正 1-4)
# ─────────────────────────────────────────────
def compute_metrics_per_run(scenario):
    """三个算法使用同一场景下的归一化+统一参考点/前沿计算指标"""
    all_fronts = load_all_fronts(scenario)

    # 修正1: 全局 min/max
    global_min, global_max = compute_global_bounds(all_fronts)

    # 修正1+2: 归一化后的统一参考前沿
    ref_front_norm = get_unified_reference_front(all_fronts, global_min, global_max)

    records = []
    for algo_key, run_fronts in all_fronts.items():
        for run_idx, front in enumerate(run_fronts):
            # 归一化
            front_norm = normalize_front(front, global_min, global_max)

            # 修正2+3: 使用 pymoo 在归一化空间计算
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


# ─────────────────────────────────────────────
# WILCOXON (Mann-Whitney U) 两两对比
# 修正5: Bonferroni 场景内 3 对比较校正 (保持不变, 已合理)
# ─────────────────────────────────────────────
def effect_size_r(U, n1, n2):
    z = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return z / np.sqrt(n1 + n2)


def effect_label(r):
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


def wilcoxon_pairwise(per_run_df, metric):
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
            r = effect_size_r(U, len(vals1), len(vals2))

            scenario_data.append({
                "Scenario": scenario,
                "Comparison": f"{algo1_label}_vs_{algo2_label}",
                "Algo1": algo1_label,
                "Algo2": algo2_label,
                "U": U,
                "p_raw": round(p, 4),
                "r": round(r, 3),
                "effect": effect_label(r),
                "med_Algo1": round(np.median(vals1), 6),
                "med_Algo2": round(np.median(vals2), 6),
            })
            scenario_p_raws.append(p)

        # 修正5: 场景内 Bonferroni 校正 (3 次比较)
        _, p_adj, _, _ = multipletests(scenario_p_raws, method="bonferroni")

        for i, row in enumerate(scenario_data):
            p_a = round(p_adj[i], 4)
            row["p_adj"] = p_a
            sig = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "ns"
            row["sig"] = sig

            if metric == "HV":
                winner = row["Algo1"] if row["med_Algo1"] >= row["med_Algo2"] else row["Algo2"]
            else:
                winner = row["Algo1"] if row["med_Algo1"] <= row["med_Algo2"] else row["Algo2"]
            row["winner"] = winner

        all_rows.extend(scenario_data)

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("compare_v2.py — 修正版")
    print("  1. 归一化到 [0,1]")
    print("  2. 参考点 [1.1, 1.1, 1.1]")
    print("  3. pymoo HV/IGD+ 标准实现")
    print("  4. 向量化非支配过滤")
    print("  5. Bonferroni 场景内 3 对校正, N_RUNS=30")
    print("=" * 60)

    print("\nLoading data and computing metrics...")
    all_runs = []
    for scenario in SCENARIOS:
        print(f"  {scenario}...")
        df = compute_metrics_per_run(scenario)
        all_runs.append(df)
    per_run_df = pd.concat(all_runs, ignore_index=True)

    # 验证 run 数
    for scenario in SCENARIOS:
        for algo in ALGORITHMS:
            n = len(per_run_df[(per_run_df.Scenario == scenario) & (per_run_df.Algorithm == algo)])
            assert n == N_RUNS, f"{scenario}/{algo}: expected {N_RUNS} runs, got {n}"
    print(f"\n  ✓ All scenarios × algorithms have {N_RUNS} runs")

    # Wilcoxon 两两对比
    metric_sheets = {}
    for metric in ["HV", "IGD+", "Spacing"]:
        print(f"  Wilcoxon for {metric}...")
        df_w = wilcoxon_pairwise(per_run_df, metric)
        metric_sheets[metric] = df_w

    # 宽表
    wide_sheets = {}
    for metric in ["HV", "IGD+", "Spacing"]:
        long_df = metric_sheets[metric]
        wide = long_df.pivot(index="Scenario", columns="Comparison",
                             values=["U", "p_raw", "p_adj", "sig", "r", "effect",
                                     "med_Algo1", "med_Algo2", "winner"])
        wide.columns = ["_".join(col).strip() for col in wide.columns]
        wide = wide.reset_index()
        wide_sheets[metric] = wide

    # 描述统计
    desc_rows = []
    for metric in ["HV", "IGD+", "Spacing"]:
        for scenario in SCENARIOS:
            for algo_key, algo_label in [("spea2", "SPEA2"), ("nsga2", "NSGA-II"), ("moead", "MOEA/D")]:
                vals = per_run_df[
                    (per_run_df.Scenario == scenario) &
                    (per_run_df.Algorithm == algo_key)
                ][metric].values
                desc_rows.append({
                    "Metric": metric,
                    "Scenario": scenario,
                    "Algorithm": algo_label,
                    "Mean": round(np.mean(vals), 6),
                    "Std": round(np.std(vals), 6),
                    "Median": round(np.median(vals), 6),
                    "Min": round(np.min(vals), 6),
                    "Max": round(np.max(vals), 6),
                })
    desc_sheet = pd.DataFrame(desc_rows)

    # ── 算法对比汇总表: 同一场景下三算法 Mean 并排 ──
    algo_labels = [("spea2", "SPEA2"), ("nsga2", "NSGA-II"), ("moead", "MOEA/D")]
    comparison_sheets = {}
    for metric in ["HV", "IGD+", "Spacing"]:
        rows = []
        for scenario in SCENARIOS:
            row = {"Scenario": scenario}
            for algo_key, algo_label in algo_labels:
                vals = per_run_df[
                    (per_run_df.Scenario == scenario) &
                    (per_run_df.Algorithm == algo_key)
                ][metric].values
                row[algo_label] = round(np.mean(vals), 6)
            # 标记最优算法 (HV越大越好, IGD+/Spacing越小越好)
            means = {al: row[al] for _, al in algo_labels}
            if metric == "HV":
                row["Best"] = max(means, key=means.get)
            else:
                row["Best"] = min(means, key=means.get)
            rows.append(row)
        comparison_sheets[metric] = pd.DataFrame(rows)

    # ── SPEA2 vs 其他算法: Wins/Ties/Losses (α=0.05) ──
    # 只看 SPEA2 参与的两组对比: SPEA2_vs_NSGA-II, SPEA2_vs_MOEA/D
    wtl_rows = []
    for metric in ["HV", "IGD+", "Spacing"]:
        higher_better = (metric == "HV")
        ldf = metric_sheets[metric]
        for opponent_label in ["NSGA-II", "MOEA/D"]:
            comp_name = f"SPEA2_vs_{opponent_label}"
            sub = ldf[ldf["Comparison"] == comp_name]
            wins = ties = losses = 0
            for _, rec in sub.iterrows():
                if rec["p_adj"] >= 0.05:
                    ties += 1
                elif rec["winner"] == "SPEA2":
                    wins += 1
                else:
                    losses += 1
            wtl_rows.append({
                "Metric": metric,
                "Comparison": f"SPEA2 vs {opponent_label}",
                "Wins": wins,
                "Ties": ties,
                "Losses": losses,
            })
    wtl_sheet = pd.DataFrame(wtl_rows)

    print(f"\nWriting results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # 算法对比汇总表 (放在最前面, 方便查看)
        for metric in ["HV", "IGD+", "Spacing"]:
            comparison_sheets[metric].to_excel(
                writer, sheet_name=f"{metric}_compare", index=False)
        # Wilcoxon 长表
        for metric in ["HV", "IGD+", "Spacing"]:
            metric_sheets[metric].to_excel(writer, sheet_name=f"{metric}_long", index=False)
        # Wilcoxon 宽表
        for metric in ["HV", "IGD+", "Spacing"]:
            wide_sheets[metric].to_excel(writer, sheet_name=f"{metric}_wide", index=False)
        per_run_df.to_excel(writer, sheet_name="Per_run", index=False)
        desc_sheet.to_excel(writer, sheet_name="Descriptive", index=False)
        wtl_sheet.to_excel(writer, sheet_name="SPEA2_WTL", index=False)

    print("Done!")


if __name__ == "__main__":
    main()
