import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import product
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = [f"S{i}" for i in range(12)]
ALGORITHMS = {"spea2": "spea2", "nsga2": "nsga2", "moead": "moead"}
COMPARISONS = [("spea2", "nsga2", "NSGA-II"), ("spea2", "moead", "MOEA/D")]
N_RUNS = 20
OUTPUT_FILE = os.path.join(BASE_DIR, "wilcoxon_results.xlsx")


def load_front_hist(algo, scenario):
    """Load front history json for given algo and scenario."""
    fname = f"{algo}_{scenario.lower()}_front_hist.json"
    fpath = os.path.join(BASE_DIR, fname)
    with open(fpath, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_hv(pareto_front, ref_point):
    """Hypervolume via inclusion-exclusion (3-objective, WFG not required for small fronts)."""
    from scipy.spatial import ConvexHull
    # Simple sweep-based HV for 3D
    front = np.array(pareto_front)
    # Filter dominated by ref
    front = front[np.all(front < ref_point, axis=1)]
    if len(front) == 0:
        return 0.0
    return _hv_3d(front, ref_point)


def _hv_3d(front, ref):
    """3-objective hypervolume by slicing along 3rd axis."""
    front = front[np.argsort(front[:, 2])]
    hv = 0.0
    prev_z = ref[2]
    for i in range(len(front) - 1, -1, -1):
        z = front[i, 2]
        slice_front = front[:i+1, :2]
        slice_hv = _hv_2d(slice_front, ref[:2])
        hv += slice_hv * (prev_z - z)
        prev_z = z
    return hv


def _hv_2d(front, ref):
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for point in front:
        if point[1] < prev_y:
            hv += (ref[0] - point[0]) * (prev_y - point[1])
            prev_y = point[1]
    return hv


def compute_igd_plus(pareto_front, reference_front):
    """IGD+ metric."""
    front = np.array(pareto_front)
    ref = np.array(reference_front)
    total = 0.0
    for r in ref:
        dists = np.sqrt(np.sum(np.maximum(front - r, 0) ** 2, axis=1))
        total += np.min(dists)
    return total / len(ref)


def compute_spacing(pareto_front):
    """Spacing metric."""
    front = np.array(pareto_front)
    if len(front) < 2:
        return 0.0
    dists = []
    for i, p in enumerate(front):
        d = np.sqrt(np.sum((front - p) ** 2, axis=1))
        d[i] = np.inf
        dists.append(np.min(d))
    dists = np.array(dists)
    mean_d = np.mean(dists)
    return np.sqrt(np.mean((dists - mean_d) ** 2))


# ─────────────────────────────────────────────
# BUILD PER-RUN DATA
# ─────────────────────────────────────────────
def get_ref_point(scenario, all_algo_fronts):
    """Unified reference point: max across all algorithms for this scenario."""
    all_pts = []
    for pts in all_algo_fronts.values():
        for run_front in pts:
            all_pts.extend(run_front)
    arr = np.array(all_pts)
    return arr.max(axis=0) * 1.1


def get_unified_reference_front(scenario, all_algo_fronts):
    """Combined reference front from all algorithms across all runs."""
    all_pts = []
    for pts in all_algo_fronts.values():
        for run_front in pts:
            all_pts.extend(run_front)
    # Non-dominated sort
    arr = np.array(all_pts)
    dominated = np.zeros(len(arr), dtype=bool)
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j:
                if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                    dominated[i] = True
                    break
    return arr[~dominated]


def load_all_fronts(scenario):
    """Returns dict: algo -> list of run fronts (list of [x,y,z] points)."""
    result = {}
    for algo_key, algo_name in ALGORITHMS.items():
        hist = load_front_hist(algo_name, scenario)
        # hist is list of runs; each run is list of generations; take last gen
        run_fronts = []
        for run in hist:
            last_gen = run[-1] if isinstance(run, list) else run
            run_fronts.append(last_gen)
        result[algo_key] = run_fronts
    return result


def compute_metrics_per_run(scenario):
    """Compute HV, IGD+, Spacing for each run for each algorithm."""
    all_fronts = load_all_fronts(scenario)
    ref_point = get_ref_point(scenario, all_fronts)
    ref_front = get_unified_reference_front(scenario, all_fronts)

    records = []
    for algo_key, run_fronts in all_fronts.items():
        for run_idx, front in enumerate(run_fronts):
            front_arr = np.array(front)
            hv = compute_hv(front_arr, ref_point)
            igd = compute_igd_plus(front_arr, ref_front)
            sp = compute_spacing(front_arr)
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
# WILCOXON / MANN-WHITNEY
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


def wilcoxon_analysis(per_run_df, metric, base_algo, comp_algo, comp_label):
    """Run Mann-Whitney U tests per scenario."""
    rows = []
    p_raws = []
    scenarios = SCENARIOS

    for scenario in scenarios:
        base = per_run_df[(per_run_df.Scenario == scenario) & (per_run_df.Algorithm == base_algo)][metric].values
        comp = per_run_df[(per_run_df.Scenario == scenario) & (per_run_df.Algorithm == comp_algo)][metric].values
        U, p = mannwhitneyu(base, comp, alternative="two-sided")
        r = effect_size_r(U, len(base), len(comp))
        rows.append({
            "Scenario": scenario,
            f"SPEA2_vs_{comp_label}_U": U,
            f"SPEA2_vs_{comp_label}_p_raw": round(p, 4),
            f"SPEA2_vs_{comp_label}_r": round(r, 3),
            f"SPEA2_vs_{comp_label}_effect": effect_label(r),
            f"SPEA2_vs_{comp_label}_med_Base": round(np.median(base), 6),
            f"SPEA2_vs_{comp_label}_med_Comp": round(np.median(comp), 6),
        })
        p_raws.append(p)

    # Bonferroni correction
    _, p_adj, _, _ = multipletests(p_raws, method="bonferroni")
    for i, row in enumerate(rows):
        p_a = round(p_adj[i], 4)
        row[f"SPEA2_vs_{comp_label}_p_adj"] = p_a
        sig = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "ns"
        row[f"SPEA2_vs_{comp_label}_sig"] = sig

        # Winner (for HV: higher better; for IGD+/Spacing: lower better)
        med_base = row[f"SPEA2_vs_{comp_label}_med_Base"]
        med_comp = row[f"SPEA2_vs_{comp_label}_med_Comp"]
        if metric == "HV":
            winner = "SPEA2" if med_base >= med_comp else comp_label
        else:
            winner = "SPEA2" if med_base <= med_comp else comp_label
        row[f"SPEA2_vs_{comp_label}_winner"] = winner

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading data and computing metrics...")
    all_runs = []
    for scenario in SCENARIOS:
        print(f"  {scenario}...")
        df = compute_metrics_per_run(scenario)
        all_runs.append(df)
    per_run_df = pd.concat(all_runs, ignore_index=True)

    # Build Per_run sheet (SPEA2 vs each competitor)
    per_run_rows = []
    for scenario in SCENARIOS:
        for _, comp_label in [("nsga2", "NSGA-II"), ("moead", "MOEA/D")]:
            spea2 = per_run_df[(per_run_df.Scenario == scenario) & (per_run_df.Algorithm == "spea2")].reset_index(drop=True)
            comp_key = "nsga2" if comp_label == "NSGA-II" else "moead"
            comp = per_run_df[(per_run_df.Scenario == scenario) & (per_run_df.Algorithm == comp_key)].reset_index(drop=True)
            for i in range(len(spea2)):
                per_run_rows.append({
                    "Scenario": scenario,
                    "Competitor": comp_label,
                    "Run": i + 1,
                    "HV_SPEA2": spea2.loc[i, "HV"],
                    "HV_Comp": comp.loc[i, "HV"],
                    "IGD+_SPEA2": spea2.loc[i, "IGD+"],
                    "IGD+_Comp": comp.loc[i, "IGD+"],
                    "Spacing_SPEA2": spea2.loc[i, "Spacing"],
                    "Spacing_Comp": comp.loc[i, "Spacing"],
                })
    per_run_sheet = pd.DataFrame(per_run_rows)

    # Build Wilcoxon sheets per metric
    metric_sheets = {}
    for metric in ["HV", "IGD+", "Spacing"]:
        dfs = []
        for base_algo, comp_algo, comp_label in [("spea2", "nsga2", "NSGA-II"), ("spea2", "moead", "MOEA/D")]:
            df_w = wilcoxon_analysis(per_run_df, metric, base_algo, comp_algo, comp_label)
            dfs.append(df_w)
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="Scenario")
        metric_sheets[metric] = merged

    # Descriptive stats sheet
    desc_rows = []
    for metric in ["HV", "IGD+", "Spacing"]:
        for scenario in SCENARIOS:
            for comp_key, comp_label in [("nsga2", "NSGA-II"), ("moead", "MOEA/D")]:
                comparison = f"SPEA2_vs_{comp_label}"
                for algo_key, algo_label in [("spea2", "SPEA2"), (comp_key, comp_label)]:
                    vals = per_run_df[(per_run_df.Scenario == scenario) & (per_run_df.Algorithm == algo_key)][metric].values
                    desc_rows.append({
                        "Metric": metric,
                        "Scenario": scenario,
                        "Comparison": comparison,
                        "Algorithm": algo_label,
                        "Mean": round(np.mean(vals), 6),
                        "Std": round(np.std(vals), 6),
                        "Median": round(np.median(vals), 6),
                        "Min": round(np.min(vals), 6),
                        "Max": round(np.max(vals), 6),
                    })
    desc_sheet = pd.DataFrame(desc_rows)

    # Write to Excel
    print(f"Writing results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        metric_sheets["HV"].to_excel(writer, sheet_name="HV", index=False)
        metric_sheets["IGD+"].to_excel(writer, sheet_name="IGD+", index=False)
        metric_sheets["Spacing"].to_excel(writer, sheet_name="Spacing", index=False)
        per_run_sheet.to_excel(writer, sheet_name="Per_run", index=False)
        desc_sheet.to_excel(writer, sheet_name="Descriptive", index=False)

    print("Done!")


if __name__ == "__main__":
    main()