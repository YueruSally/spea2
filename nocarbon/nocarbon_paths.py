#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
from collections import Counter

class TeeStream:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  
        self.file.write(message)      

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()
# ==========================================

def main():
    NOCARBON_JSON = "nocarbon_pareto_points.json"

    with open(NOCARBON_JSON, "r", encoding="utf-8") as f:
        nc_all = json.load(f)

    nc_feasible = [s for s in nc_all if s.get("feasible", False)]
    if not nc_feasible:
        nc_feasible = nc_all

    print("=" * 80)
    print(f"  NoCarbon Pareto Solutions: {len(nc_feasible)} feasible")
    print("=" * 80)

    # ── 全局统计 ──
    all_mode_km = Counter()   # mode → total km across all solutions
    all_mode_count = Counter() # mode → number of arc segments

    for si, sol in enumerate(nc_feasible):
        nc_cost = sol["objectives"]["cost"]
        nc_time = sol["objectives"]["time_h"]

        print(f"\n{'━'*80}")
        print(f"  Solution {si:03d}  |  Cost=${nc_cost/1e6:.3f}M  Time={nc_time:.0f}h")
        print(f"{'━'*80}")

        # 统计该解的模式使用
        sol_mode_count = Counter()
        sol_mode_km_approx = Counter()

        for ai, alloc in enumerate(sol.get("allocations", [])):
            origin = alloc.get("origin", "?")
            dest   = alloc.get("destination", "?")
            paths  = alloc.get("paths", [])

            if not paths:
                continue

            print(f"\n  Batch {ai:02d}: {origin} → {dest}")

            for pi, p in enumerate(paths):
                nodes = p["nodes"]
                modes = p["modes"]
                share = p["share"]

                # 构建路径链描述
                chain_parts = []
                for i, (node, mode) in enumerate(zip(nodes[:-1], modes)):
                    chain_parts.append(f"{node} ─({mode})─▶ ")
                chain_str = "".join(chain_parts) + nodes[-1]

                # 统计模式
                mode_counter = Counter(modes)
                for m in modes:
                    sol_mode_count[m] += 1
                    all_mode_count[m] += 1

                # 模式统计字符串
                mode_str = ", ".join(f"{m}×{c}" for m, c in sorted(mode_counter.items()))

                print(f"    Path {pi} (share={share:.0%}): [{mode_str}]")
                print(f"      {chain_str}")
                print(f"      Hops: {len(modes)}  Nodes: {len(nodes)}")

        # 该解的模式汇总
        total_segs = sum(sol_mode_count.values())
        print(f"\n  ── Solution {si} Mode Summary ──")
        for m in ["rail", "road", "water"]:
            c = sol_mode_count.get(m, 0)
            pct = c / max(total_segs, 1) * 100
            print(f"    {m:6s}: {c:3d} segments ({pct:.1f}%)")

    # ── 全局模式汇总 ──
    print(f"\n\n{'='*80}")
    print(f"  GLOBAL MODE SUMMARY (across all {len(nc_feasible)} solutions)")
    print(f"{'='*80}")
    total_all = sum(all_mode_count.values())
    for m in ["rail", "road", "water"]:
        c = all_mode_count.get(m, 0)
        pct = c / max(total_all, 1) * 100
        print(f"  {m:6s}: {c:5d} segments ({pct:.1f}%)")

    print(f"\n\n{'='*80}")
    print(f"  FAST vs SLOW Solutions Comparison")
    print(f"{'='*80}")

    # 按时间排序
    sorted_sols = sorted(enumerate(nc_feasible),
                         key=lambda x: x[1]["objectives"]["time_h"])

    fast_3 = sorted_sols[:3]   
    slow_3 = sorted_sols[-3:]  

    for label, group in [("FASTEST 3", fast_3), ("SLOWEST 3", slow_3)]:
        print(f"\n  ── {label} ──")
        for si, sol in group:
            modes_all = []
            for alloc in sol.get("allocations", []):
                for p in alloc.get("paths", []):
                    modes_all.extend(p["modes"])
            mc = Counter(modes_all)
            total = sum(mc.values())
            mode_pcts = {m: mc.get(m, 0)/max(total, 1)*100 for m in ["rail","road","water"]}
            print(f"    Sol {si:03d}: time={sol['objectives']['time_h']:.0f}h  "
                  f"cost=${sol['objectives']['cost']/1e6:.3f}M  |  "
                  f"rail={mode_pcts['rail']:.0f}%  road={mode_pcts['road']:.0f}%  "
                  f"water={mode_pcts['water']:.0f}%  "
                  f"(total {total} segments)")

    # ── 每个解的路径模式分布表 ──
    print(f"\n\n{'='*80}")
    print(f"  Per-Solution Mode Distribution Table")
    print(f"{'='*80}")
    print(f"  {'Sol':>4s}  {'Cost($M)':>9s}  {'Time(h)':>8s}  "
          f"{'Rail%':>6s}  {'Road%':>6s}  {'Water%':>7s}  {'Segs':>5s}")
    print(f"  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}")

    for si, sol in enumerate(nc_feasible):
        modes_all = []
        for alloc in sol.get("allocations", []):
            for p in alloc.get("paths", []):
                modes_all.extend(p["modes"])
        mc = Counter(modes_all)
        total = sum(mc.values())
        if total == 0:
            continue
        print(f"  {si:4d}  {sol['objectives']['cost']/1e6:9.3f}  "
              f"{sol['objectives']['time_h']:8.0f}  "
              f"{mc.get('rail',0)/total*100:5.1f}%  "
              f"{mc.get('road',0)/total*100:5.1f}%  "
              f"{mc.get('water',0)/total*100:6.1f}%  "
              f"{total:5d}")

    print(f"\n\n{'='*80}")
    print(f"  Key Question: Do faster solutions use more ROAD?")
    print(f"{'='*80}")

    times = []
    road_pcts = []
    for si, sol in enumerate(nc_feasible):
        modes_all = []
        for alloc in sol.get("allocations", []):
            for p in alloc.get("paths", []):
                modes_all.extend(p["modes"])
        mc = Counter(modes_all)
        total = sum(mc.values())
        if total == 0: continue
        times.append(sol["objectives"]["time_h"])
        road_pcts.append(mc.get("road", 0) / total * 100)

    if len(times) >= 2:
        corr = np.corrcoef(times, road_pcts)[0, 1]
        print(f"  Correlation(time, road%): {corr:.3f}")
        if corr < -0.3:
            print(f"  → YES: faster solutions tend to use MORE road (negative correlation)")
        elif corr > 0.3:
            print(f"  → NO: faster solutions use LESS road (positive correlation)")
        else:
            print(f"  → WEAK relationship between time and road usage")

    print(f"\n[DONE]")


if __name__ == "__main__":
    OUTPUT_FILE = "analysis_results.txt"
    
    tee = TeeStream(OUTPUT_FILE)
    sys.stdout = tee
    
    try:
        main()
    finally:
        sys.stdout = tee.terminal
        tee.close()