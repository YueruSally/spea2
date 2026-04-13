#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPEA2 — NoCarbon 2-Objective (Cost & Time)  30 runs
  pop=125  gen=400  pc=0.9  pm=0.1  archive=125
  Fixed equal-weight mutation (0.25 each, no adaptive roulette)
  Parallel runs

修改清单（相对原版）：
  ① mutate_roulette_adaptive → mutate_fixed (固定0.25)
  ② feasibility_boost 中 mutate_roulette_adaptive → mutate_fixed
  ③ 删除 roulette 对象及所有传递，删除 mut_tracker adaptive prob 记录
  ④ HV_SAMPLES 改为 50000（与Baseline一致）
  ⑤ plot_mutation_adaptive_prob 标题改为 Fixed Equal-Weight
  ⑥ Step5 双重evaluate 与 Baseline Doc9 行为一致（两者均保留末尾evaluate）

修改清单（解数量修复）：
  ⑦ unique_individuals_by_objectives tol: 1e-3 → 1e-6（2D避免过度合并）
  ⑧ select_topk_by_cost_time 增加加权组合排序维度（补偿emission排序维度）
  ⑨ feasibility_boost 与 Baseline 一致，去掉多余 evaluate
"""

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════
# Global settings
# ════════════════════════════════════════════════════════

TIME_BUCKET_H = 24.0

CHINA_REGIONS = {"CN"}
EUROPE_REGIONS = {"EE", "WE"}
TRANSIT_REGIONS = {"KZ", "KG", "UZ", "RU", "BY"}

CORRIDOR_ORDER: Dict[str, int] = {"CN": 0, "CA": 1, "RU": 2, "EE": 3, "WE": 4}

CHINA_BORDER_NODES: set = {"Erenhot", "Manzhouli", "Khorgos", "Lianyungang",
                            "Chongqing", "Yiwu"}

NODE_GROUP: Dict[str, str] = {}

HARD_TIME_WINDOW = False

PEN_MISS_TT = 5e7
PEN_MISS_ALLOC = 1e9
PEN_CAP_EXCESS_PER_TEU = 5e7

WAITING_COST_PER_TEU_HOUR_DEFAULT = 0.8

# [①] Fixed equal-weight mutation — no adaptive roulette
W_ADD = 0.25
W_DEL = 0.25
W_MOD = 0.25
W_MODE = 0.25
OPS = ["add", "del", "mod", "mode"]

_FIXED_OP_WEIGHTS = [W_ADD, W_DEL, W_MOD, W_MODE]
_FIXED_OP_TOTAL   = sum(_FIXED_OP_WEIGHTS)
_FIXED_OP_PROBS   = [w / _FIXED_OP_TOTAL for w in _FIXED_OP_WEIGHTS]

CROSSOVER_RATE = 0.9
MUTATION_RATE  = 0.1

PATHS_TOPK_PER_CRITERION = 20
PATH_LIB_CAP_TOTAL       = 60    # 与Baseline一致
DFS_MAX_PATHS_PER_OD     = 200   # 与Baseline一致

CROSSOVER_SEGMENT_PROB = 0.50

MIN_FEASIBLE_SOLUTIONS       = 10
FEASIBLE_BOOST_ROUNDS        = 20
FEASIBLE_BOOST_MUTATION_RATE = 0.60
FEASIBLE_BOOST_TOPK_PARENTS  = 10

HV_EVERY     = 5
HV_SAMPLES   = 50000
METRIC_EVERY = 5

PSTAR_TAIL_GENS   = 30
PSTAR_CAP_PER_GEN = 40
PSTAR_MAX_TOTAL   = 50000

HV_REF_NORM = (1.2, 1.2)
HV_MC_SEED  = 12345

DEFAULT_PENALTY_PER_TEU_H = 65.0
NUM_OBJ = 2  # Cost, Time


# ════════════════════════════════════════════════════════
# Corridor helpers
# ════════════════════════════════════════════════════════

def china_border_monotone_ok(nodes, node_region):
    passed_border = False
    left_china = False
    start_is_border = (len(nodes) > 0 and nodes[0] in CHINA_BORDER_NODES)
    for i, n in enumerate(nodes):
        r = str(node_region.get(n, "")).strip()
        in_china  = (r in CHINA_REGIONS)
        is_border = (n in CHINA_BORDER_NODES)
        if left_china and in_china:
            return False
        if in_china:
            if passed_border:
                return False
            if is_border and not (i == 0 and start_is_border):
                passed_border = True
        else:
            left_china = True
    return True


def region_monotone_ok(nodes, node_region):
    max_level = -1
    for n in nodes:
        grp   = NODE_GROUP.get(n, "")
        level = CORRIDOR_ORDER.get(grp, -1)
        if level < 0:
            continue
        if level < max_level:
            return False
        max_level = level
    return True


# ════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════

def normalize_mode(mode_raw):
    m = str(mode_raw).strip().lower()
    if m in {"railway", "rail"}:  return "rail"
    if m in {"road", "truck"}:    return "road"
    if m in {"water", "ship", "sea"}: return "water"
    return m


def safe_float(x, default=0.0):
    try:
        if pd.isna(x): return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def parse_distance_km(x):
    s = str(x)
    cleaned = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    return float(cleaned) if cleaned else 0.0


def norm_region(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", ""}: return ""
    sl = s.lower()
    if sl in {"china", "prc", "chn"}:              return "CN"
    if sl in {"we", "west europe", "western europe"}: return "WE"
    if sl in {"ee", "east europe", "eastern europe"}: return "EE"
    return s.upper()


def unique_objective_tuples(objs, tol=1e-9):
    out = []
    for o in objs:
        dup = any(all(abs(o[i] - p[i]) <= tol for i in range(NUM_OBJ)) for p in out)
        if not dup:
            out.append(o)
    return out


def _is_bad_text_token(s):
    if s is None: return True
    t = str(s).strip()
    return t == "" or t.startswith("...")


def _ffill_nan(arr):
    x = np.array(arr, dtype=float).copy()
    if x.size == 0: return x
    finite_idx = np.where(np.isfinite(x))[0]
    if finite_idx.size == 0: return x
    x[~np.isfinite(x)] = np.nan
    first = finite_idx[0]
    if first > 0: x[:first] = x[first]
    for i in range(1, len(x)):
        if np.isnan(x[i]) and np.isfinite(x[i-1]):
            x[i] = x[i-1]
    return x


def _finite_points_array(pts):
    if not pts:
        return np.empty((0, NUM_OBJ), dtype=float)
    arr = np.array(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != NUM_OBJ:
        return np.empty((0, NUM_OBJ), dtype=float)
    return arr[np.all(np.isfinite(arr), axis=1)]


# ════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float
    cost_per_teu_km: float
    speed_kmh: float
    from_region: str = ""
    to_region: str = ""


@dataclass
class TimetableEntry:
    from_node: str
    to_node: str
    mode: str
    frequency_per_week: float
    first_departure_hour: float
    headway_hours: float


@dataclass
class Batch:
    batch_id: int
    origin: str
    destination: str
    quantity: float
    ET: float
    LT: float
    penalty_per_teu_h: float = DEFAULT_PENALTY_PER_TEU_H


@dataclass
class Path:
    path_id: int
    origin: str
    destination: str
    nodes: List[str]
    modes: List[str]
    arcs: List[Arc]
    base_cost_per_teu: float
    base_travel_time_h: float

    def __eq__(self, other):
        if not isinstance(other, Path): return NotImplemented
        return self.nodes == other.nodes and self.modes == other.modes

    def __hash__(self):
        return hash((tuple(self.nodes), tuple(self.modes)))


@dataclass
class PathAllocation:
    path: Path
    share: float

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            chain += f"{node}--({self.path.modes[i]})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


@dataclass(eq=False)
class Individual:
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    objectives: Tuple[float, float] = (float("inf"), float("inf"))
    penalty: float = 0.0
    feasible: bool = False
    feasible_hard: bool = False
    vio_breakdown: Dict[str, float] = field(default_factory=dict)
    spea2_fitness: float = float("inf")


# ════════════════════════════════════════════════════════
# Merge & normalise
# ════════════════════════════════════════════════════════

def merge_and_normalize(allocs):
    if not allocs: return []
    merged = {}
    for a in allocs:
        merged[a.path] = merged.get(a.path, 0.0) + float(a.share)
    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged.items()]
    total = sum(a.share for a in unique_allocs)
    if total <= 1e-12:
        avg = 1.0 / max(1, len(unique_allocs))
        for a in unique_allocs: a.share = avg
    else:
        for a in unique_allocs: a.share /= total
    filtered = [a for a in unique_allocs if a.share > 0.05]
    if not filtered:
        best = max(unique_allocs, key=lambda a: a.share)
        best.share = 1.0
        return [best]
    total2 = sum(a.share for a in filtered)
    if abs(total2 - 1.0) > 1e-9:
        for a in filtered: a.share /= total2
    return filtered


# ════════════════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════════════════

def load_border_delay_map(xls):
    out = {}
    if "Node_Border" not in xls.sheet_names: return out
    try:
        nb_df = pd.read_excel(xls, "Node_Border")
        node_col  = next((c for c in ["EnglishName","NodeEN"] if c in nb_df.columns), None)
        mode_col  = next((c for c in ["Mode","mode"] if c in nb_df.columns), None)
        delay_col = next((c for c in ["BorderDelay_h","Delay_h","BD"] if c in nb_df.columns), None)
        if node_col and mode_col and delay_col:
            for _, row in nb_df.iterrows():
                n = str(row.get(node_col,"")).strip()
                m = normalize_mode(row.get(mode_col,""))
                if n and m:
                    out[(n, m)] = safe_float(row.get(delay_col), default=0.0)
        print(f"[INFO] Loaded border delay map: {len(out)} entries.")
    except Exception as e:
        print(f"[WARN] Failed to load border delay ({e}).")
    return out


def load_mode_speeds(xls):
    out = {}
    if "Mode_Speeds" not in xls.sheet_names: return out
    try:
        df = pd.read_excel(xls, "Mode_Speeds")
        mc  = next((c for c in ["Mode","mode"] if c in df.columns), None)
        spc = next((c for c in ["Speed_kmh","speed_kmh","Speed"] if c in df.columns), None)
        if mc and spc:
            for _, row in df.iterrows():
                m = normalize_mode(row.get(mc,""))
                if m: out[m] = safe_float(row.get(spc), default=0.0)
        print(f"[INFO] Loaded mode speeds: {out}")
    except Exception as e:
        print(f"[WARN] Failed to read Mode_Speeds ({e}).")
    return out


def load_transshipment_map(xls):
    out = {}
    if "Transshipment" not in xls.sheet_names: return out
    try:
        df   = pd.read_excel(xls, "Transshipment")
        ndc  = next((c for c in ["Node","NodeEN","EnglishName"] if c in df.columns), None)
        imc  = next((c for c in ["InMode","FromMode","mode_in"] if c in df.columns), None)
        omc  = next((c for c in ["OutMode","ToMode","mode_out"] if c in df.columns), None)
        cstc = next((c for c in ["TransCost","Cost","trans_cost","Cost_per_TEU"] if c in df.columns), None)
        tmc  = next((c for c in ["TransTime_h","Time_h","trans_time_h","Time"] if c in df.columns), None)
        if ndc and imc and omc:
            for _, row in df.iterrows():
                node     = str(row.get(ndc,"")).strip()
                in_mode  = normalize_mode(row.get(imc,""))
                out_mode = normalize_mode(row.get(omc,""))
                if node and in_mode and out_mode:
                    out[(node, in_mode, out_mode)] = {
                        "cost_per_teu": safe_float(row.get(cstc), default=0.0) if cstc else 0.0,
                        "time_h":       safe_float(row.get(tmc),  default=0.0) if tmc  else 0.0,
                    }
        print(f"[INFO] Loaded transshipment entries: {len(out)}")
    except Exception as e:
        print(f"[WARN] Failed to read Transshipment ({e}).")
    return out


def load_waiting_params(xls):
    wc = WAITING_COST_PER_TEU_HOUR_DEFAULT
    if "Waiting_Costs" not in xls.sheet_names: return wc
    try:
        df = pd.read_excel(xls, "Waiting_Costs")
        for c in ["WaitingCost_per_TEU_h","WaitCost_per_TEU_h"]:
            if c in df.columns:
                vals = df[c].dropna().tolist()
                if vals: wc = safe_float(vals[0], default=wc); break
        print(f"[INFO] Loaded waiting cost: {wc}")
    except Exception as e:
        print(f"[WARN] Failed to read Waiting_Costs ({e}).")
    return wc


def load_network_from_extended(filename):
    global CHINA_BORDER_NODES, NODE_GROUP
    xls = pd.ExcelFile(filename)

    mode_speeds_map  = load_mode_speeds(xls)
    trans_map        = load_transshipment_map(xls)
    border_delay_map = load_border_delay_map(xls)

    nodes_df   = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).str.strip().tolist()

    node_region = {
        str(name).strip(): norm_region(reg)
        for name, reg in zip(nodes_df["EnglishName"], nodes_df["Region"])
    }

    if "RegionGroup" in nodes_df.columns:
        NODE_GROUP = {
            str(name).strip(): str(grp).strip()
            for name, grp in zip(nodes_df["EnglishName"], nodes_df["RegionGroup"])
            if str(grp).strip() not in ("", "nan", "None")
        }

    if "Node_Border" in xls.sheet_names:
        try:
            nb_df = pd.read_excel(xls, "Node_Border")
            loaded_borders = set()
            for _, row in nb_df.iterrows():
                region_val = str(row.get("Region","")).strip()
                is_border  = safe_float(row.get("IsBorderNode", 0), default=0.0) == 1.0
                if region_val == "CN" and is_border:
                    node_name = str(row.get("EnglishName","")).strip()
                    if node_name: loaded_borders.add(node_name)
            if loaded_borders:
                CHINA_BORDER_NODES = loaded_borders
                print(f"[INFO] Loaded China border nodes ({len(CHINA_BORDER_NODES)}): {sorted(CHINA_BORDER_NODES)}")
        except Exception as e:
            print(f"[WARN] Failed to load Node_Border: {e}. Using defaults.")

    CHINA_BORDER_NODES.update({"Ningbo","Shanghai"})

    node_hold_cost = {}
    node_proc_cost = {}
    for _, row in nodes_df.iterrows():
        n = str(row.get("EnglishName","")).strip()
        node_hold_cost[n] = safe_float(row.get("HoldCost_per_TEU_h"), default=WAITING_COST_PER_TEU_HOUR_DEFAULT)
        node_proc_cost[n] = safe_float(row.get("ProcCost_per_TEU_h"), default=0.0)

    SEAPORT_NODES = {"Ningbo","Shanghai"}
    for n in SEAPORT_NODES:
        if n not in node_region:
            node_region[n] = "CN"; NODE_GROUP[n] = "CN"
            if n not in node_names: node_names.append(n)

    waiting_cost_per_teu_h = load_waiting_params(xls)

    arcs_df   = pd.read_excel(xls, "Arcs_All")
    arcs      = []
    cost_cols = ["Cost_$_per_km","Cost_per_km","Cost"]

    for _, row in arcs_df.iterrows():
        mode  = normalize_mode(row.get("Mode","road"))
        speed = {"road": 75.0, "water": 30.0}.get(mode, 50.0)
        if mode in mode_speeds_map and mode_speeds_map[mode] > 0:
            speed = mode_speeds_map[mode]

        origin = str(row.get("OriginEN","")).strip()
        dest   = str(row.get("DestEN","")).strip()
        if _is_bad_text_token(origin) or _is_bad_text_token(dest): continue

        from_region = str(node_region.get(origin,"")).strip()
        to_region   = str(node_region.get(dest,"")).strip()
        distance    = parse_distance_km(row.get("Distance_km", 0.0))

        if "Capacity_TEUday" in arcs_df.columns and not pd.isna(row.get("Capacity_TEUday", np.nan)):
            capacity = safe_float(row.get("Capacity_TEUday"), default=1e9)
        elif "Capacity_TEUh" in arcs_df.columns and not pd.isna(row.get("Capacity_TEUh", np.nan)):
            capacity = safe_float(row.get("Capacity_TEUh"), default=1e9) * 24.0
        else:
            capacity = 1e9

        cpkm = 0.0
        for c in cost_cols:
            if c in arcs_df.columns:
                val = safe_float(row.get(c), default=None)
                if val is not None and val > 0: cpkm = val; break
        if cpkm <= 1e-9: cpkm = 0.5

        arcs.append(Arc(
            from_node=origin, to_node=dest, mode=mode,
            distance=distance, capacity=capacity,
            cost_per_teu_km=cpkm,
            speed_kmh=speed, from_region=from_region, to_region=to_region
        ))

    tdf        = pd.read_excel(xls, "Timetable")
    timetables = []
    for _, row in tdf.iterrows():
        origin    = str(row.get("OriginEN","")).strip()
        dest      = str(row.get("DestEN","")).strip()
        mode_norm = normalize_mode(row.get("Mode",""))
        if _is_bad_text_token(origin) or _is_bad_text_token(dest): continue
        if mode_norm not in {"road","rail","water"}: continue
        freq   = safe_float(row.get("Frequency_per_week"), default=1.0)
        hd_raw = row.get("Headway_Hours", np.nan)
        hd     = 168.0 / max(freq, 1.0) if pd.isna(hd_raw) else safe_float(hd_raw, default=168.0)
        v      = row.get("FirstDepartureHour", np.nan)
        fd     = 0.0
        if not pd.isna(v):
            try:
                s  = str(v).strip()
                fd = float(s.split(":")[0]) if ":" in s else float(s)
            except Exception: fd = 0.0
        timetables.append(TimetableEntry(
            from_node=origin, to_node=dest, mode=mode_norm,
            frequency_per_week=freq, first_departure_hour=fd, headway_hours=hd
        ))

    bdf     = pd.read_excel(xls, "Batches")
    bdf     = augment_batches_to_20(bdf, node_region=node_region, random_seed=2026)
    batches = []
    for _, row in bdf.iterrows():
        origin = str(row.get("OriginEN","")).strip()
        dest   = str(row.get("DestEN","")).strip()
        if node_region.get(origin) in CHINA_REGIONS and node_region.get(dest) in EUROPE_REGIONS:
            batches.append(Batch(
                batch_id=int(row.get("BatchID", 0)),
                origin=origin, destination=dest,
                quantity=safe_float(row.get("QuantityTEU"), default=0.0),
                ET=safe_float(row.get("ET"), default=0.0),
                LT=safe_float(row.get("LT"), default=0.0),
                penalty_per_teu_h=safe_float(row.get("PenaltyCost_per_TEU_h"),
                                             default=DEFAULT_PENALTY_PER_TEU_H)
            ))

    print(f"[INFO] Batches loaded: {len(batches)}")
    print(f"[INFO] 2-Objective SPEA2: Cost & Time (emission/carbon REMOVED)")
    return (
        node_names, node_region,
        node_hold_cost, node_proc_cost,
        arcs, timetables, batches,
        waiting_cost_per_teu_h,
        mode_speeds_map, trans_map, border_delay_map
    )


def build_graph(arcs):
    g = {}
    for a in arcs:
        g.setdefault(a.from_node, []).append((a.to_node, a))
    return g


def build_timetable_dict(timetables):
    tt_dict = {}
    for t in timetables:
        tt_dict.setdefault((t.from_node, t.to_node, t.mode), []).append(t)
    return tt_dict


def build_arc_lookup(arcs):
    mp = {}
    for a in arcs:
        k = (a.from_node, a.to_node, a.mode)
        if k not in mp: mp[k] = a
    return mp


# ════════════════════════════════════════════════════════
# Path library — [⑧] select_topk_by_cost_time 增加加权组合
# ════════════════════════════════════════════════════════

def random_dfs_paths(graph, origin, dest, node_region,
                     max_len=12, max_paths=200, timeout_sec=8.0):
    deadline = time.time() + timeout_sec
    paths, found_set = [], set()
    attempts, max_attempts = 0, max_paths * 20
    while len(paths) < max_paths and attempts < max_attempts:
        if time.time() > deadline: break
        attempts += 1
        node, cur_arcs, visited, cur_nodes, ok = origin, [], {origin}, [origin], True
        for _ in range(max_len):
            if node == dest: break
            neighbors = list(graph.get(node, []))
            if not neighbors: ok = False; break
            random.shuffle(neighbors)
            moved = False
            for nxt, arc in neighbors:
                if nxt in visited: continue
                new_nodes = cur_nodes + [nxt]
                if not china_border_monotone_ok(new_nodes, node_region): continue
                if not region_monotone_ok(new_nodes, node_region): continue
                cur_arcs.append(arc); visited.add(nxt); cur_nodes.append(nxt)
                node = nxt; moved = True; break
            if not moved: ok = False; break
        if ok and node == dest and cur_arcs:
            key = tuple(cur_nodes)
            if key not in found_set:
                found_set.add(key); paths.append(cur_arcs)
    return paths


def repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup):
    new_seq = []
    for arc in arc_seq:
        if arc.mode == "road": new_seq.append(arc); continue
        if tt_dict.get((arc.from_node, arc.to_node, arc.mode), []):
            new_seq.append(arc); continue
        k_road = (arc.from_node, arc.to_node, "road")
        if k_road in arc_lookup: new_seq.append(arc_lookup[k_road])
        else: return None
    return new_seq


def select_topk_by_cost_time(paths, k=30, cap_total=90):
    """[⑧] Baseline用cost/time/emission三维排序各取topk；
    NoCarbon去掉emission后，用归一化加权组合(0.5*cost+0.5*time)作为第三排序维度，
    补偿失去的emission多样性来源。"""
    if not paths: return []
    by_cost = sorted(paths, key=lambda p: p.base_cost_per_teu)
    by_time = sorted(paths, key=lambda p: p.base_travel_time_h)
    # 归一化后加权组合
    costs = [p.base_cost_per_teu for p in paths]
    times = [p.base_travel_time_h for p in paths]
    c_min, c_max = min(costs), max(costs)
    t_min, t_max = min(times), max(times)
    c_range = c_max - c_min if c_max - c_min > 1e-12 else 1.0
    t_range = t_max - t_min if t_max - t_min > 1e-12 else 1.0
    by_combo = sorted(paths, key=lambda p: (
        0.5 * (p.base_cost_per_teu - c_min) / c_range +
        0.5 * (p.base_travel_time_h - t_min) / t_range
    ))
    picked, used = [], set()
    for lst in [by_cost, by_time, by_combo]:
        for p in lst[:k]:
            if p not in used: picked.append(p); used.add(p)
    return picked[:cap_total] if cap_total else picked


def build_path_library(node_names, node_region, arcs, batches, tt_dict, arc_lookup):
    graph    = build_graph(arcs)
    path_lib = {}
    next_pid = 0
    for b in batches:
        od = (b.origin, b.destination)
        if od in path_lib: continue
        arc_paths = random_dfs_paths(graph, b.origin, b.destination,
                                     node_region=node_region, max_len=12,
                                     max_paths=DFS_MAX_PATHS_PER_OD)
        paths_od = []
        for arc_seq in arc_paths:
            repaired = repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup)
            if repaired is None: continue
            nodes = [repaired[0].from_node] + [a.to_node for a in repaired]
            if len(set(nodes)) != len(nodes): continue
            if not region_monotone_ok(nodes, node_region): continue
            if not china_border_monotone_ok(nodes, node_region): continue
            modes = [a.mode for a in repaired]
            paths_od.append(Path(
                path_id=next_pid, origin=b.origin, destination=b.destination,
                nodes=nodes, modes=modes, arcs=repaired,
                base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in repaired),
                base_travel_time_h=sum(a.distance / max(a.speed_kmh, 1.0) for a in repaired),
            ))
            next_pid += 1
        if paths_od:
            path_lib[od] = select_topk_by_cost_time(
                paths_od, k=PATHS_TOPK_PER_CRITERION, cap_total=PATH_LIB_CAP_TOTAL)
    removed = 0
    for od in list(path_lib.keys()):
        before = len(path_lib[od])
        path_lib[od] = [p for p in path_lib[od]
                        if china_border_monotone_ok(p.nodes, node_region)]
        removed += before - len(path_lib[od])
        if not path_lib[od]: del path_lib[od]
    if removed: print(f"[WARN] Post-filter removed {removed} paths.")
    else:       print("[INFO] All paths pass border monotonicity. ✅")
    return path_lib


def sanity_check_path_lib(batches, path_lib):
    missing = [(b.batch_id, (b.origin, b.destination))
               for b in batches if not path_lib.get((b.origin, b.destination), [])]
    if missing:
        for bid, od in missing[:20]:
            print(f"[SANITY] missing paths Batch {bid} OD={od}")
        raise RuntimeError("Path library missing some ODs.")
    print("[SANITY] ✅ All batches have paths.")


def repair_missing_allocations(ind, batches, path_lib):
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        if ind.od_allocations.get(key, []): continue
        paths = path_lib.get((b.origin, b.destination), [])
        if paths:
            ind.od_allocations[key] = [PathAllocation(path=paths[0], share=1.0)]


# ════════════════════════════════════════════════════════
# Simulation & evaluation (no emission)
# ════════════════════════════════════════════════════════

def next_departure_time_programB(t, entries):
    best_dep = float("inf")
    for e in entries:
        if t <= e.first_departure_hour:
            dep = e.first_departure_hour
        else:
            waited = t - e.first_departure_hour
            n      = math.ceil(waited / max(e.headway_hours, 1e-6))
            dep    = e.first_departure_hour + n * e.headway_hours
        if dep < best_dep: best_dep = dep
    return best_dep if best_dep < float("inf") else t


def simulate_path_time_capacity(path, batch, flow_teu, tt_dict, arc_flow_map,
                                 trans_map=None, border_delay_map=None):
    t               = float(batch.ET)
    miss_tt         = 0
    trans_map       = trans_map or {}
    border_delay_map= border_delay_map or {}
    prev_arc        = None
    node_wait_list  = []

    for arc in path.arcs:
        current_node   = arc.from_node
        arc_trans_wait = 0.0

        if prev_arc is not None and prev_arc.mode != arc.mode:
            rec = trans_map.get((current_node, prev_arc.mode, arc.mode))
            if rec:
                trans_h = safe_float(rec.get("time_h"), default=0.0)
                if trans_h > 0: t += trans_h; arc_trans_wait += trans_h

        if current_node in CHINA_BORDER_NODES:
            bd = border_delay_map.get((current_node, arc.mode), 0.0)
            if bd > 0: t += bd; arc_trans_wait += bd

        travel_time_arc = arc.distance / max(arc.speed_kmh, 1.0)
        entries = [] if arc.mode == "road" else \
                  tt_dict.get((current_node, arc.to_node, arc.mode), [])

        if arc.mode != "road" and not entries:
            miss_tt += 1; return float("inf"), [], miss_tt

        dep            = t if not entries else next_departure_time_programB(t, entries)
        arc_sched_wait = max(0.0, dep - t)
        node_wait_list.append((current_node, arc_sched_wait, arc_trans_wait))

        arr        = dep + travel_time_arc
        start_slot = int(dep // 24)
        arc_key    = (current_node, arc.to_node, arc.mode)
        arc_flow_map[(arc_key, start_slot)] = \
            arc_flow_map.get((arc_key, start_slot), 0.0) + flow_teu

        t        = arr
        prev_arc = arc

    return (t - batch.ET), node_wait_list, miss_tt


def evaluate_individual(ind, batches, arcs, tt_dict, waiting_cost_per_teu_h,
                         node_hold_cost=None, node_proc_cost=None,
                         trans_map=None, border_delay_map=None):
    node_hold_cost   = node_hold_cost   or {}
    node_proc_cost   = node_proc_cost   or {}
    trans_map        = trans_map        or {}
    border_delay_map = border_delay_map or {}

    total_cost = makespan = 0.0
    arc_flow_map = {}
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    miss_alloc = miss_tt = 0
    cap_excess = late_teu_h_total = wait_teu_h_total = 0.0
    trans_teu_h_total = trans_cost_total = 0.0

    for b in batches:
        key   = (b.origin, b.destination, b.batch_id)
        allocs= ind.od_allocations.get(key, [])
        if not allocs: miss_alloc += 1; continue

        batch_finish = b.ET
        for alloc in allocs:
            if alloc.share <= 1e-12: continue
            flow = alloc.share * b.quantity
            p    = alloc.path

            travel_time, node_wait_list, mtt = simulate_path_time_capacity(
                p, b, flow, tt_dict, arc_flow_map,
                trans_map=trans_map, border_delay_map=border_delay_map)
            if math.isinf(travel_time): miss_tt += mtt; continue

            total_cost += p.base_cost_per_teu * flow

            tc = 0.0
            for i in range(len(p.arcs) - 1):
                if p.arcs[i].mode != p.arcs[i+1].mode:
                    node = p.arcs[i+1].from_node
                    rec  = trans_map.get((node, p.arcs[i].mode, p.arcs[i+1].mode), {})
                    tc  += safe_float(rec.get("cost_per_teu"), default=0.0) * flow
            total_cost += tc; trans_cost_total += tc

            for (wait_node, sched_h, trans_h) in node_wait_list:
                hold_rate = node_hold_cost.get(wait_node, WAITING_COST_PER_TEU_HOUR_DEFAULT)
                proc_rate = node_proc_cost.get(wait_node, 0.0)
                if sched_h > 0.0:
                    total_cost       += hold_rate * flow * sched_h
                    wait_teu_h_total += flow * sched_h
                if trans_h > 0.0:
                    total_cost        += proc_rate * flow * trans_h
                    trans_teu_h_total += flow * trans_h

            arrival_time = b.ET + travel_time
            batch_finish = max(batch_finish, arrival_time)
            if arrival_time > b.LT:
                late_teu_h        = flow * (arrival_time - b.LT)
                late_teu_h_total += late_teu_h
                total_cost       += b.penalty_per_teu_h * late_teu_h

        makespan = max(makespan, batch_finish)

    for (arc_key, slot), sf in arc_flow_map.items():
        cap = arc_caps.get(arc_key, 1e9)
        if sf > cap: cap_excess += (sf - cap)

    penalty = (PEN_MISS_ALLOC * float(miss_alloc) +
               PEN_MISS_TT    * float(miss_tt)     +
               PEN_CAP_EXCESS_PER_TEU * float(cap_excess))

    ind.objectives    = (float(total_cost), float(makespan))
    ind.penalty       = float(penalty)
    hard_ok           = (miss_alloc == 0 and miss_tt == 0 and cap_excess <= 1e-9)
    ind.feasible_hard = bool(hard_ok)
    ind.feasible      = bool(hard_ok)
    ind.vio_breakdown = {
        "miss_alloc":   float(miss_alloc),  "miss_tt":      float(miss_tt),
        "cap_excess":   float(cap_excess),  "late_teu_h":   float(late_teu_h_total),
        "wait_teu_h":   float(wait_teu_h_total),
        "trans_teu_h":  float(trans_teu_h_total),
        "trans_cost":   float(trans_cost_total),
    }


# ════════════════════════════════════════════════════════
# GA operators
# ════════════════════════════════════════════════════════

def clone_gene(alloc):
    return PathAllocation(path=alloc.path, share=float(alloc.share))


def crossover_structural(ind1, ind2, batches):
    child1, child2 = Individual(), Individual()
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1  = ind1.od_allocations.get(key, [])
        g2  = ind2.od_allocations.get(key, [])
        if not g1 and not g2: continue
        if not g1:
            child1.od_allocations[key] = [clone_gene(x) for x in g2]
            child2.od_allocations[key] = [clone_gene(x) for x in g2]; continue
        if not g2:
            child1.od_allocations[key] = [clone_gene(x) for x in g1]
            child2.od_allocations[key] = [clone_gene(x) for x in g1]; continue
        cut1, cut2 = random.randint(0, len(g1)), random.randint(0, len(g2))
        c1 = [clone_gene(x) for x in g1[:cut1]] + [clone_gene(x) for x in g2[cut2:]]
        c2 = [clone_gene(x) for x in g2[:cut2]] + [clone_gene(x) for x in g1[cut1:]]
        child1.od_allocations[key] = merge_and_normalize(c1)
        child2.od_allocations[key] = merge_and_normalize(c2)
    return child1, child2


def path_from_arcs(new_arcs, origin, destination, path_id=-1, node_region=None):
    if not new_arcs: return None
    nodes = [new_arcs[0].from_node] + [a.to_node for a in new_arcs]
    if nodes[0] != origin or nodes[-1] != destination: return None
    if len(set(nodes)) != len(nodes): return None
    if node_region is not None:
        if not china_border_monotone_ok(nodes, node_region): return None
        if not region_monotone_ok(nodes, node_region):       return None
    return Path(
        path_id=path_id, origin=origin, destination=destination,
        nodes=nodes, modes=[a.mode for a in new_arcs], arcs=new_arcs,
        base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in new_arcs),
        base_travel_time_h=sum(a.distance / max(a.speed_kmh, 1.0) for a in new_arcs),
    )


def rebuild_path_from_nodes_modes(origin, destination, nodes, modes,
                                   tt_dict, arc_lookup, allow_road_fallback=True):
    if not nodes or len(nodes) < 2 or nodes[0] != origin or nodes[-1] != destination:
        return None
    if len(modes) != len(nodes) - 1 or len(set(nodes)) != len(nodes): return None
    new_arcs = []
    for i in range(len(modes)):
        u, v, m = nodes[i], nodes[i+1], modes[i]
        k = (u, v, m)
        if k not in arc_lookup: return None
        arc = arc_lookup[k]
        if arc.mode != "road" and not tt_dict.get((u, v, arc.mode), []):
            if allow_road_fallback and (u, v, "road") in arc_lookup:
                arc = arc_lookup[(u, v, "road")]
            else: return None
        new_arcs.append(arc)
    return path_from_arcs(new_arcs, origin, destination)


def find_common_internal_nodes(p1, p2):
    return list(set(p1.nodes[1:-1]) & set(p2.nodes[1:-1]))


def perform_single_point_crossover_paths(pA, pB, join_node, tt_dict, arc_lookup):
    if join_node not in pA.nodes or join_node not in pB.nodes: return None
    ia, ib = pA.nodes.index(join_node), pB.nodes.index(join_node)
    return rebuild_path_from_nodes_modes(
        pA.origin, pA.destination,
        pA.nodes[:ia+1] + pB.nodes[ib+1:],
        pA.modes[:ia]   + pB.modes[ib:],
        tt_dict, arc_lookup)


def crossover_common_node(ind1, ind2, batches, tt_dict, arc_lookup):
    child1, child2 = Individual(), Individual()
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1  = ind1.od_allocations.get(key, [])
        g2  = ind2.od_allocations.get(key, [])
        if not g1 and not g2: continue
        if not g1:
            child1.od_allocations[key] = [clone_gene(x) for x in g2]
            child2.od_allocations[key] = [clone_gene(x) for x in g2]; continue
        if not g2:
            child1.od_allocations[key] = [clone_gene(x) for x in g1]
            child2.od_allocations[key] = [clone_gene(x) for x in g1]; continue
        c1_allocs = [clone_gene(x) for x in g1]
        c2_allocs = [clone_gene(x) for x in g2]
        p1, p2    = random.choice(g1).path, random.choice(g2).path
        common    = find_common_internal_nodes(p1, p2)
        if common:
            join = random.choice(common)
            np1  = perform_single_point_crossover_paths(p1, p2, join, tt_dict, arc_lookup)
            np2  = perform_single_point_crossover_paths(p2, p1, join, tt_dict, arc_lookup)
            if np1: c1_allocs.append(PathAllocation(path=np1, share=0.20))
            if np2: c2_allocs.append(PathAllocation(path=np2, share=0.20))
        child1.od_allocations[key] = merge_and_normalize(c1_allocs)
        child2.od_allocations[key] = merge_and_normalize(c2_allocs)
    return child1, child2


def crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup):
    if random.random() < CROSSOVER_SEGMENT_PROB:
        c1, c2 = crossover_common_node(p1, p2, batches, tt_dict, arc_lookup)
        if c1.od_allocations or c2.od_allocations: return c1, c2
    return crossover_structural(p1, p2, batches)


def random_initial_individual(batches, path_lib, max_paths=3):
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths: continue
        k      = random.randint(1, min(max_paths, len(paths)))
        chosen = random.sample(paths, k)
        raw    = [PathAllocation(path=p, share=random.random()) for p in chosen]
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = merge_and_normalize(raw)
    return ind


def greedy_initial_individual(batches, path_lib):
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths: continue
        best  = min(paths, key=lambda p: p.base_travel_time_h)
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = \
            [PathAllocation(path=best, share=1.0)]
    return ind


def mutate_add(ind, batch, path_lib):
    key   = (batch.origin, batch.destination, batch.batch_id)
    allocs= ind.od_allocations.get(key, [])
    pool  = path_lib.get((batch.origin, batch.destination), [])
    if not pool: return False
    cur   = {a.path for a in allocs}
    cands = [p for p in pool if p not in cur]
    if not cands:
        if allocs:
            allocs[random.randrange(len(allocs))] = PathAllocation(
                path=random.choice(pool), share=0.2)
            ind.od_allocations[key] = merge_and_normalize(allocs)
            return True
        return False
    allocs.append(PathAllocation(path=random.choice(cands), share=0.2))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_del(ind, batch):
    key   = (batch.origin, batch.destination, batch.batch_id)
    allocs= ind.od_allocations.get(key, [])
    if len(allocs) <= 1: return False
    allocs.pop(random.randrange(len(allocs)))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mod(ind, batch):
    key   = (batch.origin, batch.destination, batch.batch_id)
    allocs= ind.od_allocations.get(key, [])
    if not allocs: return False
    random.choice(allocs).share *= random.uniform(0.5, 1.5)
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mode(ind, batch, tt_dict, arc_lookup, max_trials=20):
    key   = (batch.origin, batch.destination, batch.batch_id)
    allocs= ind.od_allocations.get(key, [])
    if not allocs: return False
    idx      = random.randrange(len(allocs))
    old_alloc= allocs[idx]
    p        = old_alloc.path
    if not p.arcs: return False
    arc_i    = random.randrange(len(p.arcs))
    old_arc  = p.arcs[arc_i]
    u, v     = old_arc.from_node, old_arc.to_node
    for _ in range(max_trials):
        new_mode = random.choice([m for m in ["road","rail","water"] if m != old_arc.mode])
        k_arc    = (u, v, new_mode)
        if k_arc not in arc_lookup: continue
        if new_mode != "road" and not tt_dict.get((u, v, new_mode), []): continue
        new_arcs    = list(p.arcs); new_arcs[arc_i] = arc_lookup[k_arc]
        new_path    = path_from_arcs(new_arcs, p.origin, p.destination)
        if new_path is None: continue
        allocs_new  = deepcopy(allocs)
        allocs_new[idx] = PathAllocation(path=new_path, share=old_alloc.share)
        ind.od_allocations[key] = merge_and_normalize(allocs_new)
        return True
    return False


def augment_batches_to_20(bdf, node_region, random_seed=2026):
    df = bdf.copy()
    required_cols = ["BatchID","OriginEN","DestEN","QuantityTEU","ET","LT"]
    if any(c not in df.columns for c in required_cols) or len(df) >= 20:
        return df
    china_nodes  = [n for n, r in node_region.items()
                    if r in CHINA_REGIONS and n not in CHINA_BORDER_NODES]
    europe_nodes = [n for n, r in node_region.items() if r in EUROPE_REGIONS]
    if not china_nodes or not europe_nodes: return df
    q_vals  = pd.to_numeric(df["QuantityTEU"], errors="coerce").dropna()
    q_min   = int(q_vals.min()) if len(q_vals) else 80
    q_max   = int(q_vals.max()) if len(q_vals) else 150
    lt_vals = pd.to_numeric(df["LT"], errors="coerce").dropna()
    lt_vals = lt_vals[lt_vals >= 300]
    lt_min, lt_max = (int(lt_vals.min()), int(lt_vals.max())) if len(lt_vals) else (360, 504)
    pen_col_exists = "PenaltyCost_per_TEU_h" in df.columns
    if pen_col_exists:
        pv = pd.to_numeric(df["PenaltyCost_per_TEU_h"], errors="coerce").dropna()
        pv = pv[(pv >= 1.0) & (pv <= 500.0)]
        pen_min, pen_max = (float(pv.min()), float(pv.max())) if len(pv) else (30.0, 100.0)
    else:
        pen_min, pen_max = 30.0, 100.0
    existing_ids = set(pd.to_numeric(df["BatchID"], errors="coerce").dropna().astype(int).tolist())
    next_id      = max(existing_ids) + 1 if existing_ids else 11
    rng          = np.random.default_rng(random_seed)
    new_rows     = []
    for i in range(20 - len(df)):
        new_rows.append({
            "BatchID": next_id + i, "OriginEN": str(rng.choice(china_nodes)),
            "DestEN": str(rng.choice(europe_nodes)),
            "QuantityTEU": int(rng.integers(q_min, q_max + 1)),
            "ET": 0, "LT": int(rng.integers(lt_min, lt_max + 1)),
            "PenaltyCost_per_TEU_h": round(float(rng.uniform(pen_min, pen_max)), 2),
        })
    df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"[INFO] Batches augmented: {len(df)} -> {len(df_out)}")
    return df_out


# ════════════════════════════════════════════════════════
# Fixed equal-weight operator selection
# ════════════════════════════════════════════════════════

def sample_operator() -> str:
    r, cum = random.random(), 0.0
    for op, prob in zip(OPS, _FIXED_OP_PROBS):
        cum += prob
        if r <= cum: return op
    return OPS[-1]


def apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup):
    if op == "add":  return mutate_add(ind, batch, path_lib)
    if op == "del":  return mutate_del(ind, batch)
    if op == "mod":  return mutate_mod(ind, batch)
    if op == "mode": return mutate_mode(ind, batch, tt_dict, arc_lookup)
    return False


def mutate_fixed(
    ind, batches, path_lib, tt_dict, arc_lookup,
    arcs, waiting_cost_per_teu_h,
    node_hold_cost=None, node_proc_cost=None,
    trans_map=None, border_delay_map=None,
):
    batch = random.choice(batches)
    op    = sample_operator()
    ok    = apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup)
    if not ok:
        return op, False
    repair_missing_allocations(ind, batches, path_lib)
    evaluate_individual(
        ind, batches, arcs, tt_dict, waiting_cost_per_teu_h,
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        trans_map=trans_map, border_delay_map=border_delay_map)
    return op, True


# ════════════════════════════════════════════════════════
# Dominance (2-objective)
# ════════════════════════════════════════════════════════

def dominates(a, b):
    if a.feasible and not b.feasible: return True
    if b.feasible and not a.feasible: return False
    if a.feasible and b.feasible:
        return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
                any(x <  y for x, y in zip(a.objectives, b.objectives)))
    if a.penalty < b.penalty - 1e-12: return True
    if b.penalty < a.penalty - 1e-12: return False
    return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
            any(x <  y for x, y in zip(a.objectives, b.objectives)))


def unique_individuals_by_objectives(front, tol=1e-6):
    """[⑦] tol从1e-3改为1e-6，2D下避免过度合并近似解。"""
    uniq, seen = [], []
    for ind in front:
        obj = ind.objectives
        if not any(all(abs(obj[i]-o[i]) <= tol for i in range(NUM_OBJ)) for o in seen):
            seen.append(obj); uniq.append(ind)
    return uniq


# ════════════════════════════════════════════════════════
# SPEA2 Core
# ════════════════════════════════════════════════════════

def compute_spea2_fitness(combined: List[Individual]) -> None:
    N = len(combined)
    if N == 0: return
    strength = np.zeros(N, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and dominates(combined[i], combined[j]):
                strength[i] += 1.0
    raw = np.zeros(N, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and dominates(combined[j], combined[i]):
                raw[i] += strength[j]
    k = max(1, int(math.sqrt(N)))
    obj_mat = np.zeros((N, NUM_OBJ), dtype=float)
    for i, ind in enumerate(combined):
        obj_mat[i] = (np.array(ind.objectives, dtype=float)
                      if ind.feasible
                      else np.array(ind.objectives, dtype=float) + ind.penalty)
    col_min   = obj_mat.min(axis=0)
    col_max   = obj_mat.max(axis=0)
    col_range = np.where(col_max - col_min > 1e-12, col_max - col_min, 1.0)
    obj_norm  = (obj_mat - col_min) / col_range
    density = np.zeros(N, dtype=float)
    for i in range(N):
        diffs   = obj_norm - obj_norm[i]
        dists   = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i]= np.inf
        sigma_k = np.sort(dists)[min(k-1, len(dists)-1)]
        density[i] = 1.0 / (sigma_k + 2.0)
    for i, ind in enumerate(combined):
        ind.spea2_fitness = float(raw[i] + density[i])


def spea2_environmental_selection(combined: List[Individual],
                                   archive_size: int) -> List[Individual]:
    compute_spea2_fitness(combined)
    non_dom  = [ind for ind in combined if ind.spea2_fitness < 1.0]
    dom_rest = sorted([ind for ind in combined if ind.spea2_fitness >= 1.0],
                      key=lambda x: x.spea2_fitness)
    if len(non_dom) < archive_size:
        archive = non_dom + dom_rest[:archive_size - len(non_dom)]
    elif len(non_dom) == archive_size:
        archive = non_dom
    else:
        archive = _truncate_by_distance(non_dom, archive_size)
    return archive[:archive_size]


def _truncate_by_distance(candidates: List[Individual],
                           target_size: int) -> List[Individual]:
    N       = len(candidates)
    obj_mat = np.array([ind.objectives for ind in candidates], dtype=float)
    col_min = obj_mat.min(axis=0); col_max = obj_mat.max(axis=0)
    col_range = np.where(col_max - col_min > 1e-12, col_max - col_min, 1.0)
    obj_norm  = (obj_mat - col_min) / col_range
    active = list(range(N))
    while len(active) > target_size:
        sub   = obj_norm[active]
        n_a   = len(active)
        dist_mat = np.full((n_a, n_a), np.inf)
        for ii in range(n_a):
            for jj in range(ii+1, n_a):
                d = float(np.sqrt(np.sum((sub[ii]-sub[jj])**2)))
                dist_mat[ii, jj] = dist_mat[jj, ii] = d
        sorted_rows = np.sort(dist_mat, axis=1)
        remove_idx  = 0
        for idx in range(1, n_a):
            for col in range(n_a):
                if sorted_rows[idx, col] < sorted_rows[remove_idx, col] - 1e-15:
                    remove_idx = idx; break
                elif sorted_rows[idx, col] > sorted_rows[remove_idx, col] + 1e-15:
                    break
        active.pop(remove_idx)
    return [candidates[i] for i in active]


def spea2_binary_tournament(archive: List[Individual]) -> Individual:
    a, b = random.sample(archive, 2)
    return a if a.spea2_fitness <= b.spea2_fitness else b


# ════════════════════════════════════════════════════════
# [⑨] Feasibility Boost — 与Baseline一致
# ════════════════════════════════════════════════════════

def _select_boost_parents(archive, topk=FEASIBLE_BOOST_TOPK_PARENTS):
    feasible = sorted([i for i in archive if i.feasible],
                      key=lambda x: x.spea2_fitness)
    if len(feasible) >= topk: return feasible[:topk]
    infeasible = sorted([i for i in archive if not i.feasible],
                        key=lambda x: x.penalty)
    return (feasible + infeasible)[:topk]


def feasibility_boost(archive, batches, path_lib, tt_dict, arc_lookup,
                       arcs, waiting_cost_per_teu_h, eval_kwargs,
                       boost_rounds=FEASIBLE_BOOST_ROUNDS,
                       boost_mutation_rate=FEASIBLE_BOOST_MUTATION_RATE,
                       topk_parents=FEASIBLE_BOOST_TOPK_PARENTS):
    """[⑨] 与Baseline完全一致：
    - mutate_fixed内部在ok=True时evaluate
    - 不mutate或mutate失败时child未被evaluate
    - 没有额外的evaluate调用"""
    parents  = _select_boost_parents(archive, topk=topk_parents)
    new_inds = []
    if len(parents) < 2:
        for _ in range(boost_rounds):
            ind = greedy_initial_individual(batches, path_lib)
            repair_missing_allocations(ind, batches, path_lib)
            evaluate_individual(ind, batches, arcs, tt_dict,
                                waiting_cost_per_teu_h, **eval_kwargs)
            new_inds.append(ind)
    else:
        for _ in range(boost_rounds):
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup)
            for child in (c1, c2):
                repair_missing_allocations(child, batches, path_lib)
                if random.random() < boost_mutation_rate:
                    mutate_fixed(
                        child, batches, path_lib, tt_dict, arc_lookup,
                        arcs, waiting_cost_per_teu_h, **eval_kwargs)
                new_inds.append(child)
            if len(new_inds) >= boost_rounds: break

    arch_id_idx  = {id(ind): idx for idx, ind in enumerate(archive)}
    worst_sorted = sorted([i for i in archive if not i.feasible],
                          key=lambda x: x.penalty, reverse=True)
    new_sorted   = sorted(new_inds, key=lambda x: (0 if x.feasible else 1, x.penalty))
    num_new_feas = 0
    replaced     = set()
    for new_ind in new_sorted:
        for target in worst_sorted:
            if id(target) in replaced: continue
            if new_ind.feasible or new_ind.penalty < target.penalty:
                idx = arch_id_idx.get(id(target))
                if idx is not None:
                    archive[idx] = new_ind
                    replaced.add(id(target))
                    if new_ind.feasible: num_new_feas += 1
                break
    return archive, num_new_feas


# ════════════════════════════════════════════════════════
# HV (exact 2-D sweep)
# ════════════════════════════════════════════════════════

class HypervolumeCalculator2D:
    def __init__(self, ref_point):
        self.ref = np.array(ref_point, dtype=float)

    def calculate_points(self, points):
        if not points: return 0.0
        pts = np.array(points, dtype=float)
        pts = pts[np.all(pts < self.ref, axis=1)]
        if len(pts) == 0: return 0.0
        pts  = pts[pts[:, 0].argsort()]
        hv   = 0.0; prev_x = 0.0; min_y = self.ref[1]
        for x, y in pts:
            if y < min_y:
                hv    += (x - prev_x) * (self.ref[1] - min_y)
                min_y  = y; prev_x = x
        hv += (self.ref[0] - prev_x) * (self.ref[1] - min_y)
        return float(hv)


# ════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════

def dominates_obj(a, b):
    return (all(a[i] <= b[i] for i in range(NUM_OBJ)) and
            any(a[i] < b[i]  for i in range(NUM_OBJ)))


def nondominated_set(points):
    pts = unique_objective_tuples(points, tol=1e-9)
    return [p for i, p in enumerate(pts)
            if not any(dominates_obj(q, p) for j, q in enumerate(pts) if i != j)]


def normalize_points(points, mins, maxs):
    out = []
    for p in points:
        pp = []
        for i in range(NUM_OBJ):
            rng = maxs[i] - mins[i]
            pp.append(0.0 if rng <= 1e-12 else (p[i]-mins[i])/rng)
        out.append(tuple(pp))
    return out


def clip_points(points, ref):
    return [tuple(min(max(p[i], 0.0), ref[i]) for i in range(NUM_OBJ)) for p in points]


def igd_plus(P_star, A):
    if not P_star or not A: return float("inf")
    P, Q = np.array(P_star, dtype=float), np.array(A, dtype=float)
    return float(np.mean([
        float(np.min(np.sqrt(np.sum(np.maximum(Q-p, 0.0)**2, axis=1))))
        for p in P]))


def spacing_metric(A):
    if not A or len(A) < 2: return 0.0
    Q = np.array(A, dtype=float); n = Q.shape[0]
    dmin = []
    for i in range(n):
        diff = Q - Q[i]; d = np.sqrt(np.sum(diff**2, axis=1)); d[i] = np.inf
        dmin.append(float(np.min(d)))
    dmin = np.array(dmin)
    return float(np.sqrt(np.sum((dmin - np.mean(dmin))**2) / max(1, n-1)))


def build_P_star_fast(run_front_hist, tail_gens=PSTAR_TAIL_GENS,
                      cap_per_gen=PSTAR_CAP_PER_GEN, max_total=PSTAR_MAX_TOTAL):
    pts = []
    for hist in run_front_hist:
        tail = hist[-tail_gens:] if tail_gens > 0 else hist
        for gen_front in tail:
            pts.extend(gen_front[:cap_per_gen])
            if len(pts) >= max_total: break
        if len(pts) >= max_total: break
    return nondominated_set(pts)


# ════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════

def plot_hv_curve(gen, hv_mean, hv_std, save="plot_HV.png"):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen, hv_mean, lw=2.2, color="#9C27B0", label="HV (mean)")
    ax.fill_between(gen, hv_mean-hv_std, hv_mean+hv_std, alpha=0.22, color="#9C27B0", label="±std")
    ax.set_xlabel("Generation"); ax.set_ylabel("HV (normalised)")
    ax.set_ylim(bottom=0)
    ax.set_title("NoCarbon SPEA2: Hypervolume 2D Exact (mean ± std)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_igd_curve(gen, igd_mean, igd_std, save="plot_IGDplus.png"):
    ip  = np.where(np.isfinite(igd_mean), igd_mean, np.nan)
    is_ = np.where(np.isfinite(igd_std),  igd_std,  np.nan)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen, ip, lw=2.2, color="#E53935", label="IGD+ (mean)")
    ax.fill_between(gen, np.maximum(ip-is_, 0), ip+is_, alpha=0.2, color="#E53935", label="±std")
    ax.set_xlabel("Generation"); ax.set_ylabel("IGD+ (lower is better)")
    ax.set_title("NoCarbon SPEA2: IGD+ (mean ± std)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_spacing_curve(gen, sp_mean, sp_std, save="plot_Spacing.png"):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen, sp_mean, lw=2.2, color="#00897B", label="Spacing (mean)")
    ax.fill_between(gen, np.maximum(sp_mean-sp_std, 0), sp_mean+sp_std,
                    alpha=0.2, color="#00897B", label="±std")
    ax.set_xlabel("Generation"); ax.set_ylabel("Spacing")
    ax.set_title("NoCarbon SPEA2: Spacing (mean ± std)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_feasible_ratio_curve(gen, fr_mean, fr_std, frs_mean, frs_std,
                               boost_hist=None, save="plot_FeasibleRatio.png"):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen, fr_mean, lw=2.2, color="#1976D2", label="Soft (mean)")
    ax.fill_between(gen, np.clip(fr_mean-fr_std, 0, 1), np.clip(fr_mean+fr_std, 0, 1),
                    alpha=0.18, color="#1976D2", label="soft ±std")
    ax.plot(gen, frs_mean, lw=2.0, ls="--", color="#FB8C00", label="Strict (mean)")
    ax.fill_between(gen, np.clip(frs_mean-frs_std, 0, 1), np.clip(frs_mean+frs_std, 0, 1),
                    alpha=0.13, color="#FB8C00", label="strict ±std")
    if boost_hist:
        bgs = [g for g, v in enumerate(boost_hist) if v > 0]
        if bgs:
            ax.scatter(bgs, [fr_mean[g] for g in bgs if g < len(fr_mean)],
                       marker="^", color="red", s=40, zorder=5, label=f"Boost ({len(bgs)} gens)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Generation"); ax.set_ylabel("Feasible Ratio")
    ax.set_title("NoCarbon SPEA2: Feasible Ratio")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_min_objectives(gen, min_cost, min_time, save="plot_MinObjectives.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=180, sharex=True)
    fig.subplots_adjust(hspace=0.35)
    for ax, (data, ylabel, color) in zip(axes, [
        (min_cost, "Min Cost ($)",  "#E53935"),
        (min_time, "Min Time (h)",  "#FB8C00"),
    ]):
        ax.plot(gen, data, lw=2.2, color=color)
        mask = np.isfinite(data)
        if mask.any():
            fg = int(np.where(mask)[0][0])
            ax.axvline(fg, color="green", ls="--", alpha=0.7, lw=1.0,
                       label=f"First feasible gen {fg}")
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9); ax.grid(True, ls=":", alpha=0.5)
    axes[-1].set_xlabel("Generation")
    fig.suptitle("NoCarbon SPEA2: Min Objectives per Generation (feasible, 2-obj)",
                 fontsize=12, fontweight="bold")
    plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_convergence_combined(gen, hv_mean, hv_std, igd_mean, igd_std,
                               save="plot_Convergence.png"):
    ip  = np.where(np.isfinite(igd_mean), igd_mean, np.nan)
    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=180)
    ax1.plot(gen, hv_mean, lw=2.0, color="#9C27B0", label="HV (mean)")
    ax1.fill_between(gen, hv_mean-hv_std, hv_mean+hv_std, alpha=0.18, color="#9C27B0")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("HV (normalised)", color="#9C27B0")
    ax1.tick_params(axis="y", labelcolor="#9C27B0"); ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(gen, ip, lw=2.0, ls="--", color="#E53935", label="IGD+ (mean)")
    ax2.set_ylabel("IGD+ (lower is better)", color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="lower right", fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)
    plt.title("NoCarbon SPEA2: Convergence HV & IGD+ (2-obj, mean ± std)")
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_pareto_2d(pareto_points, save="plot_Pareto2D.png",
                   title="NoCarbon SPEA2 Pareto Front (Cost vs Time)"):
    if not pareto_points: return
    A = _finite_points_array(pareto_points)
    if A.shape[0] == 0: return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sc = ax.scatter(A[:, 0], A[:, 1], c=A[:, 0], cmap="viridis", s=45, alpha=0.9)
    plt.colorbar(sc, ax=ax, label="Cost ($)")
    idx = np.argsort(A[:, 0])
    ax.plot(A[idx, 0], A[idx, 1], lw=1.2, color="grey", alpha=0.5, ls="--")
    ax.set_xlabel("Cost ($)"); ax.set_ylabel("Time (h)")
    ax.set_title(title); ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_pareto_2d_allruns(all_pareto_pts_by_run, best_run_idx=None,
                            save="plot_Pareto2D_allruns.png"):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    for r_idx, pts in enumerate(all_pareto_pts_by_run):
        arr = _finite_points_array(pts)
        if arr.shape[0] == 0: continue
        if best_run_idx is not None and r_idx == best_run_idx: continue
        ax.scatter(arr[:, 0], arr[:, 1], s=12, alpha=0.3, color="#90CAF9", zorder=2)
    if best_run_idx is not None:
        arr_best = _finite_points_array(all_pareto_pts_by_run[best_run_idx])
        if arr_best.shape[0] > 0:
            sc = ax.scatter(arr_best[:, 0], arr_best[:, 1],
                            s=40, alpha=0.95, c=arr_best[:, 0], cmap="viridis",
                            zorder=5, label=f"Best run #{best_run_idx}")
            plt.colorbar(sc, ax=ax, label="Cost ($)")
            idx = np.argsort(arr_best[:, 0])
            ax.plot(arr_best[idx, 0], arr_best[idx, 1], lw=1.2, color="grey", alpha=0.5, ls="--")
            ax.legend(fontsize=9)
    ax.set_xlabel("Cost ($)"); ax.set_ylabel("Time (h)")
    ax.set_title("NoCarbon SPEA2: Pareto All Runs (grey) + Best Run (colour)")
    ax.grid(True, ls=":", alpha=0.45)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_operator_prob(gen, save="plot_OperatorProb.png"):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    colors = ["#1976D2","#E53935","#00897B","#FB8C00"]
    for op, color in zip(OPS, colors):
        ax.axhline(0.25, color=color, lw=2.0, label=op.capitalize(), alpha=0.85)
    ax.set_ylim(0, 0.5); ax.set_xlabel("Generation"); ax.set_ylabel("Selection Probability")
    ax.set_title("NoCarbon SPEA2: Fixed Equal-Weight Mutation Operator (0.25 each)")
    ax.legend(loc="upper right"); ax.grid(axis="y", ls="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_runtime_per_run(run_rows, save="plot_Runtime.png"):
    df  = pd.DataFrame(run_rows)
    avg = df["runtime_s"].mean()
    colors = ["#E53935" if r > avg else "#1976D2" for r in df["runtime_s"]]
    fig, ax = plt.subplots(figsize=(max(8, len(df)*0.35), 4), dpi=180)
    ax.bar(df["run_id"], df["runtime_s"], color=colors, alpha=0.85)
    ax.axhline(avg, color="black", ls="--", lw=1.5, label=f"Mean = {avg:.1f}s")
    ax.set_xlabel("Run ID"); ax.set_ylabel("Runtime (s)")
    ax.set_title("NoCarbon SPEA2: Runtime per Run"); ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_summary_metrics_table(hv_runs, igd_runs, sp_runs, fr_runs, frs_runs, run_rows,
                                save="plot_SummaryTable.png"):
    metrics = {
        "HV_norm (↑)": hv_runs[:, -1],
        "IGD+ (↓)":    np.array(igd_runs)[:, -1],
        "Spacing (↓)": np.array(sp_runs)[:, -1],
        "FeasRatio soft (↑)":   fr_runs[:, -1],
        "FeasRatio strict (↑)": frs_runs[:, -1],
        "Runtime (s)": np.array([r["runtime_s"] for r in run_rows]),
    }
    table_data = []
    for name, vals in metrics.items():
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            table_data.append([name,"—","—","—","—"])
        else:
            table_data.append([name, f"{np.min(finite):.4f}", f"{np.max(finite):.4f}",
                                f"{np.mean(finite):.4f}", f"{np.std(finite):.4f}"])
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.axis("off")
    tbl = ax.table(cellText=table_data,
                   colLabels=["Metric","Min","Max","Mean","Std"],
                   cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for j in range(5):
        tbl[(0,j)].set_facecolor("#1F4E79")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data)):
        clr = "#EBF3FB" if i%2==0 else "#FFFFFF"
        for j in range(5): tbl[(i+1,j)].set_facecolor(clr)
    fig.suptitle(f"NoCarbon SPEA2 Summary — {len(run_rows)} Runs (2-Obj: Cost & Time)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(save, bbox_inches="tight"); plt.close()
    print(f"[PLOT] {save}")


# ════════════════════════════════════════════════════════
# Export helpers
# ════════════════════════════════════════════════════════

def _min_obj_from_front(front_objs, obj_idx):
    arr = _finite_points_array(front_objs)
    if arr.shape[0] == 0: return np.nan
    return float(np.min(arr[:, obj_idx]))


def _min_obj_over_history(run_hist, obj_idx):
    best = np.nan
    for gen_front in run_hist:
        v = _min_obj_from_front(gen_front, obj_idx)
        if np.isfinite(v):
            best = v if (not np.isfinite(best) or v < best) else best
    return best


def export_metrics_csv(gen_arr, hv_mean, hv_std, igd_mean, igd_std, sp_mean, sp_std,
                        fr_mean, fr_std, frs_mean, frs_std,
                        min_cost_best, min_time_best,
                        boost_hist_best, vio_mean_dict_mean,
                        out_csv="metrics_per_generation.csv"):
    df = pd.DataFrame({
        "generation": gen_arr,
        "HV_norm_mean": hv_mean,  "HV_norm_std": hv_std,
        "IGD_plus_mean": igd_mean, "IGD_plus_std": igd_std,
        "Spacing_mean": sp_mean,   "Spacing_std": sp_std,
        "FeasRatio_soft_mean": fr_mean,  "FeasRatio_soft_std": fr_std,
        "FeasRatio_strict_mean": frs_mean, "FeasRatio_strict_std": frs_std,
        "MinCost_bestrun": min_cost_best,
        "MinTime_h_bestrun": min_time_best,
        "boost_triggered_bestrun": boost_hist_best if boost_hist_best else [0]*len(gen_arr),
    })
    for k, series in vio_mean_dict_mean.items():
        df[f"vio_{k}_mean"] = series
    df.to_csv(out_csv, index=False)
    print(f"[EXPORT] Per-generation metrics → {out_csv}")
    return df


def export_run_summary_excel(run_rows, hv_runs, igd_runs, sp_runs, run_front_hist,
                              out_xlsx="run_summary.xlsx"):
    df = pd.DataFrame(run_rows)
    df["final_HV_norm"]  = np.array(hv_runs,   dtype=float)[:, -1]
    df["final_IGD_plus"] = np.array(igd_runs,  dtype=float)[:, -1]
    df["final_Spacing"]  = np.array(sp_runs,   dtype=float)[:, -1]
    final_min_cost = [_min_obj_from_front(h[-1] if h else [], 0) for h in run_front_hist]
    final_min_time = [_min_obj_from_front(h[-1] if h else [], 1) for h in run_front_hist]
    best_cost      = [_min_obj_over_history(h, 0) for h in run_front_hist]
    best_time      = [_min_obj_over_history(h, 1) for h in run_front_hist]
    df["final_min_cost"]          = final_min_cost
    df["final_min_time_h"]        = final_min_time
    df["best_min_cost_over_gens"] = best_cost
    df["best_min_time_over_gens_h"] = best_time
    def _summ(s):
        s = pd.to_numeric(s, errors="coerce"); s = s[np.isfinite(s)]
        if len(s) == 0: return {"min":np.nan,"max":np.nan,"mean":np.nan,"std":np.nan}
        return {"min":float(s.min()),"max":float(s.max()),
                "mean":float(s.mean()),"std":float(s.std())}
    summary_metrics = {k: _summ(df[k]) for k in
        ["final_min_cost","final_min_time_h","final_HV_norm",
         "final_IGD_plus","final_Spacing","runtime_s"]}
    summary_df = pd.DataFrame([{"metric":k,**v} for k, v in summary_metrics.items()])
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="RunSummary", index=False)
        summary_df.to_excel(writer, sheet_name="SummaryStats", index=False)
    print(f"[EXPORT] Run summary → {out_xlsx}")
    return df


def save_pareto_solutions(pareto, batches, filename="result.txt"):
    # [⑦] tol=1e-6
    pareto = unique_individuals_by_objectives([i for i in pareto if i.feasible], tol=1e-6)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NoCarbon SPEA2 Pareto (Cost & Time, Fixed 0.25 Mutation) =====\n\n")
        if not pareto:
            f.write("NO FEASIBLE SOLUTION FOUND.\n"); return
        for i, ind in enumerate(pareto):
            c, t = ind.objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"Cost={c:.6f}  Time={t:.6f}\n")
            f.write(f"Penalty={ind.penalty:.6f}  Feasible={ind.feasible}\n")
            f.write(f"Breakdown={ind.vio_breakdown}\n\n")
            for b in batches:
                key   = (b.origin, b.destination, b.batch_id)
                allocs= ind.od_allocations.get(key, [])
                if not allocs: continue
                f.write(f"Batch {b.batch_id}: {b.origin} -> {b.destination}, Q={b.quantity}\n")
                for a in allocs: f.write(str(a) + "\n")
                f.write("\n")
            f.write("\n")
    print(f"[EXPORT] {len(pareto)} feasible Pareto solutions → {filename}")


from pathlib import Path as FSPath
import json as _json


def export_pareto_points_json(pareto, batches, out_json="nocarbon_pareto_points.json"):
    FSPath(FSPath(out_json).parent).mkdir(parents=True, exist_ok=True)
    out = []
    for ind in pareto:
        sol = {
            "objectives": {"cost": float(ind.objectives[0]),
                           "time_h": float(ind.objectives[1]),
                           "penalty": float(ind.penalty)},
            "feasible": bool(ind.feasible),
            "vio_breakdown": {k: float(v) for k, v in (ind.vio_breakdown or {}).items()},
            "allocations": []
        }
        for b in batches:
            key = (b.origin, b.destination, b.batch_id)
            blk = {"batch_id": int(b.batch_id), "origin": b.origin,
                   "destination": b.destination, "paths": []}
            for a in ind.od_allocations.get(key, []):
                blk["paths"].append({"share": float(a.share),
                                     "nodes": list(a.path.nodes),
                                     "modes": list(a.path.modes)})
            sol["allocations"].append(blk)
        out.append(sol)
    with open(out_json, "w", encoding="utf-8") as f:
        _json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[EXPORT] Pareto JSON → {out_json}")


# ════════════════════════════════════════════════════════
# Extract helpers
# ════════════════════════════════════════════════════════

def extract_pareto_size_per_gen(run_front_hist):
    runs = len(run_front_hist)
    if runs == 0: return np.array([])
    generations = len(run_front_hist[0])
    mat = np.zeros((runs, generations), dtype=float)
    for r, hist in enumerate(run_front_hist):
        for g, gf in enumerate(hist):
            mat[r, g] = float(len(gf))
    return mat


def extract_min_obj_per_gen_allruns(run_front_hist, obj_idx):
    runs = len(run_front_hist)
    if runs == 0: return np.array([]), np.array([])
    generations = len(run_front_hist[0])
    mat = np.full((runs, generations), np.nan)
    for r, hist in enumerate(run_front_hist):
        for g, gf in enumerate(hist):
            arr = _finite_points_array(gf)
            if arr.shape[0] > 0:
                mat[r, g] = float(np.min(arr[:, obj_idx]))
    for r in range(runs):
        mat[r] = _ffill_nan(mat[r])
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


# ════════════════════════════════════════════════════════
# SPEA2 Runner
# ════════════════════════════════════════════════════════

def load_shared_data(filename="data.xlsx"):
    print("Loading data...")
    (node_names, node_region,
     node_hold_cost, node_proc_cost,
     arcs, timetables, batches,
     waiting_cost_per_teu_h,
     mode_speeds_map, trans_map, border_delay_map) = load_network_from_extended(filename)
    tt_dict    = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)
    print("Building path library...")
    path_lib = build_path_library(node_names, node_region, arcs, batches, tt_dict, arc_lookup)
    sanity_check_path_lib(batches, path_lib)
    return dict(
        node_names=node_names, node_region=node_region,
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        arcs=arcs, timetables=timetables, batches=batches,
        waiting_cost_per_teu_h=waiting_cost_per_teu_h,
        mode_speeds_map=mode_speeds_map, trans_map=trans_map,
        border_delay_map=border_delay_map,
        tt_dict=tt_dict, arc_lookup=arc_lookup, path_lib=path_lib,
    )


def run_spea2_nocarbon(filename="data.xlsx", pop_size=125, generations=400,
                        shared_data=None):
    if shared_data is None:
        shared_data = load_shared_data(filename)

    arcs             = shared_data["arcs"]
    batches          = shared_data["batches"]
    tt_dict          = shared_data["tt_dict"]
    arc_lookup       = shared_data["arc_lookup"]
    path_lib         = shared_data["path_lib"]
    node_hold_cost   = shared_data["node_hold_cost"]
    node_proc_cost   = shared_data["node_proc_cost"]
    trans_map        = shared_data["trans_map"]
    border_delay_map = shared_data["border_delay_map"]
    waiting_cost_per_teu_h = shared_data["waiting_cost_per_teu_h"]

    archive_size = pop_size

    eval_kwargs = dict(
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        trans_map=trans_map, border_delay_map=border_delay_map,
    )

    # Initialise population
    population = []
    n_greedy   = max(1, pop_size // 3)
    for i in range(pop_size):
        ind = (greedy_initial_individual(batches, path_lib) if i < n_greedy
               else random_initial_individual(batches, path_lib))
        repair_missing_allocations(ind, batches, path_lib)
        evaluate_individual(ind, batches, arcs, tt_dict,
                            waiting_cost_per_teu_h, **eval_kwargs)
        population.append(ind)

    archive: List[Individual] = []

    front_hist_objs             = []
    feasible_ratio_hist         = []
    feasible_ratio_strict_hist  = []
    vio_mean_hist = {k: [] for k in
        ["miss_alloc","miss_tt","cap_excess","late_teu_h","wait_teu_h"]}
    boost_trigger_hist      = []
    boost_new_feasible_hist = []

    _run_start = time.perf_counter()
    _prev_best = [float("inf")] * NUM_OBJ

    for gen in range(generations):

        # Step 1: Environmental selection
        combined = population + archive
        archive  = spea2_environmental_selection(combined, archive_size)

        # Step 2: Record metrics — [⑦] tol=1e-6
        feas_archive = [ind for ind in archive if ind.feasible]
        display      = unique_individuals_by_objectives(feas_archive, tol=1e-6)

        front_hist_objs.append([ind.objectives for ind in display])
        feas_total = len(feas_archive)
        feasible_ratio_hist.append(feas_total / max(1, archive_size))
        feasible_ratio_strict_hist.append(
            sum(1 for i in archive if i.feasible_hard) / max(1, archive_size))

        for k in vio_mean_hist:
            vals = [ind.vio_breakdown.get(k, 0.0) for ind in archive]
            vio_mean_hist[k].append(float(np.mean(vals)) if vals else 0.0)

        elapsed = time.perf_counter() - _run_start
        if feas_archive:
            cur    = [min(i.objectives[j] for i in feas_archive) for j in range(NUM_OBJ)]
            d      = ["↓" if cur[j] < _prev_best[j] - 1e-3 else "→" for j in range(NUM_OBJ)]
            _prev_best = cur
            obj_str = f"Cost={cur[0]:.3e}{d[0]} Time={cur[1]:.1f}h{d[1]}"
        else:
            obj_str = "No feasible solutions yet"

        best_pen = min(i.penalty for i in archive) if archive else float("inf")
        sep      = "=" * 72
        print(f"\n{sep}")
        print(f"  [NoCarbon SPEA2] Gen {gen:03d}/{generations-1}  |  {elapsed:.1f}s elapsed")
        print(f"  Archive feasible: {feas_total}/{archive_size} ({feasible_ratio_hist[-1]:.1%})"
              f"  |  NonDom={len(display)}  |  BestPenalty={best_pen:.2e}")
        print(f"  Best: {obj_str}")
        print(sep)

        # Step 3: Feasibility boost [⑨ 与Baseline一致]
        boost_triggered = boost_new_feas = 0
        if feas_total < MIN_FEASIBLE_SOLUTIONS:
            boost_triggered = 1
            archive, boost_new_feas = feasibility_boost(
                archive, batches, path_lib, tt_dict, arc_lookup,
                arcs, waiting_cost_per_teu_h, eval_kwargs)
            after = sum(1 for i in archive if i.feasible)
            print(f"  ⚡ [BOOST] {feas_total} < {MIN_FEASIBLE_SOLUTIONS} → after: {after} (+{boost_new_feas})")

        boost_trigger_hist.append(boost_triggered)
        boost_new_feasible_hist.append(boost_new_feas)

        # Step 4: Binary tournament mating pool
        compute_spea2_fitness(archive)
        mating_pool = [spea2_binary_tournament(archive) for _ in range(pop_size)]

        # Step 5: Crossover + Mutation — 与Baseline一致
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup)
            else:
                c1 = random_initial_individual(batches, path_lib)
                c2 = random_initial_individual(batches, path_lib)

            if random.random() < MUTATION_RATE:
                mutate_fixed(c1, batches, path_lib, tt_dict, arc_lookup,
                             arcs, waiting_cost_per_teu_h, **eval_kwargs)
            if random.random() < MUTATION_RATE:
                mutate_fixed(c2, batches, path_lib, tt_dict, arc_lookup,
                             arcs, waiting_cost_per_teu_h, **eval_kwargs)

            # Mandatory single evaluate per child (mirrors Baseline)
            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict,
                                waiting_cost_per_teu_h, **eval_kwargs)
            evaluate_individual(c2, batches, arcs, tt_dict,
                                waiting_cost_per_teu_h, **eval_kwargs)
            offspring.extend([c1, c2])

        population = offspring[:pop_size]

    # Final Pareto — [⑦] tol=1e-6
    compute_spea2_fitness(archive)
    pareto = unique_individuals_by_objectives(
        [ind for ind in archive if ind.spea2_fitness < 1.0 and ind.feasible], tol=1e-6)
    if not pareto and feas_archive:
        pareto = unique_individuals_by_objectives(
            sorted(feas_archive, key=lambda x: x.spea2_fitness), tol=1e-6)

    total_t = time.perf_counter() - _run_start
    print(f"\n{'='*72}")
    print(f"  [NoCarbon SPEA2] Run complete: {generations} gens, {total_t:.1f}s, Pareto={len(pareto)}")
    print(f"  Boost: {sum(boost_trigger_hist)} gens, {sum(boost_new_feasible_hist)} new feasible")
    print(f"{'='*72}")

    return (population, pareto, batches,
            front_hist_objs,
            feasible_ratio_hist, feasible_ratio_strict_hist,
            vio_mean_hist,
            boost_trigger_hist, boost_new_feasible_hist)


# ════════════════════════════════════════════════════════
# Parallel worker
# ════════════════════════════════════════════════════════

def _run_one(args):
    run_id, seed, pop_size, generations, shared_data = args
    random.seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}\n  RUN {run_id}  seed={seed}\n{'='*60}")
    t0     = time.perf_counter()
    result = run_spea2_nocarbon(
        pop_size=pop_size, generations=generations, shared_data=shared_data)
    runtime_s = time.perf_counter() - t0
    print(f"[RUN {run_id}] done in {runtime_s:.1f}s")
    return run_id, runtime_s, result


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing as mp

    filename    = "data.xlsx"
    pop_size    = 125
    generations = 400
    runs        = 30
    n_workers   = min(runs, mp.cpu_count())

    print(f"[CONFIG] Algorithm : NoCarbon SPEA2 — 2-Objective (Cost & Time)")
    print(f"[CONFIG] pop_size={pop_size}  archive_size={pop_size}")
    print(f"[CONFIG] generations={generations}")
    print(f"[CONFIG] CROSSOVER_RATE={CROSSOVER_RATE}  MUTATION_RATE={MUTATION_RATE}")
    print(f"[CONFIG] Mutation   : Fixed equal-weight (W_ADD=W_DEL=W_MOD=W_MODE=0.25)")
    print(f"[CONFIG] HV ref     : {HV_REF_NORM} (2D exact sweep)")
    print(f"[CONFIG] runs={runs}  workers={n_workers}  CPUs={mp.cpu_count()}")

    shared_data = load_shared_data(filename)

    task_args = [
        (run_id, 1000 + run_id, pop_size, generations, shared_data)
        for run_id in range(runs)
    ]

    print(f"\n[PARALLEL] Launching {runs} runs across {n_workers} workers...\n")
    with mp.Pool(processes=n_workers) as pool:
        raw_results = pool.map(_run_one, task_args)
    raw_results.sort(key=lambda x: x[0])

    run_front_hist, run_feasible_ratio, run_feasible_ratio_strict = [], [], []
    run_vio_mean, run_rows = [], []
    run_paretos, run_batches_list   = [], []
    run_boost_trigger_hist, run_boost_new_feas_hist = [], []

    for run_id, runtime_s, result in raw_results:
        (pop, pareto, batches, front_hist, fr_hist, frs_hist,
         vio_hist, boost_trigger, boost_new_feas) = result

        run_front_hist.append(front_hist)
        run_feasible_ratio.append(fr_hist)
        run_feasible_ratio_strict.append(frs_hist)
        run_vio_mean.append(vio_hist)
        run_paretos.append(pareto)
        run_batches_list.append(batches)
        run_boost_trigger_hist.append(boost_trigger)
        run_boost_new_feas_hist.append(boost_new_feas)

        run_rows.append({
            "run_id": run_id, "seed": 1000 + run_id,
            "runtime_s": runtime_s, "runtime_min": runtime_s / 60.0,
            "final_feasible_ratio_soft":   float(fr_hist[-1])  if fr_hist  else 0.0,
            "final_feasible_ratio_strict": float(frs_hist[-1]) if frs_hist else 0.0,
            "final_pareto_size": int(len(pareto)),
            "boost_gens_triggered":       int(sum(boost_trigger)),
            "boost_total_new_feasible":   int(sum(boost_new_feas)),
        })
        print(f"[RUN {run_id}] Pareto={len(pareto)}  Runtime={runtime_s:.1f}s  "
              f"FeasSoft={fr_hist[-1]:.2%}  FeasStrict={frs_hist[-1]:.2%}")

    # P* & normalisation
    P_star = build_P_star_fast(run_front_hist)
    print(f"\n[P*] Size = {len(P_star)}")
    if P_star:
        P_arr = np.array(P_star, dtype=float)
        mins, maxs = np.min(P_arr, axis=0), np.max(P_arr, axis=0)
    else:
        mins, maxs = np.zeros(NUM_OBJ), np.ones(NUM_OBJ)

    hv_calc = HypervolumeCalculator2D(ref_point=HV_REF_NORM)
    Pn = normalize_points(P_star, mins, maxs) if P_star else []

    hv_runs, igd_runs, sp_runs = [], [], []
    for r in range(runs):
        hv_h, igd_h, sp_h = [], [], []
        last_hv = last_igd = 0.0; last_sp = 0.0
        for gi, gf in enumerate(run_front_hist[r]):
            if gi % HV_EVERY == 0:
                pts = [tuple(x) for x in _finite_points_array(gf)]
                An  = clip_points(normalize_points(pts, mins, maxs), HV_REF_NORM) if pts else []
                last_hv = hv_calc.calculate_points(An) if An else 0.0
            if gi % METRIC_EVERY == 0:
                pts     = [tuple(x) for x in _finite_points_array(gf)]
                An      = normalize_points(pts, mins, maxs) if pts else []
                last_igd= igd_plus(Pn, An) if (Pn and An) else float("inf")
                last_sp = spacing_metric(An) if An else 0.0
            hv_h.append(last_hv); igd_h.append(last_igd); sp_h.append(last_sp)
        hv_runs.append(hv_h); igd_runs.append(igd_h); sp_runs.append(sp_h)

    hv_runs_arr  = np.array(hv_runs,  dtype=float)
    igd_runs_arr = np.array(igd_runs, dtype=float)
    sp_runs_arr  = np.array(sp_runs,  dtype=float)
    fr_runs      = np.array(run_feasible_ratio,        dtype=float)
    frs_runs     = np.array(run_feasible_ratio_strict, dtype=float)

    gen = np.arange(generations)
    hv_mean,  hv_std  = np.mean(hv_runs_arr,  0), np.std(hv_runs_arr,  0)
    igd_mean, igd_std = np.mean(igd_runs_arr, 0), np.std(igd_runs_arr, 0)
    sp_mean,  sp_std  = np.mean(sp_runs_arr,  0), np.std(sp_runs_arr,  0)
    fr_mean,  fr_std  = np.mean(fr_runs,  0), np.std(fr_runs,  0)
    frs_mean, frs_std = np.mean(frs_runs, 0), np.std(frs_runs, 0)

    vio_keys = ["miss_alloc","miss_tt","cap_excess","late_teu_h","wait_teu_h"]
    vio_mean_dict_mean = {}
    for k in vio_keys:
        mat = np.array([run_vio_mean[r].get(k, [0.0]*generations)
                        for r in range(runs)], dtype=float)
        vio_mean_dict_mean[k] = list(np.mean(mat, axis=0))

    best_run_idx = int(np.argmax(hv_runs_arr[:, -1]))
    best_run_hv  = float(hv_runs_arr[best_run_idx, -1])
    best_pareto  = run_paretos[best_run_idx]
    best_batches = run_batches_list[best_run_idx]
    best_front_hist = run_front_hist[best_run_idx]
    best_boost_hist = run_boost_trigger_hist[best_run_idx]

    def _extract_min(hist_list, idx):
        vals = []
        for fh in hist_list:
            arr = _finite_points_array(fh)
            vals.append(float(np.min(arr[:, idx])) if arr.shape[0] > 0 else np.nan)
        return _ffill_nan(np.array(vals))

    min_cost_b = _extract_min(best_front_hist, 0)
    min_time_b = _extract_min(best_front_hist, 1)

    def _stat(arr_2d):
        v = arr_2d[:, -1] if arr_2d.ndim == 2 else arr_2d
        f = v[np.isfinite(v)]
        if len(f) == 0: return "all inf"
        return (f"min={np.min(f):.4f} max={np.max(f):.4f} "
                f"mean={np.mean(f):.4f} std={np.std(f):.4f}")

    print(f"\n{'='*68}")
    print(f"  NoCarbon SPEA2 SUMMARY — {runs} RUNS, 2-Obj (Cost & Time)")
    print(f"  Mutation: Fixed equal-weight (0.25 each)")
    print(f"{'='*68}")
    print(f"  HV_norm (↑): {_stat(hv_runs_arr)}")
    print(f"  IGD+    (↓): {_stat(igd_runs_arr)}")
    print(f"  Spacing (↓): {_stat(sp_runs_arr)}")
    print(f"  FeasSoft(↑): {_stat(fr_runs)}")
    print(f"  FeasStr (↑): {_stat(frs_runs)}")
    rts = np.array([r["runtime_s"] for r in run_rows])
    print(f"  Runtime (s): min={np.min(rts):.1f} max={np.max(rts):.1f} "
          f"mean={np.mean(rts):.1f} std={np.std(rts):.1f}")
    print(f"  Best run: #{best_run_idx}  HV={best_run_hv:.4f}  Pareto={len(best_pareto)}")
    print(f"{'='*68}\n")

    # Export
    export_metrics_csv(
        gen_arr=gen.tolist(),
        hv_mean=hv_mean.tolist(),  hv_std=hv_std.tolist(),
        igd_mean=igd_mean.tolist(), igd_std=igd_std.tolist(),
        sp_mean=sp_mean.tolist(),   sp_std=sp_std.tolist(),
        fr_mean=fr_mean.tolist(),   fr_std=fr_std.tolist(),
        frs_mean=frs_mean.tolist(), frs_std=frs_std.tolist(),
        min_cost_best=min_cost_b.tolist(),
        min_time_best=min_time_b.tolist(),
        boost_hist_best=best_boost_hist,
        vio_mean_dict_mean=vio_mean_dict_mean,
    )
    export_run_summary_excel(run_rows, hv_runs_arr, igd_runs_arr, sp_runs_arr,
                              run_front_hist)
    if best_pareto:
        save_pareto_solutions(best_pareto, best_batches, "result.txt")
        export_pareto_points_json(best_pareto, best_batches, "nocarbon_pareto_points.json")

    # Plots
    print("\n[PLOTTING] Generating figures...")
    plot_hv_curve(gen, hv_mean, hv_std)
    plot_igd_curve(gen, igd_mean, igd_std)
    plot_spacing_curve(gen, sp_mean, sp_std)
    plot_convergence_combined(gen, hv_mean, hv_std, igd_mean, igd_std)
    plot_feasible_ratio_curve(gen, fr_mean, fr_std, frs_mean, frs_std, best_boost_hist)
    plot_min_objectives(gen, min_cost_b, min_time_b)
    plot_runtime_per_run(run_rows)
    plot_operator_prob(gen)

    best_final_pts = unique_objective_tuples(
        [ind.objectives for ind in best_pareto if ind.feasible], tol=1e-9)
    plot_pareto_2d(best_final_pts, save="plot_Pareto2D.png",
                   title=f"NoCarbon SPEA2 Pareto — Best Run #{best_run_idx}")

    all_pareto_pts_by_run = [
        [ind.objectives for ind in run_paretos[r] if ind.feasible]
        for r in range(runs)
    ]
    plot_pareto_2d_allruns(all_pareto_pts_by_run, best_run_idx=best_run_idx)
    plot_summary_metrics_table(hv_runs_arr, igd_runs_arr, sp_runs_arr,
                               fr_runs, frs_runs, run_rows)

    mc_m, mc_s = extract_min_obj_per_gen_allruns(run_front_hist, 0)
    mt_m, mt_s = extract_min_obj_per_gen_allruns(run_front_hist, 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=180, sharex=True)
    for ax, (m, s, lbl, clr) in zip(axes, [
        (mc_m, mc_s, "Min Cost ($)",  "#E53935"),
        (mt_m, mt_s, "Min Time (h)",  "#FB8C00"),
    ]):
        ax.plot(gen, m, lw=2.2, color=clr, label=f"{lbl} (mean)")
        if s is not None:
            ax.fill_between(gen,
                np.where(np.isfinite(m-s), m-s, np.nan),
                np.where(np.isfinite(m+s), m+s, np.nan),
                alpha=0.2, color=clr, label="±std")
        mask = np.isfinite(m)
        if mask.any():
            fg = int(np.where(mask)[0][0])
            ax.axvline(fg, color="green", ls="--", lw=1.2, alpha=0.7,
                       label=f"First feasible gen {fg}")
        ax.set_ylabel(lbl); ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)
    axes[-1].set_xlabel("Generation")
    plt.suptitle("NoCarbon SPEA2: Min Objectives per Generation (mean ± std)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig("plot_MinObj_trend.png"); plt.close()
    print("[PLOT] plot_MinObj_trend.png")

    psz_mat = extract_pareto_size_per_gen(run_front_hist)
    ps_mean = np.mean(psz_mat, axis=0); ps_std = np.std(psz_mat, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    ax.plot(gen, ps_mean, lw=2.2, color="#5C6BC0", label="Archive size (mean)")
    ax.fill_between(gen, np.maximum(ps_mean-ps_std, 0), ps_mean+ps_std,
                    alpha=0.2, color="#5C6BC0", label="±std")
    ax.plot(gen, psz_mat[best_run_idx], lw=1.5, ls="--", color="#E91E63",
            alpha=0.85, label=f"Best run #{best_run_idx}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Feasible archive size")
    ax.set_title("NoCarbon SPEA2: Archive Feasible Size Evolution")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig("plot_ParetoSize_trend.png"); plt.close()
    print("[PLOT] plot_ParetoSize_trend.png")

    print("\n[DONE] NoCarbon SPEA2 (fixed mutation) complete.")