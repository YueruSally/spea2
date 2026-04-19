#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, math, random, csv, pathlib, time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# ── 导入 baseline 模块 ────────────────────────────────────────────────────
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "baseline", os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.py"))
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)

from baseline import (
    load_network_from_extended,
    build_timetable_dict, build_arc_lookup,
    build_path_library, sanity_check_path_lib,
    Individual, PathAllocation, Path,
    repair_missing_allocations, evaluate_individual,
    spea2_environmental_selection, compute_spea2_fitness,
    spea2_binary_tournament,
    greedy_initial_individual, random_initial_individual,
    crossover_hybrid, mutate_fixed,
    feasibility_boost,
    CROSSOVER_RATE, MUTATION_RATE,
    MIN_FEASIBLE_SOLUTIONS,
)

# ═══════════════════════════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════════════════════════
DATA_FILE  = "data.xlsx"
NC_JSON    = "nocarbon_pareto_points.json"
OUTPUT_DIR = "inject_results_v2"
POP_SIZE   = 125
ARCHIVE    = 125
GENS       = 400
SEED       = 1000
OBJ_TOL    = 1e-3


# ═══════════════════════════════════════════════════════════════════════════
# 淘汰链节点数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChainNode:
    """
    追踪树中的一个节点，代表一个曾在 archive 中出现过的解。
    """
    node_id:    int                        # 全局唯一 ID
    obj:        tuple                      # (cost, emission, time)
    born_gen:   int                        # 首次出现（进入追踪）的代
    level:      int                        # 在树中的深度（0 = 原始注入点）
    label:      str                        # 描述标签，如 "NC-Sol000" / "Dom-L1-#3"
    parent_id:  Optional[int] = None       # 父节点 ID（谁被它支配）

    # 以下字段在节点消失后填充
    elim_gen:    Optional[int]   = None    # 被淘汰的代
    elim_reason: str             = ""      # "dominated" / "truncation" / "survived"
    elim_by_ids: List[int]       = field(default_factory=list)  # 支配它的子节点 ID 列表

    # 辅助
    still_alive: bool = True               # 当前代是否还在 archive 中


def fmt_obj(obj):
    return f"Cost={obj[0]/1e6:.4f}M  Emis={obj[1]:.3e}g  Time={obj[2]:.1f}h"

def fmt_obj_short(obj):
    return f"[{obj[0]/1e6:.3f}M / {obj[1]:.2e}g / {obj[2]:.1f}h]"


# ═══════════════════════════════════════════════════════════════════════════
# 1. 从 JSON 重建 nocarbon Individual（与原版相同）
# ═══════════════════════════════════════════════════════════════════════════

def rebuild_nc_individual(sol, arc_lookup, batches):
    bid2batch = {b.batch_id: b for b in batches}
    ind = Individual()
    skipped_alloc = skipped_arc = 0
    for ab in sol.get("allocations", []):
        bid   = ab.get("batch_id")
        o     = ab.get("origin", "").strip()
        d     = ab.get("destination", "").strip()
        paths = ab.get("paths", [])
        if not paths: continue
        b = bid2batch.get(bid)
        if b is None: skipped_alloc += 1; continue
        key = (o, d, bid)
        pa  = []
        for pi, pj in enumerate(paths):
            nodes = pj["nodes"]; modes = pj["modes"]; share = float(pj["share"])
            arc_seq, ok = [], True
            for i in range(len(modes)):
                k = (nodes[i], nodes[i+1], modes[i])
                if k not in arc_lookup: skipped_arc += 1; ok = False; break
                arc_seq.append(arc_lookup[k])
            if not ok: continue
            pa.append(PathAllocation(
                path=Path(path_id=pi, origin=o, destination=d,
                          nodes=nodes, modes=modes, arcs=arc_seq,
                          base_cost_per_teu=sum(a.cost_per_teu_km*a.distance for a in arc_seq),
                          base_emission_per_teu=sum(a.emission_per_teu_km*a.distance for a in arc_seq),
                          base_travel_time_h=sum(a.distance/max(a.speed_kmh,1.0) for a in arc_seq)),
                share=share))
        if pa: ind.od_allocations[key] = pa
    if skipped_alloc: print(f"    [WARN] {skipped_alloc} allocs 跳过")
    if skipped_arc:   print(f"    [WARN] {skipped_arc} arcs 跳过")
    return ind


def load_nocarbon_individuals(json_path, arc_lookup, batches, path_lib,
                               arcs, tt_dict, wc, we, eval_kwargs, report_lines):
    with open(json_path, "r", encoding="utf-8") as f:
        nc_all = json.load(f)
    sols = [s for s in nc_all if s.get("feasible", False)] or nc_all
    print(f"\n[NC] 读取 {len(sols)} 个 nocarbon 可行解")
    report_lines.append(f"  从 {json_path} 读取 {len(sols)} 个可行解\n")
    inds = []
    for si, sol in enumerate(sols):
        ind = rebuild_nc_individual(sol, arc_lookup, batches)
        repair_missing_allocations(ind, batches, path_lib)
        evaluate_individual(ind, batches, arcs, tt_dict, wc, we, **eval_kwargs)
        obj  = ind.objectives
        feas = "✅" if ind.feasible else "❌"
        print(f"  Sol {si:03d}: {feas}  {fmt_obj(obj)}")
        report_lines.append(f"  Sol {si:03d}: {feas}  {fmt_obj(obj)}")
        for key, allocs in ind.od_allocations.items():
            o, d, _ = key; od = (o, d)
            if od not in path_lib: path_lib[od] = []
            for alloc in allocs:
                if not any(p.nodes==alloc.path.nodes and p.modes==alloc.path.modes
                           for p in path_lib[od]):
                    path_lib[od].append(alloc.path)
        inds.append(ind)
    return inds


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _close(a, b, rtol=OBJ_TOL):
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)): return False
        if abs(x-y) / max(abs(x), abs(y), 1.0) > rtol:  return False
    return True

def _dom(a_obj, b_obj):
    return (all(x <= y for x, y in zip(a_obj, b_obj)) and
            any(x <  y for x, y in zip(a_obj, b_obj)))

def _dom_dims(nc_o, bo):
    """返回 nc_o 比 bo 差的维度描述（Cost/Emission/Time 各自偏高百分比）。"""
    labels = ["Cost", "Emission", "Time"]
    diff   = []
    for k in range(3):
        if nc_o[k] > bo[k] * (1 + 1e-6):
            pct = (nc_o[k]-bo[k]) / max(abs(bo[k]), 1e-9) * 100
            diff.append(f"{labels[k]}偏高{pct:.1f}%")
    return diff if diff else ["差距微小"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. 带完整多级追踪的 SPEA2 主循环
# ═══════════════════════════════════════════════════════════════════════════

def run_spea2_full_chain(
    node_names, node_region, node_hold_cost, node_proc_cost,
    arcs, timetables, batches,
    wc, we, carbon_tax_map, emission_factor_map, mode_speeds_map,
    trans_map, border_delay_map, path_lib,
    nc_individuals: List[Individual],
    pop_size=125, generations=400, archive_size=125,
    report_lines: List[str] = None,
):
    if report_lines is None:
        report_lines = []

    tt_dict    = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)
    eval_kwargs = dict(node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
                       carbon_tax_map=carbon_tax_map, trans_map=trans_map,
                       border_delay_map=border_delay_map)

    n_nc = len(nc_individuals)

    # ── 初始种群 ──────────────────────────────────────────────────────────
    population = list(nc_individuals)
    n_greedy   = max(1, pop_size // 3)
    for i in range(n_nc, pop_size):
        ind = (greedy_initial_individual(batches, path_lib) if i < n_greedy
               else random_initial_individual(batches, path_lib))
        repair_missing_allocations(ind, batches, path_lib)
        evaluate_individual(ind, batches, arcs, tt_dict, wc, we, **eval_kwargs)
        population.append(ind)

    # ── 构建初始 ChainNode（Level 0，nocarbon 注入点）─────────────────────
    _id_counter = [0]
    def new_id():
        _id_counter[0] += 1
        return _id_counter[0]

    # obj_tuple → ChainNode（全局追踪字典）
    all_nodes: Dict[tuple, ChainNode] = {}

    def register_node(obj, born_gen, level, label, parent_id=None) -> ChainNode:
        key = tuple(obj)
        if key in all_nodes:
            return all_nodes[key]
        node = ChainNode(node_id=new_id(), obj=key, born_gen=born_gen,
                         level=level, label=label, parent_id=parent_id)
        all_nodes[key] = node
        return node

    # 注册 nocarbon 注入点
    nc_nodes: List[ChainNode] = []
    for si, ind in enumerate(nc_individuals):
        node = register_node(ind.objectives, born_gen=0, level=0,
                             label=f"NC-Sol{si:03d}")
        nc_nodes.append(node)

    # 当前"活跃追踪集"：obj_tuple → ChainNode（只追踪还活着的）
    active_tracked: Dict[tuple, ChainNode] = {n.obj: n for n in nc_nodes}

    archive:      List[Individual] = []
    survival_log: List[Dict]       = []
    # nocarbon 注入点的存活记录（供绘图）
    nc_alive_by_gen = []

    _run_start = time.perf_counter()

    print("\n" + "━"*72)
    print("  🚀 开始 SPEA2（多级淘汰链追踪模式）")
    print("━"*72)

    for gen in range(generations):

        # ── SPEA2 环境选择 ────────────────────────────────────────────────
        combined = population + archive
        archive  = spea2_environmental_selection(combined, archive_size)

        archive_obj_set = {tuple(ind.objectives) for ind in archive}
        archive_feas    = [ind for ind in archive if ind.feasible]
        # ── NC 后代存活统计 ──────────────────────────────
        nc_children_in_archive = [
            ind for ind in archive 
            if getattr(ind, '_tag', 'BL') == f"NC_child(gen{gen-1})"
            and ind.feasible
        ]
        if nc_children_in_archive:
            print(f"\n  🧬 Gen {gen:03d}: NC血统后代进入archive "
                  f"{len(nc_children_in_archive)} 个")
            for ind in nc_children_in_archive:
                print(f"       {fmt_obj(ind.objectives)}")

        # ── 遍历所有"活跃追踪节点"，检查是否消失 ────────────────────────
        newly_eliminated: List[Tuple[tuple, ChainNode]] = []  # (obj_key, node)
        for obj_key, node in list(active_tracked.items()):
            found = any(_close(tuple(ind.objectives), obj_key) for ind in archive)
            if not found and node.still_alive:
                node.still_alive = False
                node.elim_gen    = gen
                newly_eliminated.append((obj_key, node))

        # ── 处理每一个新消失的节点 ────────────────────────────────────────
        for obj_key, node in newly_eliminated:
            dominators = [ind for ind in archive_feas if _dom(ind.objectives, obj_key)]

            if dominators:
                node.elim_reason = "dominated"
                # 为每个支配者创建子 ChainNode，加入追踪
                for dom_ind in dominators:
                    dom_obj = tuple(dom_ind.objectives)
                    child_label = f"Dom-L{node.level+1}-#{_id_counter[0]+1}"
                    child = register_node(dom_obj, born_gen=gen,
                                          level=node.level+1,
                                          label=child_label,
                                          parent_id=node.node_id)
                    node.elim_by_ids.append(child.node_id)
                    # 如果子节点还没在追踪中，加入活跃追踪
                    if dom_obj not in active_tracked and child.still_alive:
                        active_tracked[dom_obj] = child

                # 打印消失事件
                best_dom = min(dominators, key=lambda x: x.objectives[0])
                bo = best_dom.objectives
                diff_str = ", ".join(_dom_dims(obj_key, bo))
                print(f"\n  ⚠️  [{node.label}] Gen {gen:03d} 被支配淘汰")
                print(f"       被淘汰点: {fmt_obj(obj_key)}")
                print(f"       代表支配者(共{len(dominators)}个): {fmt_obj(bo)}")
                print(f"       差距: {diff_str}")

            else:
                node.elim_reason = "truncation"
                print(f"\n  ✂️  [{node.label}] Gen {gen:03d} 被密度裁剪（未被支配）")
                print(f"       被裁剪点: {fmt_obj(obj_key)}")

            # 从活跃追踪中移除（已消失）
            active_tracked.pop(obj_key, None)

        # ── 记录 nocarbon 注入点存活状态（供绘图）────────────────────────
        nc_alive = []
        for nc_node in nc_nodes:
            alive = any(_close(tuple(ind.objectives), nc_node.obj) for ind in archive)
            nc_alive.append(1 if alive else 0)
        nc_alive_by_gen.append({"gen": gen, **{f"sol{i}": nc_alive[i]
                                                for i in range(n_nc)}})

        # ── 进度打印 ──────────────────────────────────────────────────────
        interval = 5 if gen < 30 else 20
        if gen % interval == 0 or gen == generations - 1:
            feas_cnt  = sum(1 for ind in archive if ind.feasible)
            alive_cnt = sum(1 for n in active_tracked.values() if n.still_alive)
            elapsed   = time.perf_counter() - _run_start
            nc_str    = "  ".join(f"Sol{i}={'✅' if nc_alive[i] else '❌'}"
                                   for i in range(n_nc))
            print(f"  Gen {gen:03d} | {nc_str} | "
                  f"archive_feas={feas_cnt}/{archive_size} | "
                  f"tracked_alive={alive_cnt} | {elapsed:.1f}s")

        # ── 繁殖 ──────────────────────────────────────────────────────────
        feas_archive = [ind for ind in archive if ind.feasible]
        if len(feas_archive) < MIN_FEASIBLE_SOLUTIONS:
            archive, _ = feasibility_boost(
                archive, batches, path_lib, tt_dict, arc_lookup,
                arcs, wc, we, eval_kwargs)

        compute_spea2_fitness(archive)
        mating_pool = [spea2_binary_tournament(archive) for _ in range(pop_size)]

        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1, p2 = random.sample(mating_pool, 2)
    
            # ── 判断亲代是否含 NC 血统 ──────────────────────
            tag1 = getattr(p1, '_tag', 'BL')
            tag2 = getattr(p2, '_tag', 'BL')
            has_nc = ('NC' in tag1) or ('NC' in tag2)
    
            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup)
            else:
                c1 = random_initial_individual(batches, path_lib)
                c2 = random_initial_individual(batches, path_lib)
    
            # ── 打标签 ──────────────────────────────────────
            if has_nc:
                c1._tag = f"NC_child(gen{gen})"
                c2._tag = f"NC_child(gen{gen})"
            else:
                c1._tag = 'BL'
                c2._tag = 'BL'
            for child in (c1, c2):
                repair_missing_allocations(child, batches, path_lib)
            if random.random() < MUTATION_RATE:
                mutate_fixed(c1, batches, path_lib, tt_dict, arc_lookup,
                             arcs, wc, we, **eval_kwargs)
            if random.random() < MUTATION_RATE:
                mutate_fixed(c2, batches, path_lib, tt_dict, arc_lookup,
                             arcs, wc, we, **eval_kwargs)
            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict, wc, we, **eval_kwargs)
            evaluate_individual(c2, batches, arcs, tt_dict, wc, we, **eval_kwargs)
            offspring.extend([c1, c2])
        population = offspring[:pop_size]

    # ── 运行结束：检查所有仍在 active_tracked 的节点最终状态 ──────────────
    final_feas = [ind for ind in archive if ind.feasible]
    compute_spea2_fitness(archive)

    for obj_key, node in active_tracked.items():
        if node.still_alive:
            # 判断是否在最终 Pareto front（spea2_fitness < 1）
            in_pareto = any(
                _close(tuple(ind.objectives), obj_key) and ind.spea2_fitness < 1.0
                for ind in archive
            )
            node.elim_reason = "survived_pareto" if in_pareto else "survived_dominated"

    # ── 最终 Pareto front 的 Time 分布分析 ──────────────────────────────
    final_pareto = [ind for ind in archive
                    if ind.feasible and ind.spea2_fitness < 1.0]

    sep = "━" * 72
    print("\n" + sep)
    print("  📐 最终 Pareto front 中的 Time 分布分析")
    print(sep)

    if final_pareto:
        times_all  = [ind.objectives[2] for ind in final_pareto]
        time_vals  = sorted(set(round(t, 1) for t in times_all))

        # nocarbon 注入点的 Time 值
        nc_times   = sorted(set(round(n.obj[2], 1) for n in nc_nodes))

        print(f"  最终 Pareto front 共 {len(final_pareto)} 个解")
        print(f"  Time 取值（去重后）: {time_vals}")
        print(f"  Time 范围: {min(times_all):.1f}h ~ {max(times_all):.1f}h")
        print(f"  nocarbon 注入点 Time: {nc_times}")

        # 检查 nocarbon 的 Time 值是否还存在于最终 Pareto 中
        surviving_nc_times = [t for t in nc_times if t in time_vals]
        vanished_nc_times  = [t for t in nc_times if t not in time_vals]

        if surviving_nc_times:
            print(f"  ✅ nocarbon Time 值仍存在于最终 Pareto: {surviving_nc_times}h")
            print(f"     → 说明 baseline 也认为这些时间点有竞争力，但换了更好的 Cost/Emission 解")
        if vanished_nc_times:
            print(f"  ❌ nocarbon Time 值已从最终 Pareto 消失: {vanished_nc_times}h")
            print(f"     → 说明 baseline 找到了在时间维度上也更优的路径，从三维上全面替代")

        # 按 Time 分组，显示每个时间段的 Cost/Emission 范围
        print(f"\n  按 Time 分组的 Pareto 解分布:")
        print(f"  {'Time(h)':<10} {'解数量':<8} {'Cost范围(M)':<24} {'Emission范围(e9g)'}")
        print(f"  {'─'*10} {'─'*8} {'─'*24} {'─'*20}")
        from collections import defaultdict
        by_time = defaultdict(list)
        for ind in final_pareto:
            t_key = round(ind.objectives[2], 1)
            by_time[t_key].append(ind)
        for t in sorted(by_time.keys()):
            grp   = by_time[t]
            costs = [x.objectives[0]/1e6 for x in grp]
            emiss = [x.objectives[1]/1e9  for x in grp]
            nc_tag = " ← nocarbon 原始时间" if t in nc_times else ""
            print(f"  {t:<10.1f} {len(grp):<8} "
                  f"{min(costs):.4f}~{max(costs):.4f}{'':>4} "
                  f"{min(emiss):.3f}~{max(emiss):.3f}{nc_tag}")
    else:
        print("  ⚠️  最终 archive 中没有可行的非支配解")

    print(sep + "\n")

    # ── 最终 NC 后代存活统计 ──────────────────────────
    final_pareto = [ind for ind in archive
                    if ind.feasible and ind.spea2_fitness < 1.0]

    nc_offspring_survived = [
        ind for ind in final_pareto
        if 'NC_child' in getattr(ind, '_tag', '')
    ]
    bl_survived = [
        ind for ind in final_pareto
        if getattr(ind, '_tag', 'BL') == 'BL'
    ]
    nc_original_survived = [
        ind for ind in final_pareto
        if 'NC_child' not in getattr(ind, '_tag', '')
        and 'NC' in getattr(ind, '_tag', '')
    ]

    print("\n" + "━"*72)
    print("  🧬 最终 Pareto front 血统统计")
    print("━"*72)
    print(f"  原始NC注入解存活:   {len(nc_original_survived)} 个")
    print(f"  NC血统后代存活:     {len(nc_offspring_survived)} 个")
    print(f"  纯Baseline血统:     {len(bl_survived)} 个")
    print(f"  合计:               {len(final_pareto)} 个")

    if nc_offspring_survived:
        print(f"\n  存活的NC后代详情:")
        for ind in nc_offspring_survived:
            print(f"    {fmt_obj(ind.objectives)}")
            t = ind.objectives[2]
            bl_times = [i.objectives[2] for i in bl_survived]
            if bl_times:
                bl_min_t = min(bl_times)
                if t < bl_min_t:
                    print(f"    → Time={t:.1f}h 低于纯BL解最小Time"
                          f"({bl_min_t:.1f}h)，落在空隙区间 ✅")

    return archive, nc_alive_by_gen, all_nodes, nc_nodes  


# ═══════════════════════════════════════════════════════════════════════════
# 4. 生成详细报告
# ═══════════════════════════════════════════════════════════════════════════

def build_chain_report(nc_nodes, all_nodes, report_lines):
    """
    对每个 nocarbon 注入点，递归打印完整的淘汰链（树形结构）。
    """
    # 构建 parent_id → children 的映射
    children_map: Dict[int, List[ChainNode]] = {}
    for node in all_nodes.values():
        if node.parent_id is not None:
            children_map.setdefault(node.parent_id, []).append(node)

    def _reason_str(node: ChainNode) -> str:
        if node.elim_reason == "survived_pareto":
            return "✅ 存活，进入最终 Pareto front"
        elif node.elim_reason == "survived_dominated":
            return "⚠️  存活于最终 archive，但仍被支配"
        elif node.elim_reason == "truncation":
            return f"✂️  Gen {node.elim_gen:03d} 被密度裁剪（未被支配）"
        elif node.elim_reason == "dominated":
            return f"❌ Gen {node.elim_gen:03d} 被支配淘汰"
        else:
            return "❓ 状态未知"

    def _print_tree(node: ChainNode, prefix="", is_last=True,
                    lines_out: List[str] = None):
        """递归打印树。"""
        connector = "└─" if is_last else "├─"
        reason    = _reason_str(node)
        line1 = f"{prefix}{connector} [{node.label}]  {reason}"
        line2 = f"{prefix}{'  ' if is_last else '│ '}   目标: {fmt_obj_short(node.obj)}"
        if node.born_gen > 0:
            line2 += f"  (Gen {node.born_gen:03d} 入场)"
        if lines_out is not None:
            lines_out.append(line1)
            lines_out.append(line2)
        print(line1)
        print(line2)

        children = children_map.get(node.node_id, [])
        # 去重：同一目标的子节点只保留一个
        seen_obj = set()
        unique_children = []
        for c in children:
            k = c.obj
            if k not in seen_obj:
                seen_obj.add(k)
                unique_children.append(c)

        # 按消失代排序（先消失的先展示）
        unique_children.sort(key=lambda x: (x.elim_gen or 9999, x.obj[0]))

        ext = "  " if is_last else "│ "
        for idx, child in enumerate(unique_children):
            _print_tree(child, prefix + ext,
                        is_last=(idx == len(unique_children)-1),
                        lines_out=lines_out)

    lines_out = []
    sep = "═" * 72
    print("\n" + sep)
    print("  📊 每个 nocarbon 注入点的完整淘汰链（树形）")
    print(sep)
    lines_out.append(sep)
    lines_out.append("  每个 nocarbon 注入点的完整淘汰链")
    lines_out.append(sep)

    for nc_node in nc_nodes:
        header = f"\n  ┌{'─'*68}┐"
        title  = f"  │  {nc_node.label}  起始目标: {fmt_obj_short(nc_node.obj)}"
        title  = title + " " * max(0, 71 - len(title)) + "│"
        footer = f"  └{'─'*68}┘"
        print(header); print(title); print(footer)
        lines_out += [header, title, footer]
        _print_tree(nc_node, prefix="  ", is_last=True, lines_out=lines_out)
        print()
        lines_out.append("")

    report_lines += lines_out
    return lines_out


def build_chain_summary(nc_nodes, all_nodes):
    """
    生成每个 nocarbon 点的「线性淘汰摘要」：
    NC → A (Gen X, Cost↓ Emis↓) → B (Gen Y, Cost↓ Emis↓) → Pareto ✅
    """
    children_map: Dict[int, List] = {}
    for node in all_nodes.values():
        if node.parent_id is not None:
            children_map.setdefault(node.parent_id, []).append(node)

    def _find_main_chain(node: ChainNode) -> List[ChainNode]:
        """沿着支配者中 Cost 最低（最具代表性）的一条路径延伸。"""
        chain = [node]
        children = children_map.get(node.node_id, [])
        if not children:
            return chain
        # 选 Cost 最低的子节点作为"主链"
        main_child = min(children, key=lambda x: x.obj[0])
        chain += _find_main_chain(main_child)
        return chain

    lines = []
    lines.append("=" * 72)
    lines.append("  淘汰链线性摘要（每个 nocarbon 点主链追踪）")
    lines.append("=" * 72)

    for nc_node in nc_nodes:
        chain = _find_main_chain(nc_node)
        lines.append(f"\n  {nc_node.label}")
        lines.append(f"  起始目标: {fmt_obj(nc_node.obj)}")
        lines.append(f"  淘汰过程（共 {len(chain)-1} 步）:")

        for i, node in enumerate(chain):
            if i == 0:
                prefix = "    Step 0 [注入]"
            else:
                prev = chain[i-1]
                diff = _dom_dims(prev.obj, node.obj)
                prefix = f"    Step {i} [Gen {node.born_gen:03d} 出现，替代上一步]"
                lines.append(f"            差距: {', '.join(diff)}")

            reason_map = {
                "survived_pareto":   "✅ 进入最终 Pareto front",
                "survived_dominated":"⚠️  存活但被支配",
                "truncation":        f"✂️  Gen {node.elim_gen} 被密度裁剪",
                "dominated":         f"❌ Gen {node.elim_gen} 被支配（继续追踪）",
                "":                  "🔄 进化中",
            }
            fate = reason_map.get(node.elim_reason, "❓")
            lines.append(f"  {prefix}")
            lines.append(f"            目标: {fmt_obj(node.obj)}")
            lines.append(f"            命运: {fate}")

        lines.append("")
    return lines


# ═══════════════════════════════════════════════════════════════════════════
# 5. 绘图（与原版相同）
# ═══════════════════════════════════════════════════════════════════════════

def plot_survival(survival_log, nc_nodes, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_nc   = len(nc_nodes)
    gens   = [r["gen"] for r in survival_log]
    colors = ["#e74c3c","#2ecc71","#3498db","#f39c12","#9b59b6","#1abc9c","#e67e22"]

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, nc_node in enumerate(nc_nodes):
        vals   = [r[f"sol{i}"] for r in survival_log]
        offset = (i - n_nc//2) * 0.03
        label  = (f"{nc_node.label}  {fmt_obj_short(nc_node.obj)}")
        ax.step(gens, [v+offset for v in vals], where="post",
                label=label, color=colors[i % len(colors)], lw=2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Alive in Archive")
    ax.set_title("NoCarbon Solutions Survival in SPEA2 Baseline Archive (Multi-Level Chain Tracking)")
    ax.set_ylim(-0.2, 1.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] → {out_path}")

def plot_pareto_comparison(final_archive, nc_nodes, out_path):
    """
    对比图：nocarbon 注入点 vs 最终 Pareto front（二维投影，3 张子图）。
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 最终 Pareto front（feasible + spea2_fitness < 1）
    compute_spea2_fitness(final_archive)
    pareto_pts = [ind for ind in final_archive
                  if ind.feasible and ind.spea2_fitness < 1.0]

    if not pareto_pts:
        print("[WARN] 最终 archive 无可行非支配解，跳过 Pareto 对比图")
        return

    # 坐标
    px = [ind.objectives[0]/1e6 for ind in pareto_pts]   # Cost (M)
    py = [ind.objectives[1]/1e9 for ind in pareto_pts]   # Emission (e9 g)
    pz = [ind.objectives[2]     for ind in pareto_pts]   # Time (h)

    # nocarbon 注入点坐标
    nx = [n.obj[0]/1e6 for n in nc_nodes]
    ny = [n.obj[1]/1e9 for n in nc_nodes]
    nz = [n.obj[2]     for n in nc_nodes]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    proj = [
        (px, py, nx, ny, "Cost (M)",       "Emission (e9 g)", "Cost vs Emission"),
        (px, pz, nx, nz, "Cost (M)",       "Time (h)",        "Cost vs Time"),
        (py, pz, ny, nz, "Emission (e9 g)","Time (h)",        "Emission vs Time"),
    ]

    for ax, (ex, ey, ncx, ncy, xlabel, ylabel, title) in zip(axes, proj):
        # 最终 Pareto front 按第一维排序后连线，直观看前沿形状
        order = sorted(range(len(ex)), key=lambda i: ex[i])
        ax.plot([ex[i] for i in order], [ey[i] for i in order],
                color="#3498db", lw=1.2, alpha=0.5, zorder=1)
        ax.scatter(ex, ey, color="#3498db", s=40, alpha=0.8,
                   label=f"Final Pareto ({len(pareto_pts)})", zorder=2)
        ax.scatter(ncx, ncy, color="#e74c3c", s=100, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=3,
                   label=f"NC inject ({len(nc_nodes)})")

        # 标注每个 nocarbon 点
        for i, nc in enumerate(nc_nodes):
            ax.annotate(nc.label.replace("NC-Sol", "NC"),
                        (ncx[i], ncy[i]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="#c0392b")

        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Pareto Front After Evolution  (★ = NoCarbon Injection Points)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Pareto 对比图 → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. 主入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    report_lines: List[str] = []
    report_lines.append("=" * 72)
    report_lines.append("  NoCarbon 解注入 SPEA2 Baseline ── 完整多级淘汰链实验报告")
    report_lines.append("=" * 72)

    # ── 加载网络 ──────────────────────────────────────────────────────────
    print("[INIT] 加载网络数据...")
    (node_names, node_region, node_hold_cost, node_proc_cost,
     arcs, timetables, raw_batches,
     wc, we, carbon_tax_map, emission_factor_map,
     mode_speeds_map, trans_map, border_delay_map) = load_network_from_extended(DATA_FILE)

    eval_kwargs = dict(node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
                       carbon_tax_map=carbon_tax_map, trans_map=trans_map,
                       border_delay_map=border_delay_map)

    print("[INIT] 构建路径库...")
    tt_dict    = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)
    random.seed(SEED); np.random.seed(SEED)
    path_lib = build_path_library(node_names, node_region, arcs,
                                   raw_batches, tt_dict, arc_lookup)
    sanity_check_path_lib(raw_batches, path_lib)

    # ── 阶段 A：重建 nocarbon 解 ──────────────────────────────────────────
    print("\n[PHASE A] 从 JSON 重建 nocarbon 解...")
    report_lines.append("\n阶段 A：nocarbon 解重建与 baseline 评估")
    report_lines.append("─" * 72)
    nc_inds = load_nocarbon_individuals(
        NC_JSON, arc_lookup, raw_batches, path_lib,
        arcs, tt_dict, wc, we, eval_kwargs, report_lines)

    # ── 阶段 B：运行带追踪的 SPEA2 ───────────────────────────────────────
    print("\n[PHASE B] 运行 SPEA2（多级淘汰链追踪）...")
    report_lines.append("\n阶段 B：进化追踪（多级淘汰链）")
    report_lines.append("─" * 72)
    random.seed(SEED); np.random.seed(SEED)

    final_archive, nc_alive_by_gen, all_nodes, nc_nodes = run_spea2_full_chain(
        node_names, node_region, node_hold_cost, node_proc_cost,
        arcs, timetables, raw_batches,
        wc, we, carbon_tax_map, emission_factor_map,
        mode_speeds_map, trans_map, border_delay_map, path_lib,
        nc_individuals=nc_inds,
        pop_size=POP_SIZE, generations=GENS, archive_size=ARCHIVE,
        report_lines=report_lines,
    )

    # ── 阶段 C：生成树形报告 ──────────────────────────────────────────────
    print("\n[PHASE C] 生成完整淘汰链报告...")
    report_lines.append("\n阶段 C：完整多级淘汰链")
    report_lines.append("─" * 72)
    build_chain_report(nc_nodes, all_nodes, report_lines)

    # ── 统计信息 ──────────────────────────────────────────────────────────
    total_nodes   = len(all_nodes)
    pareto_nodes  = sum(1 for n in all_nodes.values()
                        if n.elim_reason == "survived_pareto")
    trunc_nodes   = sum(1 for n in all_nodes.values()
                        if n.elim_reason == "truncation")
    dom_nodes     = sum(1 for n in all_nodes.values()
                        if n.elim_reason == "dominated")
    max_depth     = max(n.level for n in all_nodes.values()) if all_nodes else 0

    stats = [
        "\n统计摘要",
        "─" * 72,
        f"  追踪节点总数:      {total_nodes}",
        f"  最大淘汰链深度:    {max_depth} 级",
        f"  最终进入 Pareto:   {pareto_nodes} 个节点",
        f"  被密度裁剪:        {trunc_nodes} 个节点",
        f"  被支配淘汰:        {dom_nodes} 个节点",
    ]
    for s in stats:
        print(s)
        report_lines.append(s)

    # ── 保存详细报告 ──────────────────────────────────────────────────────
    report_path = f"{OUTPUT_DIR}/injection_report_v2.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[OUTPUT] 详细报告 → {report_path}")

    # ── 保存线性摘要 ──────────────────────────────────────────────────────
    summary_lines = build_chain_summary(nc_nodes, all_nodes)
    summary_path  = f"{OUTPUT_DIR}/chain_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"[OUTPUT] 淘汰链摘要 → {summary_path}")
    print("\n" + "\n".join(summary_lines))

    # ── 保存 CSV ──────────────────────────────────────────────────────────
    csv_path   = f"{OUTPUT_DIR}/survival_log.csv"
    fieldnames = ["gen"] + [f"sol{i}" for i in range(len(nc_inds))]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nc_alive_by_gen)
    print(f"[OUTPUT] CSV → {csv_path}")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    plot_survival(nc_alive_by_gen, nc_nodes, f"{OUTPUT_DIR}/survival_plot.png")
    plot_pareto_comparison(final_archive, nc_nodes,
                           f"{OUTPUT_DIR}/pareto_comparison.png")

    print("\n[DONE] 实验完成。")