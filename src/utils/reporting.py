"""
Console reporting utilities for the MDVRP solver.

Centralises all human-readable terminal output so that algorithm modules
and the main entry point stay focused on logic rather than formatting.

Functions
---------
print_cluster_summary   — per-depot feasibility breakdown after CCBC clustering
print_solution_summary  — per-route feasibility breakdown after GA routing
print_run_summary       — one-line cost/time/feasibility summary + GA history
"""

import math
from typing import Callable, Dict, List, Optional

from core.entities import Customer, Depot
from core.solution import Solution
from algorithms.ga_router import GADepotHistory

_SEP = "-" * 78


def print_cluster_summary(
    clusters: Dict[Depot, List[Customer]],
    dist_fn: Callable[[int, int], float],
) -> None:
    """
    Print a per-depot feasibility breakdown of the CCBC clustering result.

    Checks whether the assigned customers can theoretically be served within
    the depot's capacity and duration budgets.  Duration is assessed via a
    lower bound:

      dur_lb = Σ service_times
             + 2 × k_min × min_depot_dist   (cheapest depot round-trip legs)
             + Σ nearest_neighbour_dist / 2  (inter-customer TSP lower bound)

    If dur_lb already exceeds max_duration × max_vehicles, the cluster is
    structurally infeasible and the GA cannot fix it.
    """
    total_customers = sum(len(c) for c in clusters.values())
    print(_SEP)
    print(f"CCBC Cluster Summary  ({total_customers} customers -> {len(clusters)} depots)")
    print(_SEP)
    any_infeasible = False

    for depot, assigned in sorted(clusters.items(), key=lambda kv: kv[0].index):
        # ── capacity ──────────────────────────────────────────────────────────
        total_demand = sum(c.demand for c in assigned)
        capacity_budget = depot.max_capacity * depot.max_vehicles
        cap_pct = 100 * total_demand / capacity_budget if capacity_budget else 0.0
        cap_ok = total_demand <= capacity_budget

        # ── duration (lower bound) ─────────────────────────────────────────
        use_dur = depot.max_duration > 0
        if use_dur:
            k_min = max(1, math.ceil(total_demand / depot.max_capacity)) if depot.max_capacity > 0 else 1
            service_total = sum(c.service_time for c in assigned)
            min_depot_dist = min(dist_fn(c.index, depot.index) for c in assigned) if assigned else 0.0
            depot_leg_lb = 2 * k_min * min_depot_dist
            if len(assigned) > 1:
                nn_lb = sum(
                    min(dist_fn(c.index, o.index) for o in assigned if o is not c)
                    for c in assigned
                ) / 2
            else:
                nn_lb = 0.0
            dur_lb = service_total + depot_leg_lb + nn_lb
            duration_budget = depot.max_duration * depot.max_vehicles
            dur_pct = 100 * dur_lb / duration_budget if duration_budget else 0.0
            dur_ok = dur_lb <= duration_budget
            dur_str = (
                f" | dur_lb {dur_lb:>8.1f} / {duration_budget:.1f}"
                f" ({dur_pct:6.1f}%) [{'OK ' if dur_ok else 'OVER'}]"
            )
        else:
            dur_ok = True
            dur_str = " | duration: unconstrained"

        tags = []
        if not cap_ok:
            tags.append("!! demand exceeds vehicle budget")
            any_infeasible = True
        if not dur_ok:
            tags.append("!! duration lower bound exceeds vehicle budget")
            any_infeasible = True

        print(
            f"  Depot {depot.index:>3} | customers: {len(assigned):>4}"
            f" | demand {total_demand:>8.1f} / {capacity_budget:.1f} ({cap_pct:6.1f}%) [{'OK ' if cap_ok else 'OVER'}]"
            f"{dur_str}"
            + (f"  {', '.join(tags)}" if tags else "")
        )

    print(_SEP)
    if any_infeasible:
        print("  WARNING: one or more depots are structurally infeasible after clustering.")
        print("           The GA cannot produce feasible routes for these depots.")
        print(_SEP)


def print_solution_summary(solution: Solution) -> None:
    """Print a per-route feasibility breakdown of a solved solution."""
    report = solution.feasibility_report()

    depot_routes: Dict[int, list] = {}
    for idx, info in report.items():
        depot_routes.setdefault(info["depot"], []).append(idx)

    print(_SEP)
    print("Solution Summary")
    print(_SEP)
    for depot_idx in sorted(depot_routes):
        route_indices = depot_routes[depot_idx]
        first = report[route_indices[0]]
        used = first["routes_for_depot"]
        limit = first["max_vehicles"]
        fleet_tag = "" if first["fleet_ok"] else "  !! exceeds vehicle limit"
        print(f"  Depot {depot_idx:>3}  |  vehicles: {used}/{limit}{fleet_tag}")
        for pos, idx in enumerate(route_indices, 1):
            r = report[idx]
            cap_pct = 100 * r["demand"] / r["max_capacity"] if r["max_capacity"] else 0.0
            dur_limit = r["max_duration"]
            if dur_limit:
                dur_pct = 100 * r["duration"] / dur_limit
                dur_str = f"{r['duration']:7.2f} / {dur_limit:.2f} ({dur_pct:6.1f}%)"
            else:
                dur_str = f"{r['duration']:7.2f} / inf"
            feasible_tag = "OK        " if (r["capacity_ok"] and r["duration_ok"]) else "INFEASIBLE"
            if not r["capacity_ok"]:
                feasible_tag += " [cap]"
            if not r["duration_ok"]:
                feasible_tag += " [dur]"
            print(
                f"    Route {pos:>3} | {feasible_tag}"
                f" | demand {r['demand']:7.2f} / {r['max_capacity']:.2f} ({cap_pct:6.1f}%)"
                f" | duration {dur_str}"
            )

    infeasible_count = sum(
        1 for r in report.values() if not (r["capacity_ok"] and r["duration_ok"])
    )
    print(_SEP)
    print(
        f"  Total routes: {len(report)}"
        f"  |  infeasible: {infeasible_count}"
        f"  |  cost: {solution.total_cost():.2f}"
        f"  |  fully feasible: {solution.fully_feasible()}"
    )


def print_run_summary(
    solution: Solution,
    elapsed: float,
    ga_history: List[GADepotHistory],
    clone_delta: float,
    reference_cost: Optional[float],
    algorithm_repr: str,
    clustering_file: str,
    routing_file: str,
) -> None:
    """Print a structured cost/time/feasibility summary and per-depot GA stats."""
    cost = solution.total_cost()

    print(_SEP)
    print("Run Summary")
    print(_SEP)

    # ── Algorithm & result ────────────────────────────────────────────────────
    print(f"  Algorithm  : {algorithm_repr}")
    if reference_cost is not None:
        gap = 100 * (cost - reference_cost) / reference_cost if reference_cost else float("inf")
        gap_str = f"{gap:+.2f}%"
        print(
            f"  Cost       : {cost:.2f}"
            f"  |  Reference: {reference_cost:.2f}"
            f"  |  Gap: {gap_str}"
        )
    else:
        print(f"  Cost       : {cost:.2f}  |  Reference: N/A")

    routes_ok = solution.is_feasible()
    fleet_ok  = solution.fleet_is_feasible()
    feasible  = solution.fully_feasible()
    print(
        f"  Feasible   : {feasible}"
        f"  (routes: {routes_ok}, fleet: {fleet_ok})"
    )
    print(f"  Time       : {elapsed:.2f}s")

    # ── Per-depot GA stats ────────────────────────────────────────────────────
    if ga_history:
        print(_SEP)
        print("  GA per-depot")
        for h in ga_history:
            gens      = len(h.best)
            early_tag = f"* (stopped at gen {gens})" if h.stopped_early else f"  ({gens} gens)"
            feasible_pct = 100 * h.feasible_seen / h.total_evaluated if h.total_evaluated else 0.0
            feasible_tag = f"{h.feasible_seen}/{h.total_evaluated} ({feasible_pct:.1f}%)"
            print(
                f"  Depot {h.depot_index:>3} {early_tag:<20}"
                f" | feasible: {feasible_tag:<16}"
                f" | clones removed: {h.clones_removed} (delta={clone_delta})"
            )

    # ── Saved files ───────────────────────────────────────────────────────────
    print(_SEP)
    print(f"  Saved  clusters : {clustering_file}")
    print(f"         routes   : {routing_file}")
    print(_SEP)


def print_simulation_validation(validation_result: dict, log_path) -> None:
    """
    Print a structured simulation validation summary.

    ``validation_result`` is expected to contain:
      - ``blocked_edge_violations``: list of (route_id, node_a, node_b, depart, arrival, block)
      - ``unserved_customers``      : list of customer ids
    """
    blocked = validation_result["blocked_edge_violations"]
    unserved = validation_result["unserved_customers"]
    passed = not blocked and not unserved

    print(_SEP)
    print(f"Simulation Validation  ({'PASSED' if passed else 'FAILED'})")
    print(_SEP)

    if blocked:
        print(f"  Blocked-edge violations : {len(blocked)}")
        route_id, node_a, node_b, depart_time, arrival_time, block_time = blocked[0]
        print(
            f"    first: route {route_id} used edge {node_a} <-> {node_b}"
            f" between t={depart_time:.3f}min and t={arrival_time:.3f}min,"
            f" but it was blocked at t={block_time:.3f}min"
        )
    else:
        print("  Blocked-edge violations : none")

    if unserved:
        print(f"  Unserved customers      : {len(unserved)}  {unserved}")
    else:
        print("  Unserved customers      : none")

    print(f"  Log                     : {log_path}")
    print(_SEP)
