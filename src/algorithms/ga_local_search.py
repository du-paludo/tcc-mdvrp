"""
Prins (2004) local search for VRP.

Provides route-cost helpers and the 9-move local search (M1-M9) that
improves a multi-route VRP solution by scanning all O(n^2) customer pairs
and applying the first improving move found, then restarting.
"""

from typing import Callable, List

from algorithms.ga_split import linear_split
from core.entities import Customer, Depot, Route


def _route_cost(route: List[Customer], depot: Depot, dist_fn: Callable[[int, int], float]) -> float:
    """Total round-trip cost: depot -> route[0] -> ... -> route[-1] -> depot."""
    if not route:
        return 0.0
    cost = dist_fn(depot.index, route[0].index)
    for i in range(len(route) - 1):
        cost += dist_fn(route[i].index, route[i + 1].index)
    cost += dist_fn(route[-1].index, depot.index)
    return cost


def _open_path_cost(
    route: List[Customer],
    start_node: Depot | Customer,
    end_node: Depot | Customer,
    dist_fn: Callable[[int, int], float],
) -> float:
    """Path cost for start_node -> route -> end_node."""
    if not route:
        return dist_fn(start_node.index, end_node.index)

    cost = dist_fn(start_node.index, route[0].index)
    for i in range(len(route) - 1):
        cost += dist_fn(route[i].index, route[i + 1].index)
    cost += dist_fn(route[-1].index, end_node.index)
    return cost


def _penalized_route_cost(
    route: List[Customer],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
    capacity_penalty: float,
    duration_penalty: float,
    *,
    consumed_capacity: float = 0.0,
    consumed_duration: float = 0.0,
    start_node: "Depot | Customer | None" = None,
) -> float:
    """Travel cost + soft-constraint penalties, accounting for already-consumed resources."""
    if not route:
        return 0.0
    actual_start = start_node if start_node is not None else depot
    travel = _open_path_cost(route, actual_start, depot, dist_fn)
    remaining_capacity = depot.max_capacity - consumed_capacity
    cap_excess = max(0.0, sum(c.demand for c in route) - remaining_capacity)
    cost = travel + capacity_penalty * cap_excess
    if depot.max_duration > 0:
        remaining_duration = max(0.0, depot.max_duration - consumed_duration)
        service = sum(c.service_time for c in route)
        dur_excess = max(0.0, travel + service - remaining_duration)
        cost += duration_penalty * dur_excess
    return cost


def local_search_stage1_intra(
    customers: List[Customer],
    start_node: Depot | Customer,
    end_node: Depot | Customer,
    dist_fn: Callable[[int, int], float],
) -> List[Customer]:
    """
    Stage-1 disaster containment local search.

    Uses exactly three intra-route operators:
    M1: relocate, M2: swap, M3: 2-opt.
    """
    best = list(customers)
    if len(best) <= 1:
        return best

    def _duration(route: List[Customer]) -> float:
        service = sum(c.service_time for c in route)
        return _open_path_cost(route, start_node, end_node, dist_fn) + service

    best_cost = _duration(best)
    improved = True
    while improved:
        improved = False
        n = len(best)

        # M1: relocate
        for from_idx in range(n):
            if improved:
                break
            for to_idx in range(n + 1):
                if to_idx == from_idx or to_idx == from_idx + 1:
                    continue

                candidate = list(best)
                moved = candidate.pop(from_idx)
                insert_idx = to_idx if to_idx <= from_idx else to_idx - 1
                candidate.insert(insert_idx, moved)
                candidate_cost = _duration(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

        if improved:
            continue

        # M2: swap
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                candidate = list(best)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_cost = _duration(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

        if improved:
            continue

        # M3: 2-opt
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                candidate = list(best)
                candidate[i : j + 1] = candidate[i : j + 1][::-1]
                candidate_cost = _duration(candidate)
                if candidate_cost + 1e-9 < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True
                    break

    return best


def local_search(
    routes: List[Route | List[Customer]],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
    local_search_max_iterations: int,
    granularity: int = 0,
    capacity_penalty: float,
    duration_penalty: float,
    is_stage_2: bool = False,
    frozen_route_indices: set[int] | None = None,
    executed_capacity_by_route: List[float] | None = None,
    executed_duration_by_route: List[float] | None = None,
    executed_last_nodes: List[Depot | Customer] | None = None,
) -> List[List[Customer]]:
    """
    Prins (2004) 9-move local search over a multi-route VRP solution.

    Operates on mutable lists of customers (routes).  The depot is implicit
    at the start and end of every route.  Scans all O(n^2) (u, v) customer
    pairs and applies the first improving move found, then restarts.  Moves
    M1-M6 are inter/intra-route relocate/swap moves; M7 is intra-route 2-opt;
    M8-M9 are inter-route 2-opt variants.

    Empty routes are removed at the end.

    Parameters
    ----------
    routes:
        Mutable list-of-lists; modified in-place but also returned for
        convenience.
    depot:
        Shared depot for all routes (capacity / duration limits).
    dist_fn:
        O(1) pre-computed distance callable.
    is_stage_2:
        Enable VND phase control (INTER -> INTRA) for Stage 2 cluster reopt.
    frozen_route_indices:
        Route indices whose first customer must remain fixed during local search.
    executed_capacity_by_route:
        Capacity already consumed by each route prefix before optimization.
    executed_duration_by_route:
        Duration already consumed by each route prefix before optimization.
    executed_last_nodes:
        Real vehicle positions at the start of each pending suffix.
    granularity:
        Number of nearest neighbors considered for each customer (0 = all).

    Returns
    -------
    Cleaned list of non-empty routes.
    """
    # Work on copies so callers keep the originals until committed.
    normalized_routes: List[List[Customer]] = []
    for route in routes:
        if not route:
            continue
        if isinstance(route, Route):
            normalized_routes.append(list(route.customers))
        else:
            normalized_routes.append(list(route))
    routes = normalized_routes

    local_search_max_iterations = max(1, local_search_max_iterations)

    consumed_capacity = [0.0] * len(routes)
    if executed_capacity_by_route is not None:
        for idx, value in enumerate(executed_capacity_by_route[: len(routes)]):
            consumed_capacity[idx] = float(value)

    consumed_duration = [0.0] * len(routes)
    if executed_duration_by_route is not None:
        for idx, value in enumerate(executed_duration_by_route[: len(routes)]):
            consumed_duration[idx] = float(value)

    real_start_nodes: list[Depot | Customer] = [depot] * len(routes)
    if executed_last_nodes is not None:
        for idx, node in enumerate(executed_last_nodes[: len(routes)]):
            real_start_nodes[idx] = node

    # Granularity filter: pre-compute the γ nearest neighbors for each customer
    # (Prins 2004 / Vidal 2011 §4.5).  Only (u, v) pairs where v is among u's
    # γ closest customers are evaluated.  granularity=0 disables the filter.
    _neighbors: dict[int, set[int]] | None = None
    if granularity > 0:
        _all_customers = [c for route in routes for c in route]
        _neighbors = {}
        for _c in _all_customers:
            _others = [o for o in _all_customers if o is not _c]
            _others.sort(key=lambda o: dist_fn(_c.index, o.index))
            _neighbors[_c.index] = {o.index for o in _others[:granularity]}

    def _pen(route: List[Customer], route_idx: int) -> float:
        c_cap = consumed_capacity[route_idx] if route_idx < len(consumed_capacity) else 0.0
        c_dur = consumed_duration[route_idx] if route_idx < len(consumed_duration) else 0.0
        start = real_start_nodes[route_idx] if route_idx < len(real_start_nodes) else depot
        return _penalized_route_cost(
            route, depot, dist_fn, capacity_penalty, duration_penalty,
            consumed_capacity=c_cap, consumed_duration=c_dur, start_node=start,
        )

    current_phase = "INTER" if is_stage_2 else "ALL"
    improved = True
    iterations = 0
    while iterations < local_search_max_iterations:
        if not improved:
            if is_stage_2 and current_phase == "INTER":
                current_phase = "INTRA"
                improved = True
            else:
                break

        iterations += 1
        improved = False
        n_routes = len(routes)

        for ru_idx in range(n_routes):
            ru = routes[ru_idx]
            pen_ru = _pen(ru, ru_idx)
            if improved:
                break
            for rv_idx in range(n_routes):
                rv = routes[rv_idx]
                pen_rv = _pen(rv, rv_idx)
                if improved:
                    break
                for u_idx in range(len(ru)):
                    if improved:
                        break
                    if (
                        frozen_route_indices is not None
                        and ru_idx in frozen_route_indices
                        and u_idx == 0
                    ):
                        continue
                    u = ru[u_idx]
                    x = ru[u_idx + 1] if u_idx + 1 < len(ru) else depot

                    for v_idx in range(len(rv)):
                        if improved:
                            break
                        if ru_idx == rv_idx and u_idx == v_idx:
                            continue

                        is_intra = (ru_idx == rv_idx)
                        if current_phase == "INTER" and is_intra:
                            continue
                        if current_phase == "INTRA" and not is_intra:
                            continue

                        v = rv[v_idx]
                        y = rv[v_idx + 1] if v_idx + 1 < len(rv) else depot

                        # Granularity filter: skip if v is not among u's nearest neighbors.
                        if _neighbors is not None and v.index not in _neighbors.get(u.index, set()):
                            continue

                        # M1: relocate u after v
                        if not (ru_idx == rv_idx and v_idx == u_idx - 1):
                            if ru_idx == rv_idx:
                                new_r = [c for k, c in enumerate(ru) if c is not u]
                                insert_pos = next(k for k, c in enumerate(new_r) if c is v) + 1
                                new_r.insert(insert_pos, u)
                                if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break
                            else:
                                new_ru = [c for k, c in enumerate(routes[ru_idx]) if k != u_idx]
                                new_rv = list(routes[rv_idx])
                                new_rv.insert(v_idx + 1, u)
                                if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        # M2: relocate (u, x) after v
                        _m2_noop = ru_idx == rv_idx and v_idx in (u_idx - 1, u_idx + 1)
                        if isinstance(x, Customer) and not _m2_noop:
                            if ru_idx == rv_idx:
                                new_r = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                insert_pos = next(k for k, c in enumerate(new_r) if c is v) + 1
                                new_r.insert(insert_pos, x)
                                new_r.insert(insert_pos, u)
                                if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break
                            else:
                                new_ru = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                new_rv = list(rv)
                                new_rv.insert(v_idx + 1, x)
                                new_rv.insert(v_idx + 1, u)
                                if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        # M3: relocate (x, u) after v
                        _m3_noop = ru_idx == rv_idx and v_idx in (u_idx - 1, u_idx + 1)
                        if isinstance(x, Customer) and not _m3_noop:
                            if ru_idx == rv_idx:
                                new_r = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                insert_pos = next(k for k, c in enumerate(new_r) if c is v) + 1
                                new_r.insert(insert_pos, u)
                                new_r.insert(insert_pos, x)
                                if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break
                            else:
                                new_ru = [c for k, c in enumerate(ru) if k not in (u_idx, u_idx + 1)]
                                new_rv = list(rv)
                                new_rv.insert(v_idx + 1, u)
                                new_rv.insert(v_idx + 1, x)
                                if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        skip_swap = (
                            frozen_route_indices is not None
                            and rv_idx in frozen_route_indices
                            and v_idx == 0
                        )
                        if skip_swap:
                            continue

                        # M4: swap u and v
                        if ru_idx == rv_idx:
                            new_r = list(ru)
                            new_r[u_idx] = v
                            new_r[v_idx] = u
                            if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                routes[ru_idx] = new_r
                                improved = True
                                break
                        else:
                            new_ru = list(ru)
                            new_rv = list(rv)
                            new_ru[u_idx] = v
                            new_rv[v_idx] = u
                            if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                routes[ru_idx] = new_ru
                                routes[rv_idx] = new_rv
                                improved = True
                                break

                        if improved:
                            break

                        # M5: swap (u, x) with v
                        if isinstance(x, Customer) and not (ru_idx == rv_idx and v_idx == u_idx + 1):
                            if (
                                frozen_route_indices is not None
                                and rv_idx in frozen_route_indices
                                and v_idx == 0
                            ):
                                continue
                            if ru_idx == rv_idx:
                                if u_idx < v_idx:
                                    new_r = list(ru)
                                    new_r[u_idx] = v
                                    new_r.pop(u_idx + 1)
                                    adj_v = v_idx - 1
                                    new_r[adj_v] = u
                                    new_r.insert(adj_v + 1, x)
                                else:
                                    new_r = list(ru)
                                    new_r[v_idx] = u
                                    new_r.insert(v_idx + 1, x)
                                    adj_u = u_idx + 1
                                    new_r[adj_u] = v
                                    new_r.pop(adj_u + 1)
                                if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break
                            else:
                                new_ru = list(ru)
                                new_ru[u_idx] = v
                                new_ru.pop(u_idx + 1)
                                new_rv = list(rv)
                                new_rv[v_idx] = u
                                new_rv.insert(v_idx + 1, x)
                                if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        # M6: swap (u, x) with (v, y)
                        _m6_overlap = ru_idx == rv_idx and abs(v_idx - u_idx) <= 1
                        if isinstance(x, Customer) and isinstance(y, Customer) and not _m6_overlap:
                            if (
                                frozen_route_indices is not None
                                and rv_idx in frozen_route_indices
                                and v_idx == 0
                            ):
                                continue
                            if ru_idx == rv_idx:
                                new_r = list(ru)
                                new_r[u_idx] = v
                                new_r[u_idx + 1] = y
                                new_r[v_idx] = u
                                new_r[v_idx + 1] = x
                                if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                    routes[ru_idx] = new_r
                                    improved = True
                                    break
                            else:
                                new_ru = list(ru)
                                new_ru[u_idx] = v
                                new_ru[u_idx + 1] = y
                                new_rv = list(rv)
                                new_rv[v_idx] = u
                                new_rv[v_idx + 1] = x
                                if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                    routes[ru_idx] = new_ru
                                    routes[rv_idx] = new_rv
                                    improved = True
                                    break

                        if improved:
                            break

                        # M7: 2-opt within same route
                        if ru_idx == rv_idx and u_idx + 1 < v_idx:
                            new_r = list(ru)
                            new_r[u_idx + 1: v_idx + 1] = new_r[u_idx + 1: v_idx + 1][::-1]
                            if _pen(new_r, ru_idx) < pen_ru - 1e-9:
                                routes[ru_idx] = new_r
                                improved = True
                                break

                        if improved:
                            break

                        # M8: inter-route 2-opt (reconnect)
                        if ru_idx < rv_idx:
                            if (
                                frozen_route_indices is not None
                                and (
                                    ru_idx in frozen_route_indices
                                    or rv_idx in frozen_route_indices
                                )
                            ):
                                continue
                            new_ru = ru[: u_idx + 1] + rv[: v_idx + 1][::-1]
                            new_rv = ru[u_idx + 1:] + rv[v_idx + 1:]
                            if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                routes[ru_idx] = new_ru
                                routes[rv_idx] = new_rv
                                improved = True
                                break

                        if improved:
                            break

                        # M9: inter-route tail-swap
                        if ru_idx < rv_idx:
                            new_ru = ru[: u_idx + 1] + rv[v_idx + 1:]
                            new_rv = rv[: v_idx + 1] + ru[u_idx + 1:]
                            if _pen(new_ru, ru_idx) + _pen(new_rv, rv_idx) < pen_ru + pen_rv - 1e-9:
                                routes[ru_idx] = new_ru
                                routes[rv_idx] = new_rv
                                improved = True
                                break

    # Remove empty routes and return (except in Stage 2, where physical
    # vehicle index mapping is strictly required).
    if is_stage_2:
        return routes
    return [r for r in routes if r]


