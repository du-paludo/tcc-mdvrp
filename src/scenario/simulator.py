"""Main simulation engine for dynamic vehicle routing with failures."""

from copy import deepcopy
from dataclasses import dataclass

from pathlib import Path
from typing import Callable, List, Tuple
import json

from core.entities import Depot, Customer, Route
from core.solution import Solution
from algorithms.base import MDVRPAlgorithm
from algorithms.ga_local_search import local_search, local_search_stage1_intra
from utils.config import AppConfig
from utils.results_io import save_history_log, save_reroute_result

from .event_queue import EventQueue, SimulationEvent, arrival_events_from_solution, travel_time
from .event_handlers import handle_arrival, handle_service_end, determine_fixed_next_customer, build_pending_customers_list
from .reroute_handler import (
    find_affected_route_by_broken_edge,
    calculate_wasted_distance,
    build_reroute_vehicle_payload,
    schedule_rerouted_events,
)
from .simulation_metrics import (
    extract_blocked_edges,
    extract_route_stop_events,
    find_routes_using_broken_edges,
    extract_visited_customers,
    calculate_cost_metrics,
)
from .models import FailureEvent
from .stage3_global_repair import stage3_global_cross_depot_repair
from .state import VehicleState, _normalize_edge

SIMULATION_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "simulation_logs"
UNIT_SPEED = 1.0


@dataclass(frozen=True)
class SimulationRuntimeSettings:
    """Validated simulation runtime settings resolved once from AppConfig."""

    reroute_degradation_threshold: float
    cluster_degradation_threshold: float
    local_search_max_iterations: int
    penalty_overcapacity_per_unit: float
    penalty_overtime_per_minute: float


def _build_runtime_settings(cfg: AppConfig) -> SimulationRuntimeSettings:
    """Build and sanitize runtime settings from typed application config."""
    return SimulationRuntimeSettings(
        reroute_degradation_threshold=float(cfg.simulation.reroute_degradation_threshold),
        cluster_degradation_threshold=float(cfg.simulation.cluster_degradation_threshold),
        local_search_max_iterations=max(1, int(cfg.ga.local_search_max_iterations)),
        penalty_overcapacity_per_unit=max(0.0, float(cfg.simulation.penalty_overcapacity_per_unit)),
        penalty_overtime_per_minute=max(0.0, float(cfg.simulation.penalty_overtime_per_minute)),
    )


def _path_uses_blocked_edge(
    start_node: Depot | Customer,
    customers: list[Customer],
    end_node: Depot | Customer,
    blocked_edges: set[tuple[int, int]],
) -> bool:
    if not blocked_edges:
        return False

    prev_idx = start_node.index
    for customer in customers:
        if _normalize_edge(prev_idx, customer.index) in blocked_edges:
            return True
        prev_idx = customer.index

    return _normalize_edge(prev_idx, end_node.index) in blocked_edges


def is_feasible(
    candidate_route: Route,
    original_route: Route | None = None,
    *,
    start_node: Depot | Customer | None = None,
    blocked_edges: set[tuple[int, int]] | None = None,
    blocked_edge_customers: list[Customer] | None = None,
) -> bool:
    """Validate route with blocked-edge hard rule and optional historical tolerance."""
    if start_node is not None and blocked_edges is not None:
        customers_for_blocked_check = (
            blocked_edge_customers
            if blocked_edge_customers is not None
            else candidate_route.customers
        )
        if _path_uses_blocked_edge(
            start_node,
            customers_for_blocked_check,
            candidate_route.depot,
            blocked_edges,
        ):
            return False

    return candidate_route.is_feasible(original_route=original_route)


def _resolve_current_node(state: VehicleState) -> Depot | Customer:
    return (
        state.route.depot
        if state.current_node_index == state.route.depot.index
        else state.customers_by_index.get(state.current_node_index, state.route.depot)
    )


def _clone_route(route: Route) -> Route:
    """Return a shallow route clone preserving historical waste fields."""
    return Route(
        depot=route.depot,
        customers=list(route.customers),
        wasted_duration=route.wasted_duration,
        wasted_distance=route.wasted_distance,
    )


def _resolve_stage3_target_node(
    state: VehicleState,
    broken_edge: tuple[int, int],
) -> int | None:
    """Resolve target_node as the to_idx on the blocked leg in route orientation."""
    blocked_leg = _resolve_stage3_blocked_leg(state, broken_edge)
    if blocked_leg is None:
        return None
    return blocked_leg[1]


def _resolve_stage3_blocked_leg(
    state: VehicleState,
    broken_edge: tuple[int, int],
) -> tuple[int, int] | None:
    """Resolve the oriented blocked leg (from_idx, to_idx) in the future path."""
    planned_nodes = [
        state.route.depot.index,
        *[customer.index for customer in state.route.customers],
        state.route.depot.index,
    ]
    leg_start = max(0, state.next_stop_index - 1)
    for i in range(leg_start, len(planned_nodes) - 1):
        if _normalize_edge(planned_nodes[i], planned_nodes[i + 1]) == broken_edge:
            return planned_nodes[i], planned_nodes[i + 1]
    return None


def _build_stage3_distance_matrix(
    vehicle_states: dict[int, VehicleState],
    blocked_edges: set[tuple[int, int]],
) -> list[list[float]]:
    """Build node-id-indexed matrix and apply blocked edges as infinite cost."""
    active_nodes: dict[int, Depot | Customer] = {}
    for state in vehicle_states.values():
        active_nodes[state.route.depot.index] = state.route.depot
        for customer in state.route.customers:
            active_nodes[customer.index] = customer

    if not active_nodes:
        return [[0.0]]

    size = max(active_nodes) + 1
    matrix = [[float("inf")] * size for _ in range(size)]

    for node_id in range(size):
        matrix[node_id][node_id] = 0.0

    for i, node_i in active_nodes.items():
        matrix[i][i] = 0.0
        for j, node_j in active_nodes.items():
            matrix[i][j] = travel_time(node_i, node_j)

    for node_a, node_b in blocked_edges:
        if node_a < size and node_b < size:
            matrix[node_a][node_b] = float("inf")
            matrix[node_b][node_a] = float("inf")

    return matrix


def _sync_vehicle_state_route(state: VehicleState, route: Route) -> None:
    """Sync mutable vehicle state after any route rewrite."""
    state.route = route
    state.customers_by_index = {customer.index: customer for customer in route.customers}
    state.pending_customer_ids = (
        {customer.index for customer in route.customers} - state.visited_customer_ids
    )


def _register_stage3_unserved_reason(
    customer_id: int,
    diagnostics: dict[str, int] | None,
    *,
    unserved_no_active_route_ids: set[int],
    unserved_no_route_without_broken_edge_ids: set[int],
    unserved_mixed_reason_ids: set[int],
) -> None:
    """Classify Stage-3 unserved reason from diagnostics and store exclusive buckets."""
    active_other_routes = int((diagnostics or {}).get("active_other_routes_count", 0))
    eligible_other_routes = int((diagnostics or {}).get("eligible_other_routes_count", 0))
    routes_with_open_insertion = int((diagnostics or {}).get("routes_with_open_insertion_count", 0))

    no_active_route_besides_current = active_other_routes == 0
    no_route_without_broken_edge = routes_with_open_insertion == 0
    has_mixed_context = (
        active_other_routes > 0
        and no_route_without_broken_edge
        and eligible_other_routes < active_other_routes
    )

    # Priority rule: if there are no active routes besides the blocked one,
    # classify only as "no active route" (not mixed).
    if no_active_route_besides_current:
        unserved_no_active_route_ids.add(customer_id)
        unserved_no_route_without_broken_edge_ids.discard(customer_id)
        unserved_mixed_reason_ids.discard(customer_id)
        return

    if has_mixed_context:
        unserved_mixed_reason_ids.add(customer_id)
        unserved_no_active_route_ids.discard(customer_id)
        unserved_no_route_without_broken_edge_ids.discard(customer_id)
        return

    if no_route_without_broken_edge:
        unserved_no_route_without_broken_edge_ids.add(customer_id)
        unserved_no_active_route_ids.discard(customer_id)
        unserved_mixed_reason_ids.discard(customer_id)


def _commit_stage3_winner_only(
    *,
    winner_state: VehicleState,
    vehicle_states: dict[int, VehicleState],
    current_solution: Solution,
    event_queue: EventQueue,
    current_time: float,
) -> None:
    """Apply a Stage-3 winner update and rebuild only that winner's future events."""
    winner_live_state = vehicle_states[winner_state.route_id]
    _sync_vehicle_state_route(winner_live_state, winner_state.route)
    current_solution.routes[winner_live_state.route_id - 1] = winner_live_state.route

    winner_executed_count = max(0, winner_live_state.next_stop_index - 1)
    winner_future_route = Route(
        depot=winner_live_state.route.depot,
        customers=list(winner_live_state.route.customers[winner_executed_count:]),
        wasted_duration=winner_live_state.route.wasted_duration,
        wasted_distance=winner_live_state.route.wasted_distance,
    )
    winner_current_node = _resolve_current_node(winner_live_state)
    winner_fixed_next, winner_travel_to_next = determine_fixed_next_customer(
        winner_live_state,
        on_broken_edge=False,
        current_time=current_time,
    )

    event_queue.remove_future_events_for_route(winner_live_state.route_id, current_time)
    schedule_rerouted_events(
        event_queue,
        winner_live_state.route_id,
        winner_future_route,
        winner_current_node,
        winner_current_node,
        current_time,
        winner_live_state,
        winner_fixed_next,
        winner_travel_to_next,
        stop_index_offset=winner_executed_count,
    )


def _run_cascade_drop_protocol(
    *,
    affected_route_id: int,
    affected_state: VehicleState,
    initial_future_customers: list[Customer],
    target_node: int | None,
    blocked_from_idx: int,
    blocked_edges: set[tuple[int, int]],
    stage3_distance_matrix: list[list[float]],
    vehicle_states: dict[int, VehicleState],
    current_solution: Solution,
    event_queue: EventQueue,
    current_time: float,
    forced_unserved_customer_ids: set[int],
    target_unserved_diagnostics: dict[str, int] | None,
    unserved_no_active_route_ids: set[int],
    unserved_no_route_without_broken_edge_ids: set[int],
    unserved_mixed_reason_ids: set[int],
    penalty_overcapacity_per_unit: float,
    penalty_overtime_per_minute: float,
) -> tuple[list[Customer], int, int, list[int]]:
    """
    Execute forward-scan cascade protocol after Stage-3 failure on target_node.

    Returns
    -------
    tuple[list[Customer], int, int, list[int]]
        (stabilized_future_customers, rescued_count, dropped_count, hero_route_ids)
    """
    def _first_blocked_customer_idx(customers: list[Customer]) -> tuple[int, tuple[int, int]] | None:
        """Return index of the customer at the first blocked leg destination in path order."""
        prev_idx = blocked_from_idx
        for idx, customer in enumerate(customers):
            edge = _normalize_edge(prev_idx, customer.index)
            if edge in blocked_edges:
                return idx, edge
            prev_idx = customer.index

        if customers:
            depot_edge = _normalize_edge(prev_idx, affected_state.route.depot.index)
            if depot_edge in blocked_edges:
                # If the blocked leg is to depot, drop/rescue the last customer first.
                return len(customers) - 1, depot_edge
        return None

    future_customers = list(initial_future_customers)
    rescued_count = 0
    dropped_count = 0
    hero_route_ids: list[int] = []

    # Step 1: drop the original problematic customer (target_node).
    if target_node is not None:
        target_dropped = False
        for idx, customer in enumerate(future_customers):
            if customer.index == target_node:
                future_customers.pop(idx)
                forced_unserved_customer_ids.add(customer.index)
                _register_stage3_unserved_reason(
                    customer.index,
                    target_unserved_diagnostics,
                    unserved_no_active_route_ids=unserved_no_active_route_ids,
                    unserved_no_route_without_broken_edge_ids=unserved_no_route_without_broken_edge_ids,
                    unserved_mixed_reason_ids=unserved_mixed_reason_ids,
                )
                dropped_count += 1
                target_dropped = True
                print(
                    "Stage 3 Cascade: dropped unreachable target customer "
                    f"{customer.index} from affected route {affected_route_id}."
                )
                break
        if not target_dropped:
            print(
                "Stage 3 Cascade warning: target customer was not found in donor future route."
            )

    # Step 2: repeatedly remove/rescue customer at the first blocked leg destination.
    while future_customers:
        blocked_hit = _first_blocked_customer_idx(future_customers)
        if blocked_hit is None:
            break

        blocked_idx, blocked_edge = blocked_hit
        blocked_customer = future_customers.pop(blocked_idx)
        print(
            "Stage 3 Cascade: blocked leg detected; "
            f"customer={blocked_customer.index}, edge={blocked_edge}."
        )

        # Make target discoverable by Stage 3 even after removal from donor list.
        affected_state.customers_by_index[blocked_customer.index] = blocked_customer
        cascade_target_diagnostics: dict[str, int] = {}
        winner_state = stage3_global_cross_depot_repair(
            target_node=blocked_customer.index,
            vehicle_states=vehicle_states,
            distance_matrix=stage3_distance_matrix,
            blocked_vehicle_id=affected_route_id,
            penalty_overcapacity_per_unit=penalty_overcapacity_per_unit,
            penalty_overtime_per_minute=penalty_overtime_per_minute,
            diagnostics_out=cascade_target_diagnostics,
        )
        affected_state.customers_by_index.pop(blocked_customer.index, None)

        if winner_state is not None:
            _commit_stage3_winner_only(
                winner_state=winner_state,
                vehicle_states=vehicle_states,
                current_solution=current_solution,
                event_queue=event_queue,
                current_time=current_time,
            )
            rescued_count += 1
            if winner_state.route_id not in hero_route_ids:
                hero_route_ids.append(winner_state.route_id)
            print(
                "Stage 3 Cascade: rescued customer "
                f"{blocked_customer.index} via vehicle {winner_state.route_id}."
            )
            continue

        forced_unserved_customer_ids.add(blocked_customer.index)
        _register_stage3_unserved_reason(
            blocked_customer.index,
            cascade_target_diagnostics,
            unserved_no_active_route_ids=unserved_no_active_route_ids,
            unserved_no_route_without_broken_edge_ids=unserved_no_route_without_broken_edge_ids,
            unserved_mixed_reason_ids=unserved_mixed_reason_ids,
        )
        dropped_count += 1
        print(
            "Stage 3 Cascade: no feasible rescue; "
            f"customer {blocked_customer.index} marked as unserved."
        )

    return future_customers, rescued_count, dropped_count, hero_route_ids


def reoptimize_intra_route_stage1(
    pending_customers: list[Customer],
    fixed_next_customer: Customer | None,
    reroute_start_node: Depot | Customer,
    event_start_node: Depot | Customer,
    depot: Depot,
    dist_fn: Callable[[Depot | Customer, Depot | Customer], float],
    executed_customers: list[Customer],
    historical_wasted_duration: float,
    historical_wasted_distance: float,
    original_route_cost: float,
    original_route: Route,
    reroute_degradation_threshold: float,
    blocked_edges: set[tuple[int, int]],
) -> tuple[list[Customer], Route, bool]:
    stage1_pending_customers = local_search_stage1_intra(
        customers=pending_customers,
        start_node=reroute_start_node,
        end_node=depot,
        dist_fn=dist_fn,
    )
    stage1_customers = [
        *([fixed_next_customer] if fixed_next_customer is not None else []),
        *stage1_pending_customers,
    ]

    stage1_combined_route = Route(
        depot=depot,
        customers=[*executed_customers, *stage1_customers],
        wasted_duration=historical_wasted_duration,
        wasted_distance=historical_wasted_distance,
    )
    stage1_cost = stage1_combined_route.total_distance()
    stage1_cost_limit = original_route_cost * reroute_degradation_threshold
    stage1_uses_blocked_edge = _path_uses_blocked_edge(
        event_start_node,
        stage1_customers,
        depot,
        blocked_edges,
    )
    stage1_is_feasible = is_feasible(
        stage1_combined_route,
        original_route=original_route,
        start_node=event_start_node,
        blocked_edges=blocked_edges,
        blocked_edge_customers=stage1_customers,
    )
    print(
        f"Stage 1 result route: customers={[c.index for c in stage1_customers]}, "
        f"cost={stage1_cost:.2f}, "
        f"cost_limit={stage1_cost_limit:.2f}, "
        f"duration={stage1_combined_route.total_duration():.2f}, "
        f"fixed_next_customer={fixed_next_customer.index if fixed_next_customer else None}, "
        f"uses_broken_edge={stage1_uses_blocked_edge}"
    )

    accepted = stage1_is_feasible and stage1_cost <= stage1_cost_limit
    if accepted:
        print(
            "Stage 1 accepted "
            f"(cost={stage1_cost:.2f}, "
            f"cost_limit={stage1_cost_limit:.2f}, "
            f"duration={stage1_combined_route.total_duration():.2f}, "
            f"hard_duration_limit={depot.max_duration:.2f})."
        )
    else:
        print(
            "Stage 1 rejected "
            f"(cost={stage1_cost:.2f}, "
            f"cost_limit={stage1_cost_limit:.2f}, "
            f"duration={stage1_combined_route.total_duration():.2f}, "
            f"hard_duration_limit={depot.max_duration:.2f}, "
            f"feasible={stage1_is_feasible}, "
            f"uses_broken_edge={stage1_uses_blocked_edge})."
        )

    return stage1_customers, stage1_combined_route, accepted


def reoptimize_intra_cluster(
    vehicle_states: dict[int, VehicleState],
    algorithm: MDVRPAlgorithm,
    affected_route_id: int,
    current_time: float,
    blocked_edges: set[tuple[int, int]],
    local_search_max_iterations: int,
    capacity_penalty: float,
    duration_penalty: float,
    cluster_degradation_threshold: float,
    event_start_node: Depot | Customer,
    reroute_start_time: float,
    wasted_travel_time: float,
    wasted_travel_distance: float,
    fixed_next_customer: Customer | None,
    travel_to_next: float,
    leg: tuple[int, int] | None,
    on_broken_edge: bool,
) -> tuple[dict[int, dict[str, object]] | None, dict[int, dict[str, object]] | None]:
    affected_state = vehicle_states[affected_route_id]
    depot = affected_state.route.depot

    # Scope to vehicles that belong to the same depot (cluster).
    cluster_states = [
        state
        for state in vehicle_states.values()
        if state.route.depot.index == depot.index
    ]
    if not cluster_states:
        return None, None

    def _route_cost_with_return(
        route: Route,
        current_node_local: Depot | Customer,
        wasted_distance_override: float | None = None,
    ) -> float:
        if route.customers:
            if wasted_distance_override is None:
                return route.total_distance()
            temp_route = Route(
                depot=route.depot,
                customers=list(route.customers),
                wasted_duration=route.wasted_duration,
                wasted_distance=wasted_distance_override,
            )
            return temp_route.total_distance()

        base_wasted = (
            wasted_distance_override
            if wasted_distance_override is not None
            else route.wasted_distance
        )
        if current_node_local.index == route.depot.index:
            return base_wasted
        return base_wasted + travel_time(current_node_local, route.depot)

    def _executed_prefix_duration(
        depot_local: Depot,
        executed: list[Customer],
        wasted_duration_local: float,
    ) -> float:
        """Duration already consumed before the pending optimization suffix."""
        if not executed:
            return wasted_duration_local

        travel = 0.0
        prev_node: Depot | Customer = depot_local
        for customer in executed:
            travel += travel_time(prev_node, customer)
            prev_node = customer

        service = sum(customer.service_time for customer in executed)
        return wasted_duration_local + travel + service

    # Baseline cost for the whole cluster (used by the gatekeeper).
    original_cluster_cost = 0.0
    for state in cluster_states:
        current_node_local = _resolve_current_node(state)
        if state.route_id == affected_route_id:
            original_cluster_cost += _route_cost_with_return(
                state.route,
                current_node_local,
                wasted_distance_override=state.route.wasted_distance + wasted_travel_distance,
            )
        else:
            original_cluster_cost += _route_cost_with_return(
                state.route,
                current_node_local,
            )

    # Prepare per-route pending sets and a dummy route to drain orphan customers.
    unassigned_customers: list[Customer] = []
    cluster_routes: list[list[Customer]] = []
    executed_capacity_by_route: list[float] = []
    executed_duration_by_route: list[float] = []
    executed_last_nodes: list[Depot | Customer] = []
    route_items: list[dict[str, object]] = []
    route_items_by_id: dict[int, dict[str, object]] = {}
    empty_route_items: list[dict[str, object]] = []

    for state in cluster_states:
        current_node_local = _resolve_current_node(state)
        executed_count = max(0, state.next_stop_index - 1)
        executed_customers = state.route.customers[:executed_count]

        if state.route_id == affected_route_id:
            fixed_next_local = fixed_next_customer
            travel_to_next_local = travel_to_next
            event_start_node_local = event_start_node
            reroute_start_time_local = reroute_start_time
            historical_wasted_duration = state.route.wasted_duration + wasted_travel_time
            historical_wasted_distance = state.route.wasted_distance + wasted_travel_distance
            on_broken_edge_local = on_broken_edge
            leg_local = leg
        else:
            fixed_next_local, travel_to_next_local = determine_fixed_next_customer(
                state, False, current_time
            )
            event_start_node_local = current_node_local
            reroute_start_time_local = current_time
            historical_wasted_duration = state.route.wasted_duration
            historical_wasted_distance = state.route.wasted_distance
            on_broken_edge_local = False
            leg_local = state.current_leg()

        pending_customers = build_pending_customers_list(state, fixed_next_local)

        if (
            state.route_id == affected_route_id
            and on_broken_edge_local
            and leg_local is not None
        ):
            _, to_idx = leg_local
            pending_customers = [c for c in pending_customers if c.index != to_idx]
            removed = state.customers_by_index.get(to_idx)
            if removed is not None:
                unassigned_customers.append(removed)

        route_customers: list[Customer] = []
        if fixed_next_local is not None:
            route_customers.append(fixed_next_local)
        route_customers.extend(pending_customers)

        if route_customers:
            route_item = {
                "route_id": state.route_id,
                "state": state,
                "current_node": current_node_local,
                "event_start_node": event_start_node_local,
                "reroute_start_time": reroute_start_time_local,
                "fixed_next_customer": fixed_next_local,
                "travel_to_next": travel_to_next_local,
                "executed_count": executed_count,
                "executed_customers": executed_customers,
                "wasted_duration": historical_wasted_duration,
                "wasted_distance": historical_wasted_distance,
                "route_customers": route_customers,
            }
            route_items.append(route_item)
            route_items_by_id[state.route_id] = route_item
            cluster_routes.append(route_customers)
            executed_capacity_by_route.append(sum(customer.demand for customer in executed_customers))
            executed_duration_by_route.append(
                _executed_prefix_duration(
                    state.route.depot,
                    executed_customers,
                    historical_wasted_duration,
                )
            )
            executed_last_nodes.append(
                executed_customers[-1] if executed_customers else state.route.depot
            )
        elif state.route_id == affected_route_id:
            route_item = {
                "route_id": state.route_id,
                "state": state,
                "current_node": current_node_local,
                "event_start_node": event_start_node_local,
                "reroute_start_time": reroute_start_time_local,
                "fixed_next_customer": fixed_next_local,
                "travel_to_next": travel_to_next_local,
                "executed_count": executed_count,
                "executed_customers": executed_customers,
                "wasted_duration": historical_wasted_duration,
                "wasted_distance": historical_wasted_distance,
                "route_customers": route_customers,
            }
            empty_route_items.append(route_item)
            route_items_by_id[state.route_id] = route_item

    if not route_items and not empty_route_items:
        return None, None

    if not route_items and not empty_route_items and unassigned_customers:
        print("Stage 2 rejected: no routes available to absorb unassigned customers.")
        return None, None

    if unassigned_customers:
        affected_item = route_items_by_id.get(affected_route_id)
        if affected_item is not None:
            was_empty = len(affected_item["route_customers"]) == 0
            affected_item["route_customers"].extend(unassigned_customers)
            if was_empty:
                if affected_item in empty_route_items:
                    empty_route_items.remove(affected_item)
                route_items.append(affected_item)
                cluster_routes.append(affected_item["route_customers"])
                executed_capacity_by_route.append(
                    sum(customer.demand for customer in affected_item["executed_customers"])
                )
                executed_duration_by_route.append(
                    _executed_prefix_duration(
                        affected_item["state"].route.depot,
                        affected_item["executed_customers"],
                        affected_item["wasted_duration"],
                    )
                )
                executed_last_nodes.append(
                    affected_item["executed_customers"][-1]
                    if affected_item["executed_customers"]
                    else affected_item["state"].route.depot
                )
        elif cluster_routes:
            cluster_routes[0].extend(unassigned_customers)

    unique_customers: dict[int, Customer] = {}
    for route in cluster_routes:
        for customer in route:
            unique_customers[customer.index] = customer

    optimized_routes_by_id: dict[int, list[Customer]] = {}
    if route_items:
        frozen_route_indices = {
            idx
            for idx, item in enumerate(route_items)
            if item["fixed_next_customer"] is not None
        }

        # Build local matrix with blocked edges and run VND local search.
        matrix_customers: dict[int, Customer] = dict(unique_customers)
        for node in executed_last_nodes:
            if isinstance(node, Customer):
                matrix_customers[node.index] = node

        algorithm._build_matrix([depot], list(matrix_customers.values()))
        for edge in blocked_edges:
            algorithm._set_edge_inf(*edge)

        optimized_routes = local_search(
            deepcopy(cluster_routes),
            depot,
            algorithm._dist,
            local_search_max_iterations=local_search_max_iterations,
            capacity_penalty=capacity_penalty,
            duration_penalty=duration_penalty,
            is_stage_2=True,
            frozen_route_indices=frozen_route_indices,
            executed_capacity_by_route=executed_capacity_by_route,
            executed_duration_by_route=executed_duration_by_route,
            executed_last_nodes=executed_last_nodes,
        )

        # 1:1 index mapping: with Stage-2 split disabled and empty routes
        # preserved, optimized_routes aligns with route_items by position.
        for item, opt_route in zip(route_items, optimized_routes):
            optimized_routes_by_id[item["route_id"]] = opt_route

    new_routes_by_id: dict[int, dict[str, object]] = {}
    new_cluster_cost = 0.0

    for state in cluster_states:
        item = route_items_by_id.get(state.route_id)
        if item is None:
            new_cluster_cost += _route_cost_with_return(
                state.route,
                _resolve_current_node(state),
            )
            continue

        optimized_pending = optimized_routes_by_id.get(state.route_id, [])
        combined_customers = [*item["executed_customers"], *optimized_pending]
        combined_route = Route(
            depot=state.route.depot,
            customers=combined_customers,
            wasted_duration=item["wasted_duration"],
            wasted_distance=item["wasted_distance"],
        )
        if combined_route.customers:
            new_cluster_cost += combined_route.total_distance()
        else:
            new_cluster_cost += _route_cost_with_return(
                combined_route,
                item["current_node"],
                wasted_distance_override=item["wasted_distance"],
            )
        new_routes_by_id[state.route_id] = {
            "combined_route": combined_route,
            "future_route": Route(
                depot=state.route.depot,
                customers=optimized_pending,
                wasted_duration=item["wasted_duration"],
                wasted_distance=item["wasted_distance"],
            ),
            "item": item,
        }

    # Gatekeeper: feasibility, broken edge avoidance, and cluster cost threshold.
    all_routes_feasible = True
    infeasible_diagnostics: list[str] = []
    for route_id, data in new_routes_by_id.items():
        item = data["item"]
        combined_route: Route = data["combined_route"]
        future_route: Route = data["future_route"]
        tolerance_baseline = Route(
            depot=item["state"].route.depot,
            customers=list(item["state"].route.customers),
            wasted_duration=item["wasted_duration"],
            wasted_distance=item["wasted_distance"],
        )

        uses_broken = _path_uses_blocked_edge(
            item["event_start_node"],
            future_route.customers,
            combined_route.depot,
            blocked_edges,
        )
        capacity_excess = combined_route.capacity_excess(tolerance_baseline)
        overtime_excess = combined_route.overtime_excess(tolerance_baseline)
        capacity_limit = combined_route._capacity_limit(tolerance_baseline)
        duration_limit = combined_route._duration_limit(tolerance_baseline)
        duration_limit_text = "unbounded" if duration_limit == 0 else f"{duration_limit:.2f}"

        route_is_feasible = is_feasible(
            combined_route,
            original_route=tolerance_baseline,
            start_node=item["event_start_node"],
            blocked_edges=blocked_edges,
            blocked_edge_customers=future_route.customers,
        )

        if not route_is_feasible:
            if uses_broken:
                print(f"Stage 2 rejected: route {route_id} attempts to cross a blocked edge.")

            reasons: list[str] = []
            if uses_broken:
                reasons.append("blocked_edge")
            if capacity_excess > 0:
                reasons.append(f"capacity_excess={capacity_excess:.2f}")
            if overtime_excess > 0:
                reasons.append(f"duration_excess={overtime_excess:.2f}")
            if not reasons:
                reasons.append("unknown_constraint_violation")

            infeasible_diagnostics.append(
                f"route={route_id}, start_node={item['event_start_node'].index}, "
                f"reasons={','.join(reasons)}, "
                f"demand={combined_route.total_demand():.2f}/{capacity_limit:.2f}, "
                f"duration={combined_route.total_duration():.2f}/{duration_limit_text}, "
                f"pending_customers={len(future_route.customers)}"
            )
            all_routes_feasible = False

    is_within_threshold = (
        new_cluster_cost <= original_cluster_cost * cluster_degradation_threshold
    )

    affected_data = new_routes_by_id.get(affected_route_id)
    if affected_data is None:
        print("Stage 2 rejected: affected route missing in optimized cluster.")
        return None, None

    if not all_routes_feasible:
        if infeasible_diagnostics:
            print("Stage 2 infeasible diagnostics:")
            for detail in infeasible_diagnostics:
                print(f"  - {detail}")
        print(f"Stage 2 rejected (infeasible).")
        return None, None

    if not is_within_threshold:
        print(
            "Stage 2 rejected by threshold but saved as FALLBACK "
            f"(cost={new_cluster_cost:.2f}, "
            f"limit={original_cluster_cost * cluster_degradation_threshold:.2f})."
        )
        return None, new_routes_by_id

    delta_pct = 0.0
    if original_cluster_cost > 0:
        delta_pct = ((new_cluster_cost - original_cluster_cost) / original_cluster_cost) * 100.0
    print(
        "Stage 2 cluster cost change "
        f"(old={original_cluster_cost:.2f}, "
        f"new={new_cluster_cost:.2f}, "
        f"delta={delta_pct:+.2f}%)."
    )

    return new_routes_by_id, None


def _commit_stage2_updates(
    *,
    new_routes_by_id: dict[int, dict[str, object]],
    affected_route_id: int,
    current_solution: Solution,
    event_queue: EventQueue,
    current_time: float,
    instance_name: str,
    algorithm: MDVRPAlgorithm,
    reroute_index: int,
    broken_edge: tuple[int, int],
    wasted_travel_time: float,
    wasted_travel_distance: float,
) -> None:
    affected_data = new_routes_by_id.get(affected_route_id)
    if affected_data is None:
        print("Stage 2 commit aborted: affected route missing in optimized cluster.")
        return

    original_routes_by_id: dict[int, Route] = {
        route_id: _clone_route(data["item"]["state"].route)
        for route_id, data in new_routes_by_id.items()
    }

    for route_id, data in new_routes_by_id.items():
        current_solution.routes[route_id - 1] = data["combined_route"]
        state = data["item"]["state"]
        state.route = data["combined_route"]
        state.customers_by_index = {c.index: c for c in data["combined_route"].customers}
        state.pending_customer_ids = (
            {c.index for c in data["combined_route"].customers}
            - state.visited_customer_ids
        )
        state.next_stop_index = data["item"]["executed_count"] + 1

    ordered_route_ids = [affected_route_id] + [
        route_id
        for route_id in sorted(new_routes_by_id)
        if route_id != affected_route_id
    ]

    reroute_vehicles_payload: list[dict[str, object]] = []
    for route_id in ordered_route_ids:
        data = new_routes_by_id[route_id]
        state = data["item"]["state"]
        route_wasted_time = wasted_travel_time if route_id == affected_route_id else 0.0
        route_wasted_distance = wasted_travel_distance if route_id == affected_route_id else 0.0
        reroute_vehicles_payload.append(
            build_reroute_vehicle_payload(
                vehicle_state=state,
                original_route=original_routes_by_id[route_id],
                rerouted_route=data["future_route"],
                wasted_travel_time=route_wasted_time,
                wasted_travel_distance=route_wasted_distance,
            )
        )

    time_tag = int(round(current_time * 100))
    output_path = (
        f"data/processed/results/{instance_name}_reroute_{reroute_index:03d}_"
        f"t{time_tag:06d}.json"
    )
    save_reroute_result(
        output_path=output_path,
        instance_name=instance_name,
        algorithm_name=f"{algorithm} (reroute {reroute_index})",
        solution=current_solution,
        vehicles=reroute_vehicles_payload,
        current_time_minutes=current_time,
        broken_edge=broken_edge,
        reroute_index=reroute_index,
    )
    print(f"Saved reroute result to {output_path}")

    for route_id, data in new_routes_by_id.items():
        item = data["item"]
        event_queue.remove_future_events_for_route(route_id, current_time)
        schedule_rerouted_events(
            event_queue,
            route_id,
            data["future_route"],
            item["event_start_node"],
            item["current_node"],
            item["reroute_start_time"],
            item["state"],
            item["fixed_next_customer"],
            item["travel_to_next"],
            stop_index_offset=item["executed_count"],
        )


def _commit_stage3_updates(
    *,
    winner_state: VehicleState,
    target_node: int,
    donor_repaired_customers: list[Customer],
    vehicle_states: dict[int, VehicleState],
    current_solution: Solution,
    event_queue: EventQueue,
    current_time: float,
    instance_name: str,
    algorithm: MDVRPAlgorithm,
    reroute_index: int,
    broken_edge: tuple[int, int],
    original_route: Route,
    routes_before_stage3: dict[int, Route],
    baseline_route_duration: float,
    event_start_node: Depot | Customer,
    current_node: Depot | Customer,
    reroute_start_time: float,
    fixed_next_customer: Customer | None,
    travel_to_next: float,
    wasted_travel_time: float,
    wasted_travel_distance: float,
    affected_route_id: int,
    cascade_hero_route_ids: list[int] | None = None,
) -> None:
    blocked_state = vehicle_states[affected_route_id]
    affected_executed_count = max(0, blocked_state.next_stop_index - 1)
    
    blocked_updated_customers = [
        *blocked_state.route.customers[:affected_executed_count],
        *donor_repaired_customers
    ]
    
    blocked_state.route = Route(
        depot=blocked_state.route.depot,
        customers=blocked_updated_customers,
        wasted_duration=blocked_state.route.wasted_duration,
        wasted_distance=blocked_state.route.wasted_distance,
    )
    
    blocked_state.customers_by_index = {
        customer.index: customer for customer in blocked_state.route.customers
    }
    blocked_state.pending_customer_ids = (
        {customer.index for customer in blocked_state.route.customers}
        - blocked_state.visited_customer_ids
    )

    winner_live_state = vehicle_states[winner_state.route_id]
    winner_live_state.route = winner_state.route
    winner_live_state.customers_by_index = {
        customer.index: customer for customer in winner_state.route.customers
    }
    winner_live_state.pending_customer_ids = (
        {customer.index for customer in winner_state.route.customers}
        - winner_live_state.visited_customer_ids
    )

    current_solution.routes[affected_route_id - 1] = blocked_state.route
    if winner_live_state.route_id != affected_route_id:
        current_solution.routes[winner_live_state.route_id - 1] = winner_live_state.route

    affected_executed_count = max(0, blocked_state.next_stop_index - 1)
    affected_future_route = Route(
        depot=blocked_state.route.depot,
        customers=list(blocked_state.route.customers[affected_executed_count:]),
        wasted_duration=blocked_state.route.wasted_duration,
        wasted_distance=blocked_state.route.wasted_distance,
    )

    blocked_old = baseline_route_duration
    blocked_new = blocked_state.route.total_duration()
    blocked_delta_pct = 0.0
    if blocked_old > 0:
        blocked_delta_pct = ((blocked_new - blocked_old) / blocked_old) * 100.0
    print(
        "Stage 3 blocked-route change "
        f"(old={blocked_old:.2f}, "
        f"new={blocked_new:.2f}, "
        f"delta={blocked_delta_pct:+.2f}%)."
    )

    reroute_vehicles_payload = [
        build_reroute_vehicle_payload(
            vehicle_state=blocked_state,
            original_route=original_route,
            rerouted_route=affected_future_route,
            wasted_travel_time=wasted_travel_time,
            wasted_travel_distance=wasted_travel_distance,
        )
    ]

    winner_future_route: Route | None = None
    winner_old = 0.0
    winner_new = 0.0
    if winner_live_state.route_id != affected_route_id:
        winner_executed_count = max(0, winner_live_state.next_stop_index - 1)
        winner_future_route = Route(
            depot=winner_live_state.route.depot,
            customers=list(winner_live_state.route.customers[winner_executed_count:]),
            wasted_duration=winner_live_state.route.wasted_duration,
            wasted_distance=winner_live_state.route.wasted_distance,
        )
        winner_original_route = routes_before_stage3[winner_live_state.route_id]
        winner_old = winner_original_route.total_duration()
        winner_new = winner_live_state.route.total_duration()
        winner_delta_pct = 0.0
        if winner_old > 0:
            winner_delta_pct = ((winner_new - winner_old) / winner_old) * 100.0
        print(
            "Stage 3 winner-route change "
            f"(old={winner_old:.2f}, "
            f"new={winner_new:.2f}, "
            f"delta={winner_delta_pct:+.2f}%)."
        )
        reroute_vehicles_payload.append(
            build_reroute_vehicle_payload(
                vehicle_state=winner_live_state,
                original_route=winner_original_route,
                rerouted_route=winner_future_route,
                wasted_travel_time=0.0,
                wasted_travel_distance=0.0,
            )
        )
    else:
        print("Stage 3 winner-route change skipped (winner is blocked vehicle).")

    # Include secondary rescue winners from cascade protocol in snapshot payload.
    if cascade_hero_route_ids:
        for hero_route_id in cascade_hero_route_ids:
            if hero_route_id in {affected_route_id, winner_live_state.route_id}:
                continue
            if hero_route_id not in vehicle_states:
                continue

            hero_live_state = vehicle_states[hero_route_id]
            hero_executed_count = max(0, hero_live_state.next_stop_index - 1)
            hero_future_route = Route(
                depot=hero_live_state.route.depot,
                customers=list(hero_live_state.route.customers[hero_executed_count:]),
                wasted_duration=hero_live_state.route.wasted_duration,
                wasted_distance=hero_live_state.route.wasted_distance,
            )
            hero_original_route = routes_before_stage3.get(hero_route_id, hero_live_state.route)
            reroute_vehicles_payload.append(
                build_reroute_vehicle_payload(
                    vehicle_state=hero_live_state,
                    original_route=hero_original_route,
                    rerouted_route=hero_future_route,
                    wasted_travel_time=0.0,
                    wasted_travel_distance=0.0,
                )
            )

    net_old = blocked_old + winner_old
    net_new = blocked_new + winner_new
    net_delta_pct = 0.0
    if net_old > 0:
        net_delta_pct = ((net_new - net_old) / net_old) * 100.0
    print(
        "Stage 3 net change (blocked+winner only) "
        f"(old={net_old:.2f}, "
        f"new={net_new:.2f}, "
        f"delta={net_delta_pct:+.2f}%)."
    )

    time_tag = int(round(current_time * 100))
    output_path = (
        f"data/processed/results/{instance_name}_reroute_{reroute_index:03d}_"
        f"t{time_tag:06d}.json"
    )
    save_reroute_result(
        output_path=output_path,
        instance_name=instance_name,
        algorithm_name=f"{algorithm} (reroute {reroute_index})",
        solution=current_solution,
        vehicles=reroute_vehicles_payload,
        current_time_minutes=current_time,
        broken_edge=broken_edge,
        reroute_index=reroute_index,
    )
    print(f"Saved reroute result to {output_path}")

    event_queue.remove_future_events_for_route(affected_route_id, current_time)
    schedule_rerouted_events(
        event_queue,
        affected_route_id,
        affected_future_route,
        event_start_node,
        current_node,
        reroute_start_time,
        blocked_state,
        fixed_next_customer,
        travel_to_next,
        stop_index_offset=affected_executed_count,
    )

    if winner_live_state.route_id != affected_route_id and winner_future_route is not None:
        winner_current_node = _resolve_current_node(winner_live_state)
        winner_fixed_next, winner_travel_to_next = determine_fixed_next_customer(
            winner_live_state,
            on_broken_edge=False,
            current_time=current_time,
        )

        event_queue.remove_future_events_for_route(winner_live_state.route_id, current_time)
        schedule_rerouted_events(
            event_queue,
            winner_live_state.route_id,
            winner_future_route,
            winner_current_node,
            winner_current_node,
            current_time,
            winner_live_state,
            winner_fixed_next,
            winner_travel_to_next,
            stop_index_offset=max(0, winner_live_state.next_stop_index - 1),
        )


def _build_vehicle_states(initial_solution: Solution) -> dict[int, VehicleState]:
    """Create one mutable VehicleState per route in the initial solution."""
    vehicle_states: dict[int, VehicleState] = {}

    for route_id, route in enumerate(initial_solution.routes, start=1):
        vehicle_states[route_id] = VehicleState(
            route_id=route_id,
            route=route,
            current_node_index=route.depot.index,
            next_stop_index=1,
            last_event_time_min=0.0,
            status="at_depot",
        )

    return vehicle_states


def run_simulation(
    initial_solution: Solution,
    failures: List[FailureEvent],
    instance_name: str,
    algorithm: MDVRPAlgorithm,
    cfg: AppConfig,
):
    """
    Run event-driven simulation with dynamic rerouting on edge failures.
    
    Parameters
    ----------
    initial_solution : Solution
        Initial routing solution.
    failures : List[FailureEvent]
        List of edge block events.
    instance_name : str
        Instance identifier for output files.
    algorithm : MDVRPAlgorithm
        Algorithm to use for rerouting.
    cfg : AppConfig
        Global typed configuration loaded by load_config().
        
    Returns
    -------
    Tuple[Solution, List[Tuple]]
        Final solution and history log.
    """
    current_solution = initial_solution
    original_solution_cost = float(initial_solution.total_cost())

    expected_customer_indices = [
        customer.index
        for route in initial_solution.routes
        for customer in route.customers
    ]

    # Initialize event queue with route events and edge failures
    event_queue = EventQueue()
    event_queue.add_events(arrival_events_from_solution(current_solution))
    for failure in failures:
        event = SimulationEvent(
            trigger_time=failure.trigger_time,
            type=failure.type,
            payload={
                "node_a": failure.node_a,
                "node_b": failure.node_b,
            },
        )
        event_queue.add_event(event)

    vehicle_states = _build_vehicle_states(current_solution)
    reroute_count = 0
    reroute_by_stage = {"stage1": 0, "stage2": 0, "stage3": 0}
    history_log = []
    total_wasted_distance = 0.0
    current_time = 0.0
    blocked_edges: set[tuple[int, int]] = set()
    forced_unserved_customer_ids: set[int] = set()
    unserved_no_active_route_ids: set[int] = set()
    unserved_no_route_without_broken_edge_ids: set[int] = set()
    unserved_mixed_reason_ids: set[int] = set()
    runtime_settings = _build_runtime_settings(cfg)

    # Main simulation loop: process events in chronological order
    while not event_queue.is_empty():
        event = event_queue.pop_next()
        if event is None:
            break

        print(f"Processing event: time={event.trigger_time:.2f}, type={event.type}, payload={event.payload}")
        current_time = event.trigger_time
        history_log.append((current_time, event.type, event.payload))

        # Dispatch event to appropriate handler
        if event.type == "arrival":
            handle_arrival(event, current_time, vehicle_states)

        elif event.type == "service_end":
            handle_service_end(event, current_time, vehicle_states)

        elif event.type == "edge_block":
            blocked_edges.add(_normalize_edge(event.payload["node_a"], event.payload["node_b"]))
            reroute_inc, wasted, accepted_stage = _handle_disaster(
                event,
                current_time,
                event_queue,
                vehicle_states,
                current_solution,
                algorithm,
                instance_name,
                reroute_count,
                blocked_edges,
                runtime_settings,
                forced_unserved_customer_ids,
                unserved_no_active_route_ids,
                unserved_no_route_without_broken_edge_ids,
                unserved_mixed_reason_ids,
            )
            reroute_count += reroute_inc
            if reroute_inc > 0 and accepted_stage in reroute_by_stage:
                reroute_by_stage[accepted_stage] += reroute_inc
            total_wasted_distance += wasted

    depot_arrival_times = [
        event_time
        for event_time, event_type, payload in history_log
        if event_type == "arrival"
        and (
            payload.get("is_return_to_depot", False)
            or payload.get("node_index") == payload.get("depot_index")
        )
    ]
    total_execution_time = max(depot_arrival_times) if depot_arrival_times else current_time

    # Save temporal history log for validation (blocked-edge checks)
    output_path = SIMULATION_LOG_DIR / f"{instance_name}_log.json"
    save_history_log(
        str(output_path),
        instance_name,
        history_log,
        expected_customer_indices=expected_customer_indices,
    )
    print(f"Saved simulation log to {output_path}")

    # Compute cost metrics: original to post-reroute to realized
    post_reroute_cost = float(current_solution.total_cost())
    reroute_cost_increase, realized_cost, total_cost_impact, _ = calculate_cost_metrics(
        original_solution_cost, post_reroute_cost, float(total_wasted_distance)
    )
    post_reroute_cost_without_wasted = post_reroute_cost - float(total_wasted_distance)
    post_reroute_delta = post_reroute_cost_without_wasted - original_solution_cost
    post_reroute_delta_pct = (
        (post_reroute_delta / original_solution_cost) * 100.0
        if original_solution_cost
        else 0.0
    )
    total_cost_impact_pct = (
        (total_cost_impact / original_solution_cost) * 100.0
        if original_solution_cost
        else 0.0
    )

    # Extract feasibility metrics from vehicle states and history
    visited = extract_visited_customers(vehicle_states)
    expected_set = set(expected_customer_indices)
    unserved_customers = sorted(list((expected_set - visited) | forced_unserved_customer_ids))
    unserved_from_stage3_fallback = sorted(
        customer_id
        for customer_id in unserved_customers
        if customer_id in forced_unserved_customer_ids
    )
    unserved_not_from_stage3_fallback = sorted(
        customer_id
        for customer_id in unserved_customers
        if customer_id not in forced_unserved_customer_ids
    )
    unserved_due_no_active_route_besides_current = sorted(
        customer_id
        for customer_id in unserved_customers
        if customer_id in unserved_no_active_route_ids
        and customer_id not in unserved_mixed_reason_ids
    )
    unserved_due_no_route_without_broken_edge = sorted(
        customer_id
        for customer_id in unserved_customers
        if customer_id in unserved_no_route_without_broken_edge_ids
        and customer_id not in unserved_mixed_reason_ids
        and customer_id not in unserved_no_active_route_ids
    )
    unserved_due_mixed_stage3_reason = sorted(
        customer_id
        for customer_id in unserved_customers
        if customer_id in unserved_mixed_reason_ids
    )
    unserved_rate_percent = (
        (len(unserved_customers) / len(expected_set)) * 100.0
        if expected_set
        else 0.0
    )
    unserved_rate_formatted = f"{unserved_rate_percent:.2f}%"

    # Check for temporal violations: routes using blocked edges after block
    blocked_edges = extract_blocked_edges(history_log)
    route_stop_events = extract_route_stop_events(history_log)
    routes_using_broken_set = find_routes_using_broken_edges(route_stop_events, blocked_edges)
    routes_using_broken = sorted(routes_using_broken_set)

    capacity_violations: list[dict[str, object]] = []
    duration_violations: list[dict[str, object]] = []
    for route_id, route in enumerate(current_solution.routes, start=1):
        route_customers_count = len(route.customers)

        capacity_limit = float(route.depot.max_capacity)
        route_demand = float(route.total_demand())
        capacity_excess = max(0.0, route_demand - capacity_limit)
        if capacity_excess > 0.0:
            capacity_violations.append(
                {
                    "route_id": route_id,
                    "depot_index": route.depot.index,
                    "customers_count": route_customers_count,
                    "total_demand": round(route_demand, 4),
                    "limit": round(capacity_limit, 4),
                    "excess": round(capacity_excess, 4),
                    "excess_percent": round((capacity_excess / capacity_limit) * 100.0, 2)
                    if capacity_limit > 0
                    else 0.0,
                }
            )

        duration_limit = float(route.depot.max_duration)
        if duration_limit > 0.0:
            route_duration = float(route.total_duration())
            duration_excess = max(0.0, route_duration - duration_limit)
            if duration_excess > 0.0:
                duration_violations.append(
                    {
                        "route_id": route_id,
                        "depot_index": route.depot.index,
                        "customers_count": route_customers_count,
                        "total_duration": round(route_duration, 4),
                        "limit": round(duration_limit, 4),
                        "excess": round(duration_excess, 4),
                        "excess_percent": round((duration_excess / duration_limit) * 100.0, 2),
                    }
                )

    capacity_feasible = len(capacity_violations) == 0
    duration_feasible = len(duration_violations) == 0
    broken_edge_feasible = len(routes_using_broken) == 0

    def _print_violation_routes(
        violations: list[dict[str, object]],
    ) -> None:
        """Print every violating route with excess/limit details."""
        if not violations:
            return
        print("  -> Violating Routes  :")
        for item in sorted(violations, key=lambda entry: int(entry["route_id"])):
            print(
                "     - "
                f"route={int(item['route_id'])}, "
                f"depot={int(item['depot_index'])}, "
                f"customers={int(item['customers_count'])}, "
                f"excess/limit={float(item['excess']):.2f}/{float(item['limit']):.2f} "
                f"({float(item['excess_percent']):+.2f}%)"
            )

    capacity_forced_customers = sum(int(item["customers_count"]) for item in capacity_violations)
    capacity_total_excess = sum(float(item["excess"]) for item in capacity_violations)
    capacity_total_limit = sum(float(item["limit"]) for item in capacity_violations)
    capacity_excess_vs_limit_pct = (
        (capacity_total_excess / capacity_total_limit) * 100.0
        if capacity_total_limit > 0
        else 0.0
    )

    duration_forced_customers = sum(int(item["customers_count"]) for item in duration_violations)
    duration_total_excess = sum(float(item["excess"]) for item in duration_violations)
    duration_total_limit = sum(float(item["limit"]) for item in duration_violations)
    duration_excess_vs_limit_pct = (
        (duration_total_excess / duration_total_limit) * 100.0
        if duration_total_limit > 0
        else 0.0
    )

    routes_feasible_now = current_solution.is_feasible()
    fleet_feasible_now = current_solution.fleet_is_feasible()
    fully_feasible_now = current_solution.fully_feasible()
    feasible_considering_broken = routes_feasible_now and broken_edge_feasible
    feasible_soft_constraints = capacity_feasible and duration_feasible
    feasible_hard_constraints = fleet_feasible_now and broken_edge_feasible
    if feasible_hard_constraints and len(unserved_customers) == 0 and feasible_soft_constraints:
        operation_verdict = "SUCCESS"
    elif feasible_hard_constraints and len(unserved_customers) == 0:
        operation_verdict = "SUCCESS WITH CONTINGENCY"
    elif feasible_hard_constraints:
        operation_verdict = "PARTIAL SUCCESS"
    else:
        operation_verdict = "FAILURE"

    # Output final simulation metrics
    print("--- Simulation summary ---")
    print(f"Original solution cost  : {original_solution_cost:.2f}")
    print(
        "Post-reroute (without embedded U-turns): "
        f"{post_reroute_cost_without_wasted:.2f} "
        f"(change: {post_reroute_delta_pct:+.2f}% | {post_reroute_delta:+.2f})"
    )
    print(f"Wasted (U-turns)        : {total_wasted_distance:.2f}")
    print(
        f"Realized total cost     : {realized_cost:.2f} "
        f"(with embedded U-turns, total impact: "
        f"{total_cost_impact_pct:+.2f}% | {total_cost_impact:+.2f})"
    )
    print(f"Reroute operations      : {reroute_count}")
    print(
        "Reroutes by stage       : "
        f"S1={reroute_by_stage['stage1']} | "
        f"S2={reroute_by_stage['stage2']} | "
        f"S3={reroute_by_stage['stage3']}"
    )
    print(
        f"Total execution time    : {total_execution_time:.2f} min "
        "(last arrival at depot)"
    )
    if unserved_customers:
        print(
            "Unserved rate/customers : "
            f"{unserved_rate_formatted} | {unserved_customers}"
        )
        print(
            "Unserved (Stage3 fallback): "
            f"{unserved_from_stage3_fallback if unserved_from_stage3_fallback else 'none'}"
        )
        print(
            "Unserved (other)        : "
            f"{unserved_not_from_stage3_fallback if unserved_not_from_stage3_fallback else 'none'}"
        )
        if unserved_due_no_active_route_besides_current:
            print(
                "Unserved (no active route besides current): "
                f"{unserved_due_no_active_route_besides_current}"
            )
        if unserved_due_no_route_without_broken_edge:
            print(
                "Unserved (no route without broken edge): "
                f"{unserved_due_no_route_without_broken_edge}"
            )
        if unserved_due_mixed_stage3_reason:
            print(
                "Unserved (mixed: no active route + no route without broken edge): "
                f"{unserved_due_mixed_stage3_reason}"
            )
    else:
        print(f"Unserved rate/customers : {unserved_rate_formatted} | none")

    print("--- Hard Constraints (Physical Limits) ---")
    broken_edge_label = "OK" if broken_edge_feasible else "VIOLATION"
    fleet_label = "OK" if fleet_feasible_now else "VIOLATION"
    system_viability_label = (
        "PERFECT (No physical laws broken)"
        if feasible_hard_constraints
        else "CRITICAL (Physical limits violated)"
    )
    print(f"Broken Edge Integrity : {broken_edge_label} ({broken_edge_feasible})")
    print(f"Fleet Limits          : {fleet_label} ({fleet_feasible_now})")
    print(f"System Viability      : {system_viability_label}")
    if routes_using_broken:
        print(f"  -> Violating Routes : {routes_using_broken}")

    print("\n--- Soft Constraints (Operational Norms) ---")
    duration_label = "OK" if duration_feasible else "CONTINGENCY ACTIVATED"
    capacity_label = "OK" if capacity_feasible else "CONTINGENCY ACTIVATED"
    print(f"Duration Compliance   : {duration_label} ({duration_feasible})")
    if not duration_feasible:
        print(
            "  -> Contingency Stats : "
            f"forced_customers={duration_forced_customers}, "
            f"excess/limit={duration_total_excess:.2f}/{duration_total_limit:.2f} "
            f"({duration_excess_vs_limit_pct:+.2f}%)"
        )
        _print_violation_routes(duration_violations)
        duration_hero_route = max(
            duration_violations,
            key=lambda item: (item["excess"], item["customers_count"], -item["route_id"]),
        )
        print(
            "  -> Critical Route    : "
            f"route={duration_hero_route['route_id']}, "
            f"depot={duration_hero_route['depot_index']}, "
            f"customers={duration_hero_route['customers_count']}"
        )

    print(f"Capacity Compliance   : {capacity_label} ({capacity_feasible})")
    if not capacity_feasible:
        print(
            "  -> Contingency Stats : "
            f"forced_customers={capacity_forced_customers}, "
            f"excess/limit={capacity_total_excess:.2f}/{capacity_total_limit:.2f} "
            f"({capacity_excess_vs_limit_pct:+.2f}%)"
        )
        _print_violation_routes(capacity_violations)
        capacity_hero_route = max(
            capacity_violations,
            key=lambda item: (item["excess"], item["customers_count"], -item["route_id"]),
        )
        print(
            "  -> Critical Route    : "
            f"route={capacity_hero_route['route_id']}, "
            f"depot={capacity_hero_route['depot_index']}, "
            f"customers={capacity_hero_route['customers_count']}"
        )

    print("\n--- Mission Final Status ---")
    print(f"Unserved Victims      : {len(unserved_customers)}")
    print(f"Operation Verdict     : {operation_verdict}")

    # Persist aggregated summary to JSON for analysis
    try:
        summary_path = SIMULATION_LOG_DIR / f"{instance_name}_summary.json"
        summary = {
            "instance": instance_name,
            "original_solution_cost": original_solution_cost,
            "post_reroute_cost": post_reroute_cost,
            "post_reroute_cost_without_wasted": post_reroute_cost_without_wasted,
            "reroute_cost_increase": reroute_cost_increase,
            "wasted_travel_distance": total_wasted_distance,
            "realized_total_cost": realized_cost,
            "total_cost_impact": total_cost_impact,
            "total_execution_time_minutes": total_execution_time,
            "reroute_count": reroute_count,
            "reroute_by_stage": {
                "stage1": reroute_by_stage["stage1"],
                "stage2": reroute_by_stage["stage2"],
                "stage3": reroute_by_stage["stage3"],
            },
            "total_customers": len(expected_set),
            "unserved_count": len(unserved_customers),
            "unserved_customers": unserved_customers,
            "unserved_stage3_fallback_count": len(unserved_from_stage3_fallback),
            "unserved_stage3_fallback_customers": unserved_from_stage3_fallback,
            "unserved_non_stage3_count": len(unserved_not_from_stage3_fallback),
            "unserved_non_stage3_customers": unserved_not_from_stage3_fallback,
            "unserved_no_active_route_besides_current_count": len(
                unserved_due_no_active_route_besides_current
            ),
            "unserved_no_active_route_besides_current_customers": (
                unserved_due_no_active_route_besides_current
            ),
            "unserved_no_route_without_broken_edge_count": len(
                unserved_due_no_route_without_broken_edge
            ),
            "unserved_no_route_without_broken_edge_customers": (
                unserved_due_no_route_without_broken_edge
            ),
            "unserved_mixed_no_active_and_no_safe_route_count": len(
                unserved_due_mixed_stage3_reason
            ),
            "unserved_mixed_no_active_and_no_safe_route_customers": (
                unserved_due_mixed_stage3_reason
            ),
            "unserved_rate_percent": round(unserved_rate_percent, 2),
            "unserved_rate": unserved_rate_formatted,
            "feasible": routes_feasible_now,
            "feasible_capacity": capacity_feasible,
            "capacity_overflow": {
                "routes_count": len(capacity_violations),
                "forced_customers_count": capacity_forced_customers,
                "total_excess": round(capacity_total_excess, 4),
                "total_limit": round(capacity_total_limit, 4),
                "excess_vs_limit_percent": round(capacity_excess_vs_limit_pct, 2),
                "routes": capacity_violations,
            },
            "feasible_duration": duration_feasible,
            "duration_overflow": {
                "routes_count": len(duration_violations),
                "forced_customers_count": duration_forced_customers,
                "total_excess": round(duration_total_excess, 4),
                "total_limit": round(duration_total_limit, 4),
                "excess_vs_limit_percent": round(duration_excess_vs_limit_pct, 2),
                "routes": duration_violations,
            },
            "feasible_soft_constraints": feasible_soft_constraints,
            "feasible_broken_edges": broken_edge_feasible,
            "fleet_feasible": fleet_feasible_now,
            "feasible_hard_constraints": feasible_hard_constraints,
            "fully_feasible": fully_feasible_now,
            "feasible_considering_broken": feasible_considering_broken,
            "routes_using_broken": routes_using_broken,
        }
        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Saved simulation summary to {summary_path}")
    except Exception:
        pass

    return current_solution, history_log


def _handle_disaster(
    event: SimulationEvent,
    current_time: float,
    event_queue: EventQueue,
    vehicle_states: dict[int, VehicleState],
    current_solution: Solution,
    algorithm: MDVRPAlgorithm,
    instance_name: str,
    reroute_count: int,
    blocked_edges: set[tuple[int, int]],
    runtime_settings: SimulationRuntimeSettings,
    forced_unserved_customer_ids: set[int],
    unserved_no_active_route_ids: set[int],
    unserved_no_route_without_broken_edge_ids: set[int],
    unserved_mixed_reason_ids: set[int],
) -> Tuple[int, float, str | None]:
    """
    Handle edge block event by finding affected vehicle and rerouting.

    Returns
    -------
    Tuple[int, float, str | None]
        (number of reroutes performed, wasted distance from U-turn, accepted stage key).
    """
    node_a = event.payload["node_a"]
    node_b = event.payload["node_b"]

    affected_route = find_affected_route_by_broken_edge(node_a, node_b, vehicle_states)
    if affected_route is None:
        return 0, 0.0, None

    affected_vehicle_state = vehicle_states[affected_route]
    original_route = affected_vehicle_state.route
    original_route_cost = original_route.total_distance()

    # Resolve vehicle position and check if traversing the broken edge now.
    current_node = (
        original_route.depot
        if affected_vehicle_state.current_node_index == original_route.depot.index
        else affected_vehicle_state.customers_by_index.get(
            affected_vehicle_state.current_node_index, original_route.depot
        )
    )
    leg = affected_vehicle_state.current_leg()
    on_broken_edge = affected_vehicle_state.is_travelling_edge(node_a, node_b)

    # Next-node commitment: keep immediate destination fixed unless this is the current broken leg.
    fixed_next_customer, travel_to_next = determine_fixed_next_customer(
        affected_vehicle_state, on_broken_edge, current_time
    )

    # U-turn exception: when currently on the broken edge, commitment is broken and wasted travel is accounted.
    wasted_travel_time, wasted_travel_distance, event_start_node, reroute_start_time = calculate_wasted_distance(
        affected_vehicle_state, current_node, on_broken_edge, leg, current_time
    )

    reroute_degradation_threshold = runtime_settings.reroute_degradation_threshold
    cluster_degradation_threshold = runtime_settings.cluster_degradation_threshold
    local_search_max_iterations = runtime_settings.local_search_max_iterations
    penalty_overcapacity_per_unit = runtime_settings.penalty_overcapacity_per_unit
    penalty_overtime_per_minute = runtime_settings.penalty_overtime_per_minute

    # Pending pool for optimization (ordered and commitment-aware).
    pending_customers = build_pending_customers_list(affected_vehicle_state, fixed_next_customer)

    broken_edge = _normalize_edge(node_a, node_b)
    reroute_start_node: Depot | Customer = (
        fixed_next_customer if fixed_next_customer is not None else event_start_node
    )

    # Build distance matrix for the local patch and fallback reroute.
    nodes_for_matrix: list[Depot | Customer] = [original_route.depot]
    if isinstance(reroute_start_node, Customer):
        nodes_for_matrix.append(reroute_start_node)
    if fixed_next_customer is not None and fixed_next_customer.index not in {
        node.index for node in nodes_for_matrix
    }:
        nodes_for_matrix.append(fixed_next_customer)

    algorithm._build_matrix(nodes_for_matrix, pending_customers)
    for blocked_edge in blocked_edges:
        algorithm._set_edge_inf(*blocked_edge)

    historical_wasted_duration = original_route.wasted_duration + wasted_travel_time
    historical_wasted_distance = original_route.wasted_distance + wasted_travel_distance

    tolerance_baseline_route = Route(
        depot=original_route.depot,
        customers=list(original_route.customers),
        wasted_duration=historical_wasted_duration,
        wasted_distance=historical_wasted_distance,
    )
    baseline_route_duration = tolerance_baseline_route.total_duration()
    accepted_stage: str | None = None
    accepted_stage_key: str | None = None
    stage3_failure_hero_route_ids: list[int] = []
    stage3_routes_snapshot: dict[int, Route] | None = None

    executed_count = max(0, affected_vehicle_state.next_stop_index - 1)
    executed_customers = original_route.customers[:executed_count]

    # Stage 1: local containment (intra-route M1/M2/M3 only).
    stage1_customers, stage1_combined_route, stage1_accepted = reoptimize_intra_route_stage1(
        pending_customers=pending_customers,
        fixed_next_customer=fixed_next_customer,
        reroute_start_node=reroute_start_node,
        event_start_node=event_start_node,
        depot=original_route.depot,
        dist_fn=algorithm._dist,
        executed_customers=executed_customers,
        historical_wasted_duration=historical_wasted_duration,
        historical_wasted_distance=historical_wasted_distance,
        original_route_cost=original_route_cost,
        original_route=tolerance_baseline_route,
        reroute_degradation_threshold=reroute_degradation_threshold,
        blocked_edges=blocked_edges,
    )

    def _stage1_fallback_is_valid() -> bool:
        is_valid = is_feasible(
            stage1_combined_route,
            original_route=original_route,
            start_node=event_start_node,
            blocked_edges=blocked_edges,
            blocked_edge_customers=stage1_customers,
        )
        uses_broken_edge = _path_uses_blocked_edge(
            event_start_node,
            stage1_customers,
            original_route.depot,
            blocked_edges,
        )
        if not is_valid or uses_broken_edge:
            print(
                "Stage 1 fallback rejected "
                f"(feasible={is_valid}, uses_broken_edge={uses_broken_edge})."
            )
            return False
        return True

    def _stage2_fallback_is_valid(
        routes_by_id: dict[int, dict[str, object]],
    ) -> bool:
        for route_id, data in routes_by_id.items():
            combined_route: Route = data["combined_route"]
            item = data["item"]
            if not is_feasible(
                combined_route,
                original_route=item["state"].route,
                start_node=item["event_start_node"],
                blocked_edges=blocked_edges,
                blocked_edge_customers=data["future_route"].customers,
            ):
                print(
                    "Stage 2 fallback rejected "
                    f"(route={route_id}, feasible=False)."
                )
                return False

            future_route: Route = data["future_route"]
            uses_broken_edge = _path_uses_blocked_edge(
                item["event_start_node"],
                future_route.customers,
                combined_route.depot,
                blocked_edges,
            )
            if uses_broken_edge:
                print(
                    "Stage 2 fallback rejected "
                    f"(route={route_id}, uses_broken_edge=True)."
                )
                return False
        return True

    if stage1_accepted:
        accepted_stage = "Stage 1"
        accepted_stage_key = "stage1"
        new_route = Route(
            depot=original_route.depot,
            customers=stage1_customers,
            wasted_duration=historical_wasted_duration,
            wasted_distance=historical_wasted_distance,
        )
    else:
        print("Reverting local patch and attempting Stage 2 intra-cluster reoptimization.")

        stage2_accepted, stage2_fallback = reoptimize_intra_cluster(
            vehicle_states=vehicle_states,
            algorithm=algorithm,
            affected_route_id=affected_route,
            current_time=current_time,
            blocked_edges=blocked_edges,
            local_search_max_iterations=10000,
            capacity_penalty=penalty_overcapacity_per_unit,
            duration_penalty=penalty_overtime_per_minute,
            cluster_degradation_threshold=cluster_degradation_threshold,
            event_start_node=event_start_node,
            reroute_start_time=reroute_start_time,
            wasted_travel_time=wasted_travel_time,
            wasted_travel_distance=wasted_travel_distance,
            fixed_next_customer=fixed_next_customer,
            travel_to_next=travel_to_next,
            leg=leg,
            on_broken_edge=on_broken_edge,
        )
        if stage2_accepted is not None:
            _commit_stage2_updates(
                new_routes_by_id=stage2_accepted,
                affected_route_id=affected_route,
                current_solution=current_solution,
                event_queue=event_queue,
                current_time=current_time,
                instance_name=instance_name,
                algorithm=algorithm,
                reroute_index=reroute_count + 1,
                broken_edge=broken_edge,
                wasted_travel_time=wasted_travel_time,
                wasted_travel_distance=wasted_travel_distance,
            )
            return 1, wasted_travel_distance, "stage2"

        print("Stage 2 rejected; proceeding to Stage 3 global cross-depot repair.")

        # Align affected vehicle historical waste before Stage 3 transfer updates.
        affected_vehicle_state.route = Route(
            depot=affected_vehicle_state.route.depot,
            customers=list(affected_vehicle_state.route.customers),
            wasted_duration=historical_wasted_duration,
            wasted_distance=historical_wasted_distance,
        )

        target_node = _resolve_stage3_target_node(affected_vehicle_state, broken_edge)
        if target_node is None:
            print(
                "Stage 3 aborted: could not resolve target_node from blocked edge; "
                "falling back to Stage 1 contingency."
            )
            if _stage1_fallback_is_valid():
                print("Using Stage 1 feasible route as contingency after Stage 3 failure.")
                accepted_stage = "Stage 3 fallback"
                accepted_stage_key = "stage1"
                new_route = Route(
                    depot=original_route.depot,
                    customers=stage1_customers,
                    wasted_duration=historical_wasted_duration,
                    wasted_distance=historical_wasted_distance,
                )
            else:
                print("Stage 3 failed and Stage 1 is infeasible; keeping original route.")
                return 0, 0.0, None
        else:
            routes_before_stage3 = {
                route_id: _clone_route(state.route)
                for route_id, state in vehicle_states.items()
            }
            stage3_routes_snapshot = routes_before_stage3
            stage3_distance_matrix = _build_stage3_distance_matrix(vehicle_states, blocked_edges)
            stage3_target_diagnostics: dict[str, int] = {}
            rescued_state = stage3_global_cross_depot_repair(
                target_node=target_node,
                vehicle_states=vehicle_states,
                distance_matrix=stage3_distance_matrix,
                blocked_vehicle_id=affected_route,
                penalty_overcapacity_per_unit=penalty_overcapacity_per_unit,
                penalty_overtime_per_minute=penalty_overtime_per_minute,
                diagnostics_out=stage3_target_diagnostics,
            )

            if rescued_state is None:
                donor_future_customers = [
                    *([fixed_next_customer] if fixed_next_customer is not None else []),
                    *pending_customers,
                ]

                print(
                    "Stage 3 failed for target customer; "
                    "activating cascade drop protocol."
                )
                stabilized_future_customers, rescued_count, dropped_count, stage3_failure_hero_route_ids = _run_cascade_drop_protocol(
                    affected_route_id=affected_route,
                    affected_state=affected_vehicle_state,
                    initial_future_customers=donor_future_customers,
                    target_node=target_node,
                    blocked_from_idx=event_start_node.index,
                    blocked_edges=blocked_edges,
                    stage3_distance_matrix=stage3_distance_matrix,
                    vehicle_states=vehicle_states,
                    current_solution=current_solution,
                    event_queue=event_queue,
                    current_time=current_time,
                    forced_unserved_customer_ids=forced_unserved_customer_ids,
                    target_unserved_diagnostics=stage3_target_diagnostics,
                    unserved_no_active_route_ids=unserved_no_active_route_ids,
                    unserved_no_route_without_broken_edge_ids=unserved_no_route_without_broken_edge_ids,
                    unserved_mixed_reason_ids=unserved_mixed_reason_ids,
                    penalty_overcapacity_per_unit=penalty_overcapacity_per_unit,
                    penalty_overtime_per_minute=penalty_overtime_per_minute,
                )

                if fixed_next_customer is not None:
                    if (
                        not stabilized_future_customers
                        or stabilized_future_customers[0].index != fixed_next_customer.index
                    ):
                        # Fixed-next commitment was removed during cascade.
                        fixed_next_customer = None
                        travel_to_next = 0.0

                accepted_stage = "Stage 3 cascade drop"
                accepted_stage_key = "stage3"
                new_route = Route(
                    depot=original_route.depot,
                    customers=stabilized_future_customers,
                    wasted_duration=historical_wasted_duration,
                    wasted_distance=historical_wasted_distance,
                )
                print(
                    "Stage 3 cascade summary: "
                    f"rescued={rescued_count}, dropped={dropped_count}, "
                    f"remaining_future={[c.index for c in stabilized_future_customers]}."
                )
            else:
                # Seed live state with the primary Stage-3 winner before cascade rescues,
                # so secondary repairs compose over the primary transfer.
                winner_live_state = vehicle_states[rescued_state.route_id]
                _sync_vehicle_state_route(winner_live_state, rescued_state.route)
                current_solution.routes[winner_live_state.route_id - 1] = winner_live_state.route

                donor_fixed_next = fixed_next_customer
                donor_travel_to_next = travel_to_next
                if donor_fixed_next is not None and donor_fixed_next.index == target_node:
                    # Prevent cloning: target moved to winner must disappear from donor prefix.
                    print(
                        "Stage 3 donor cleanup: target equals fixed_next; "
                        "removing target from donor prefix."
                    )
                    donor_fixed_next = None
                    donor_travel_to_next = 0.0

                donor_pending = [c for c in pending_customers if c.index != target_node]
                donor_start_node: Depot | Customer = (
                    donor_fixed_next if donor_fixed_next is not None else event_start_node
                )

                donor_nodes_for_matrix: list[Depot | Customer] = [original_route.depot]
                if isinstance(donor_start_node, Customer):
                    donor_nodes_for_matrix.append(donor_start_node)
                if (
                    donor_fixed_next is not None
                    and donor_fixed_next.index not in {node.index for node in donor_nodes_for_matrix}
                ):
                    donor_nodes_for_matrix.append(donor_fixed_next)

                algorithm._build_matrix(donor_nodes_for_matrix, donor_pending)
                for b_edge in blocked_edges:
                    algorithm._set_edge_inf(*b_edge)

                # Optimize donor tail after removing target from donor pool.
                donor_optimized_tail = local_search_stage1_intra(
                    customers=donor_pending,
                    start_node=donor_start_node,
                    end_node=original_route.depot,
                    dist_fn=algorithm._dist,
                )

                donor_future_customers = [
                    *([donor_fixed_next] if donor_fixed_next is not None else []),
                    *donor_optimized_tail,
                ]

                # Full-path blocked-edge cleanup with optional secondary Stage-3 rescues.
                stabilized_future_customers, rescued_count, dropped_count, cascade_hero_route_ids = _run_cascade_drop_protocol(
                    affected_route_id=affected_route,
                    affected_state=affected_vehicle_state,
                    initial_future_customers=donor_future_customers,
                    target_node=None,
                    blocked_from_idx=event_start_node.index,
                    blocked_edges=blocked_edges,
                    stage3_distance_matrix=stage3_distance_matrix,
                    vehicle_states=vehicle_states,
                    current_solution=current_solution,
                    event_queue=event_queue,
                    current_time=current_time,
                    forced_unserved_customer_ids=forced_unserved_customer_ids,
                    target_unserved_diagnostics=None,
                    unserved_no_active_route_ids=unserved_no_active_route_ids,
                    unserved_no_route_without_broken_edge_ids=unserved_no_route_without_broken_edge_ids,
                    unserved_mixed_reason_ids=unserved_mixed_reason_ids,
                    penalty_overcapacity_per_unit=penalty_overcapacity_per_unit,
                    penalty_overtime_per_minute=penalty_overtime_per_minute,
                )

                _commit_stage3_updates(
                    winner_state=vehicle_states[rescued_state.route_id],
                    target_node=target_node,
                    donor_repaired_customers=stabilized_future_customers,
                    vehicle_states=vehicle_states,
                    current_solution=current_solution,
                    event_queue=event_queue,
                    current_time=current_time,
                    instance_name=instance_name,
                    algorithm=algorithm,
                    reroute_index=reroute_count + 1,
                    broken_edge=broken_edge,
                    original_route=original_route,
                    routes_before_stage3=routes_before_stage3,
                    baseline_route_duration=baseline_route_duration,
                    event_start_node=event_start_node,
                    current_node=current_node,
                    reroute_start_time=reroute_start_time,
                    fixed_next_customer=donor_fixed_next,
                    travel_to_next=donor_travel_to_next,
                    wasted_travel_time=wasted_travel_time,
                    wasted_travel_distance=wasted_travel_distance,
                    affected_route_id=affected_route,
                    cascade_hero_route_ids=cascade_hero_route_ids,
                )

                if rescued_count > 0 or dropped_count > 0:
                    print(
                        "Stage 3 donor cascade summary: "
                        f"rescued={rescued_count}, dropped={dropped_count}."
                    )

                return 1, wasted_travel_distance, "stage3"

    # Build reroute snapshot for output JSON (executed path + future path).
    reroute_vehicles_payload = [
        build_reroute_vehicle_payload(
            vehicle_state=affected_vehicle_state,
            original_route=original_route,
            rerouted_route=new_route,
            wasted_travel_time=wasted_travel_time,
            wasted_travel_distance=wasted_travel_distance,
        )
    ]

    # In Stage-3 failure+cascade path, include hero winners that were committed in-memory.
    if stage3_failure_hero_route_ids:
        for hero_route_id in stage3_failure_hero_route_ids:
            if hero_route_id == affected_route or hero_route_id not in vehicle_states:
                continue

            hero_live_state = vehicle_states[hero_route_id]
            hero_executed_count = max(0, hero_live_state.next_stop_index - 1)
            hero_future_route = Route(
                depot=hero_live_state.route.depot,
                customers=list(hero_live_state.route.customers[hero_executed_count:]),
                wasted_duration=hero_live_state.route.wasted_duration,
                wasted_distance=hero_live_state.route.wasted_distance,
            )
            hero_original_route = (
                stage3_routes_snapshot.get(hero_route_id, hero_live_state.route)
                if stage3_routes_snapshot is not None
                else hero_live_state.route
            )
            reroute_vehicles_payload.append(
                build_reroute_vehicle_payload(
                    vehicle_state=hero_live_state,
                    original_route=hero_original_route,
                    rerouted_route=hero_future_route,
                    wasted_travel_time=0.0,
                    wasted_travel_distance=0.0,
                )
            )

    # Merge executed path (completed stops) with rerouted path (future).
    combined_customers = [*executed_customers, *new_route.customers]
    combined_route = Route(
        depot=original_route.depot,
        customers=combined_customers,
        wasted_duration=new_route.wasted_duration,
        wasted_distance=new_route.wasted_distance,
    )

    if accepted_stage is not None:
        route_delta_pct = 0.0
        if baseline_route_duration > 0:
            route_delta_pct = (
                (combined_route.total_duration() - baseline_route_duration)
                / baseline_route_duration
            ) * 100.0
        print(
            f"{accepted_stage} cost change "
            f"(old={baseline_route_duration:.2f}, "
            f"new={combined_route.total_duration():.2f}, "
            f"delta={route_delta_pct:+.2f}%)."
        )

    # Update solution with combined route.
    current_solution.routes[affected_route - 1] = combined_route

    # Sync vehicle state with new combined route (for future event processing).
    affected_vehicle_state.route = combined_route
    affected_vehicle_state.customers_by_index = {c.index: c for c in combined_route.customers}
    affected_vehicle_state.pending_customer_ids = (
        {c.index for c in combined_route.customers} - affected_vehicle_state.visited_customer_ids
    )
    affected_vehicle_state.next_stop_index = executed_count + 1

    # Save reroute result.
    reroute_index = reroute_count + 1
    time_tag = int(round(current_time * 100))
    output_path = (
        f"data/processed/results/{instance_name}_reroute_{reroute_index:03d}_"
        f"t{time_tag:06d}.json"
    )
    save_reroute_result(
        output_path=output_path,
        instance_name=instance_name,
        algorithm_name=f"{algorithm} (reroute {reroute_index})",
        solution=current_solution,
        vehicles=reroute_vehicles_payload,
        current_time_minutes=current_time,
        broken_edge=broken_edge,
        reroute_index=reroute_index,
    )
    print(f"Saved reroute result to {output_path}")

    # Remove old future events for this route and inject new ones based on reroute.
    event_queue.remove_future_events_for_route(affected_route, current_time)
    schedule_rerouted_events(
        event_queue,
        affected_route,
        new_route,
        event_start_node,
        current_node,
        reroute_start_time,
        affected_vehicle_state,
        fixed_next_customer,
        travel_to_next,
        stop_index_offset=executed_count,
    )

    final_stage_key = accepted_stage_key if accepted_stage_key is not None else "stage1"
    return 1, wasted_travel_distance, final_stage_key

