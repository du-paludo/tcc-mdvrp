"""Utilities to serialize clustering and routing outputs to JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.entities import Customer, Depot, Route
from core.solution import Solution


def _solution_feasibility(solution: Solution) -> dict[str, bool]:
    return {
        "routes_feasible": solution.is_feasible(),
        "fleet_feasible": solution.fleet_is_feasible(),
        "feasible": solution.fully_feasible(),
    }


def _build_metadata(instance_name: str, algorithm_name: str) -> dict:
    return {
        "instance": instance_name,
        "algorithm": algorithm_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Invalid empty value for {field_name}.")
        try:
            return int(float(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid integer value for {field_name}: {value}") from exc
    raise ValueError(f"Unsupported value type for {field_name}: {type(value).__name__}")


def load_routing_result_as_solution(
    input_path: str | Path,
    customers: List[Customer],
    depots: List[Depot],
) -> Solution:
    """Load a routing-result JSON and build a Solution object.

    Expected format is the same emitted by save_routing_result: a top-level
    object with a ``routes`` list containing ``route_id``, ``depot_index`` and
    ``customer_indices``.
    """
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    routes_data = raw.get("routes", []) if isinstance(raw, dict) else raw
    if not isinstance(routes_data, list):
        raise ValueError("Invalid routes file format: expected a list in 'routes'.")

    customer_by_index = {customer.index: customer for customer in customers}
    depot_by_index = {depot.index: depot for depot in depots}

    loaded_routes: list[tuple[int, Route]] = []
    for idx, route_data in enumerate(routes_data, start=1):
        if not isinstance(route_data, dict):
            continue

        route_id = _as_int(route_data.get("route_id", idx), "route_id")
        depot_index = _as_int(route_data.get("depot_index"), "depot_index")

        depot = depot_by_index.get(depot_index)
        if depot is None:
            raise ValueError(
                f"Depot index {depot_index} from routes file was not found in instance data."
            )

        raw_customers = route_data.get(
            "customer_indices",
            route_data.get("customers", route_data.get("nodes", [])),
        )
        if not isinstance(raw_customers, list):
            raise ValueError(
                f"Invalid customer list in route {route_id}: expected list, got {type(raw_customers).__name__}."
            )

        route_customers: list[Customer] = []
        for value in raw_customers:
            customer_index = _as_int(value, "customer_index")
            customer = customer_by_index.get(customer_index)
            if customer is None:
                raise ValueError(
                    f"Customer index {customer_index} from route {route_id} was not found in instance data."
                )
            route_customers.append(customer)

        loaded_routes.append((route_id, Route(depot=depot, customers=route_customers)))

    if not loaded_routes:
        raise ValueError(f"No valid routes were loaded from file: {path}")

    loaded_routes.sort(key=lambda item: item[0])
    return Solution(routes=[route for _, route in loaded_routes])


def _serialize_route(route: Route, route_id: int) -> dict:
    return {
        "route_id": route_id,
        "depot_index": route.depot.index,
        "customer_indices": [c.index for c in route.customers],
        "total_demand": route.total_demand(),
        "total_distance": route.total_distance(),
        "total_duration": route.total_duration(),
        "feasible": route.is_feasible(),
    }


def save_clustering_result(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    clusters: Dict[int, list[int]],
) -> Path:
    """Save clustering artifact (customer assignment by depot) to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": _build_metadata(instance_name, algorithm_name),
        "summary": {
            "cluster_count": len(clusters),
            "customer_count": sum(len(v) for v in clusters.values()),
        },
        "clusters": [
            {
                "depot_index": depot_idx,
                "customer_indices": customer_indices,
                "customer_count": len(customer_indices),
            }
            for depot_idx, customer_indices in sorted(clusters.items())
        ],
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out


def save_routing_result(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    solution: Solution,
) -> Path:
    """Save routing artifact (final routes and metrics) to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    routes_payload = [
        _serialize_route(route, route_idx)
        for route_idx, route in enumerate(solution.routes, start=1)
    ]

    payload = {
        "metadata": _build_metadata(instance_name, algorithm_name),
        "summary": {
            "route_count": len(solution.routes),
            "total_cost": solution.total_cost(),
            **_solution_feasibility(solution),
        },
        "routes": routes_payload,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out


def save_reroute_result(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    solution: Solution,
    vehicles: List[dict[str, Any]],
    current_time_minutes: float,
    broken_edge: tuple[int, int],
    reroute_index: int,
) -> Path:
    """Save a reroute snapshot with executed/future/combined per-vehicle paths."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    realized_total_cost = sum(
        float(vehicle.get("full_route", {}).get("travel_distance", 0.0))
        for vehicle in vehicles
    )
    wasted_total_distance = sum(
        float(vehicle.get("wasted_travel_distance", 0.0))
        for vehicle in vehicles
    )
    realized_total_cost += wasted_total_distance

    payload = {
        "metadata": {
            **_build_metadata(instance_name, algorithm_name),
            "reroute_index": reroute_index,
            "current_time_minutes": current_time_minutes,
            "broken_edge": list(broken_edge),
        },
        "summary": {
            "route_count": len(solution.routes),
            "planned_total_cost": solution.total_cost(),
            "realized_total_cost": realized_total_cost,
            "wasted_travel_distance": wasted_total_distance,
            **_solution_feasibility(solution),
            "affected_vehicle_count": len(vehicles),
        },
        "routes": [
            _serialize_route(route, route_idx)
            for route_idx, route in enumerate(solution.routes, start=1)
        ],
        "vehicles": vehicles,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out


def save_clustering_and_routing(
    output_path: str,
    instance_name: str,
    algorithm_name: str,
    clusters: Dict[int, list[int]],
    solution: Solution,
) -> Path:
    """Backward-compatible combined export with both clusters and routes."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": _build_metadata(instance_name, algorithm_name),
        "summary": {
            "cluster_count": len(clusters),
            "route_count": len(solution.routes),
            "total_cost": solution.total_cost(),
            **_solution_feasibility(solution),
        },
        "clusters": [
            {
                "depot_index": depot_idx,
                "customer_indices": customer_indices,
                "customer_count": len(customer_indices),
            }
            for depot_idx, customer_indices in sorted(clusters.items())
        ],
        "routes": [
            _serialize_route(route, route_idx)
            for route_idx, route in enumerate(solution.routes, start=1)
        ],
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out

def save_history_log(
    output_path: str,
    instance_name: str,
    history_log: List[Tuple[float, str, Dict[str, Any]]],
    expected_customer_indices: List[int] | None = None,
) -> Path:
    """Save simulation event history to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Count events by type
    event_counts: Dict[str, int] = {}
    for _, event_type, _ in history_log:
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    # Serialize events
    events_payload = [
        {
            "time_minutes": time,
            "type": event_type,
            "payload": payload,
        }
        for time, event_type, payload in history_log
    ]

    # Build payload
    metadata = {
        "instance": instance_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if expected_customer_indices is not None:
        metadata["expected_customer_indices"] = sorted(expected_customer_indices)

    payload = {
        "metadata": metadata,
        "summary": {
            "total_events": len(history_log),
            "event_counts_by_type": event_counts,
            "total_time_minutes": history_log[-1][0] if history_log else 0.0,
        },
        "events": events_payload,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return out