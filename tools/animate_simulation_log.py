#!/usr/bin/env python3
"""Standalone animator for MDVRP simulation event logs.

Features:
- Continuous simulation clock with smooth vehicle interpolation.
- Dynamic route redraw, including support for route_update-like events.
- Blocked-edge rendering with configurable visual lifetime.
- Keyboard controls:
  - space: play/pause
  - up/down: speed up/down
  - left/right: seek backward/forward 5 minutes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.data_loader import read_cordeau_data_file


EPS = 1e-9
EVENT_TYPE_RANK = {
    "edge_block": 0,
    "route_update": 1,
    "reroute": 1,
    "route_changed": 1,
    "arrival": 2,
    "service_end": 3,
}
ROUTE_UPDATE_TYPES = {"route_update", "reroute", "route_changed", "route_replanned"}


@dataclass(frozen=True)
class TimedEvent:
    time_minutes: float
    event_type: str
    payload: dict[str, Any]
    raw_index: int


@dataclass(frozen=True)
class Segment:
    kind: str  # "travel" | "service"
    start: float
    end: float
    from_node: int | None = None
    to_node: int | None = None
    node: int | None = None


@dataclass
class RoutePlan:
    route_id: int
    depot_index: int
    customers: list[int]


@dataclass
class RouteRuntime:
    plan: RoutePlan
    timeline: list[Segment]
    base_depot_index: int
    base_customers: list[int]
    visited_order: list[int] = field(default_factory=list)
    visited_set: set[int] = field(default_factory=set)
    timeline_cursor: int = 0


@dataclass(frozen=True)
class BlockedEdgeRecord:
    time_minutes: float
    node_a: int
    node_b: int


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    mode: str
    current_node: int
    target_node: int | None


@dataclass(frozen=True)
class RouteUpdate:
    route_id: int
    depot_index: int | None
    customers: list[int]
    future_only: bool


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _normalize_event_type(value: Any) -> str:
    return str(value or "").strip().lower()


def _event_time_minutes(event: dict[str, Any]) -> float | None:
    for key in ("time_minutes", "trigger_time", "time", "timestamp"):
        if key in event:
            return _as_float(event.get(key))
    return None


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        return dict(payload)

    ignore = {
        "time_minutes",
        "trigger_time",
        "time",
        "timestamp",
        "type",
        "event_type",
        "payload",
    }
    return {k: v for k, v in event.items() if k not in ignore}


def load_event_log(log_file: Path) -> tuple[dict[str, Any], list[TimedEvent]]:
    with log_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    metadata: dict[str, Any]
    raw_events: list[dict[str, Any]]

    if isinstance(raw, dict):
        metadata = dict(raw.get("metadata", {}))
        data = raw.get("events", [])
        raw_events = [event for event in data if isinstance(event, dict)]
    elif isinstance(raw, list):
        metadata = {}
        raw_events = [event for event in raw if isinstance(event, dict)]
    else:
        raise ValueError("Unsupported log format: expected JSON object or list.")

    events: list[TimedEvent] = []
    for idx, event in enumerate(raw_events):
        event_type = _normalize_event_type(event.get("type", event.get("event_type")))
        if not event_type:
            continue

        event_time = _event_time_minutes(event)
        if event_time is None:
            continue

        events.append(
            TimedEvent(
                time_minutes=float(event_time),
                event_type=event_type,
                payload=_event_payload(event),
                raw_index=idx,
            )
        )

    events.sort(
        key=lambda item: (
            item.time_minutes,
            EVENT_TYPE_RANK.get(item.event_type, 99),
            item.raw_index,
        )
    )
    return metadata, events


def load_route_plans(routes_file: Path | None) -> dict[int, RoutePlan]:
    if routes_file is None or not routes_file.exists():
        return {}

    with routes_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    routes_data = raw.get("routes", []) if isinstance(raw, dict) else raw
    if not isinstance(routes_data, list):
        return {}

    plans: dict[int, RoutePlan] = {}
    for idx, route in enumerate(routes_data, start=1):
        if not isinstance(route, dict):
            continue

        route_id = _as_int(route.get("route_id"))
        if route_id is None:
            route_id = idx

        depot_index = _as_int(route.get("depot_index"))
        if depot_index is None:
            continue

        raw_customers = route.get("customer_indices", route.get("customers", route.get("nodes", [])))
        customers = []
        if isinstance(raw_customers, list):
            for value in raw_customers:
                customer = _as_int(value)
                if customer is None or customer == depot_index:
                    continue
                customers.append(customer)

        plans[route_id] = RoutePlan(route_id=route_id, depot_index=depot_index, customers=customers)

    return plans


def enrich_plans_from_events(
    events: list[TimedEvent],
    existing: dict[int, RoutePlan],
) -> dict[int, RoutePlan]:
    plans: dict[int, RoutePlan] = {
        rid: RoutePlan(route_id=plan.route_id, depot_index=plan.depot_index, customers=list(plan.customers))
        for rid, plan in existing.items()
    }

    for event in events:
        if event.event_type != "arrival":
            continue

        route_id = _as_int(event.payload.get("route_id"))
        depot_index = _as_int(event.payload.get("depot_index"))
        node_index = _as_int(event.payload.get("node_index"))
        if route_id is None or depot_index is None:
            continue

        if route_id not in plans:
            plans[route_id] = RoutePlan(route_id=route_id, depot_index=depot_index, customers=[])

        plan = plans[route_id]
        plan.depot_index = depot_index

        if node_index is None or node_index == depot_index:
            continue
        if node_index not in plan.customers:
            plan.customers.append(node_index)

    return plans


def _to_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    parsed: list[int] = []
    for value in values:
        item = _as_int(value)
        if item is not None:
            parsed.append(item)
    return parsed


def _route_update_from_dict(data: dict[str, Any]) -> RouteUpdate | None:
    route_id = _as_int(data.get("route_id", data.get("id")))
    if route_id is None:
        return None

    depot_index = _as_int(data.get("depot_index", data.get("depot")))
    customers: list[int] = []
    future_only = False

    for key in ("customer_indices", "customers", "route"):
        customers = _to_int_list(data.get(key))
        if customers:
            break

    if not customers and isinstance(data.get("full_route"), dict):
        customers = _to_int_list(data["full_route"].get("customer_indices"))

    if not customers:
        for key in ("future_customer_indices", "pending_customer_indices", "remaining_customer_indices"):
            customers = _to_int_list(data.get(key))
            if customers:
                future_only = True
                break

    if not customers and isinstance(data.get("future_path"), dict):
        customers = _to_int_list(data["future_path"].get("customer_indices"))
        if customers:
            future_only = True

    if not customers and depot_index is None:
        return None

    return RouteUpdate(
        route_id=route_id,
        depot_index=depot_index,
        customers=customers,
        future_only=future_only,
    )


def extract_route_updates(payload: dict[str, Any]) -> list[RouteUpdate]:
    updates: list[RouteUpdate] = []

    direct = _route_update_from_dict(payload)
    if direct is not None:
        updates.append(direct)

    for key in ("routes", "vehicles"):
        collection = payload.get(key)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            parsed = _route_update_from_dict(item)
            if parsed is not None:
                updates.append(parsed)

    return updates


def build_timelines(
    events: list[TimedEvent],
    plans: dict[int, RoutePlan],
) -> dict[int, list[Segment]]:
    route_events: dict[int, list[TimedEvent]] = defaultdict(list)
    for event in events:
        if event.event_type not in {"arrival", "service_end"}:
            continue
        route_id = _as_int(event.payload.get("route_id"))
        if route_id is None:
            continue
        route_events[route_id].append(event)

    timelines: dict[int, list[Segment]] = {}
    for route_id, plan in plans.items():
        timelines[route_id] = _build_timeline_for_route(
            route_id=route_id,
            depot_index=plan.depot_index,
            route_events=route_events.get(route_id, []),
        )

    return timelines


def _build_timeline_for_route(
    route_id: int,
    depot_index: int,
    route_events: list[TimedEvent],
) -> list[Segment]:
    arrivals = [event for event in route_events if event.event_type == "arrival"]
    service_ends = [event for event in route_events if event.event_type == "service_end"]

    arrivals.sort(key=lambda event: (event.time_minutes, event.raw_index))
    service_ends.sort(key=lambda event: (event.time_minutes, event.raw_index))

    service_lookup: dict[tuple[int, int], deque[float]] = defaultdict(deque)
    for event in service_ends:
        stop_index = _as_int(event.payload.get("stop_index"))
        node_index = _as_int(event.payload.get("node_index"))
        if stop_index is None or node_index is None:
            continue
        service_lookup[(stop_index, node_index)].append(event.time_minutes)

    timeline: list[Segment] = []
    current_node = depot_index
    current_time = 0.0

    for arrival in arrivals:
        node_index = _as_int(arrival.payload.get("node_index"))
        if node_index is None:
            continue

        arrival_time = max(current_time, arrival.time_minutes)
        if arrival_time > current_time + EPS or node_index != current_node:
            timeline.append(
                Segment(
                    kind="travel",
                    start=current_time,
                    end=arrival_time,
                    from_node=current_node,
                    to_node=node_index,
                )
            )

        is_return_to_depot = bool(arrival.payload.get("is_return_to_depot", False))
        stop_index = _as_int(arrival.payload.get("stop_index"))
        service_end_time = arrival_time

        if not is_return_to_depot and node_index != depot_index:
            if stop_index is not None:
                key = (stop_index, node_index)
                times = service_lookup.get(key)
                if times is not None:
                    while times and times[0] < arrival_time - EPS:
                        times.popleft()
                    if times:
                        service_end_time = max(arrival_time, times.popleft())

            if service_end_time <= arrival_time + EPS:
                service_time = _as_float(arrival.payload.get("service_time"))
                if service_time is not None and service_time > 0.0:
                    service_end_time = arrival_time + service_time

            if service_end_time > arrival_time + EPS:
                timeline.append(
                    Segment(
                        kind="service",
                        start=arrival_time,
                        end=service_end_time,
                        node=node_index,
                    )
                )

        current_time = service_end_time
        current_node = node_index

    return timeline


def resolve_paths(
    log_file: Path,
    metadata: dict[str, Any],
    instance_file: Path | None,
    routes_file: Path | None,
) -> tuple[str, Path, Path | None]:
    instance_name = str(metadata.get("instance") or "").strip()
    if not instance_name:
        stem = log_file.stem
        instance_name = stem[:-4] if stem.endswith("_log") else stem

    if instance_file is None:
        candidate = REPO_ROOT / "data" / "raw" / "cordeau" / instance_name
        if not candidate.exists():
            raise FileNotFoundError(
                "Could not infer instance data file. Provide --instance-file explicitly."
            )
        instance_file = candidate

    if routes_file is None:
        candidate = REPO_ROOT / "data" / "processed" / "results" / f"{instance_name}_routes.json"
        if candidate.exists():
            routes_file = candidate

    return instance_name, instance_file, routes_file


class Visualizer:
    """Interactive event-log visualizer."""

    def __init__(
        self,
        instance_name: str,
        events: list[TimedEvent],
        route_runtimes: dict[int, RouteRuntime],
        positions: dict[int, tuple[float, float]],
        customer_ids: list[int],
        depot_ids: list[int],
        *,
        fps: int,
        initial_speed: float,
        blocked_edge_ttl: float,
        max_blocked_edges: int,
        show_ids: bool,
    ):
        self.instance_name = instance_name
        self.events = events
        self.route_runtimes = route_runtimes
        self.positions = positions
        self.customer_ids = customer_ids
        self.depot_ids = depot_ids
        self.fps = max(1, int(fps))
        self.speed = max(0.1, float(initial_speed))
        self.blocked_edge_ttl = max(0.1, float(blocked_edge_ttl))
        self.max_blocked_edges = max(1, int(max_blocked_edges))
        self.show_ids = show_ids

        self.total_time = max((event.time_minutes for event in events), default=0.0)
        self.sim_time = 0.0
        self.paused = False
        self.event_index = 0
        self.blocked_edges: deque[BlockedEdgeRecord] = deque()

        self._last_wall_time = time.perf_counter()

        self.route_ids = sorted(route_runtimes)
        cmap = plt.get_cmap("tab20")
        self.route_colors = {
            route_id: cmap(index % 20)
            for index, route_id in enumerate(self.route_ids)
        }

        self.fig, self.ax = plt.subplots(figsize=(13, 10))
        self.route_lines: dict[int, Any] = {}
        self.vehicle_markers: dict[int, Any] = {}
        self.vehicle_labels: dict[int, Any] = {}
        self.blocked_collection = LineCollection([], linewidths=2.2, zorder=4)
        self.status_text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        self._animation: FuncAnimation | None = None

        self._setup_axes()
        self._init_artists()
        self._apply_events_until(0.0)
        self._refresh_artists()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    @classmethod
    def from_files(
        cls,
        *,
        instance_name: str,
        instance_file: Path,
        routes_file: Path | None,
        events: list[TimedEvent],
        fps: int,
        initial_speed: float,
        blocked_edge_ttl: float,
        max_blocked_edges: int,
        show_ids: bool,
    ) -> "Visualizer":
        instance = read_cordeau_data_file(str(instance_file))

        positions: dict[int, tuple[float, float]] = {}
        customer_ids: list[int] = []
        depot_ids: list[int] = []
        for customer in instance.customers:
            positions[customer.index] = (customer.x, customer.y)
            customer_ids.append(customer.index)
        for depot in instance.depots:
            positions[depot.index] = (depot.x, depot.y)
            depot_ids.append(depot.index)

        plans = enrich_plans_from_events(events, load_route_plans(routes_file))
        plans = {
            route_id: plan
            for route_id, plan in plans.items()
            if plan.depot_index in positions
        }
        timelines = build_timelines(events, plans)

        runtimes: dict[int, RouteRuntime] = {}
        for route_id, plan in sorted(plans.items()):
            sanitized_customers = [
                customer
                for customer in plan.customers
                if customer in positions and customer != plan.depot_index
            ]
            runtime = RouteRuntime(
                plan=RoutePlan(
                    route_id=route_id,
                    depot_index=plan.depot_index,
                    customers=sanitized_customers,
                ),
                timeline=timelines.get(route_id, []),
                base_depot_index=plan.depot_index,
                base_customers=list(sanitized_customers),
            )
            runtimes[route_id] = runtime

        return cls(
            instance_name=instance_name,
            events=events,
            route_runtimes=runtimes,
            positions=positions,
            customer_ids=customer_ids,
            depot_ids=depot_ids,
            fps=fps,
            initial_speed=initial_speed,
            blocked_edge_ttl=blocked_edge_ttl,
            max_blocked_edges=max_blocked_edges,
            show_ids=show_ids,
        )

    def _setup_axes(self) -> None:
        customer_x = [self.positions[index][0] for index in self.customer_ids if index in self.positions]
        customer_y = [self.positions[index][1] for index in self.customer_ids if index in self.positions]
        depot_x = [self.positions[index][0] for index in self.depot_ids if index in self.positions]
        depot_y = [self.positions[index][1] for index in self.depot_ids if index in self.positions]

        self.ax.scatter(
            customer_x,
            customer_y,
            s=20,
            c="#4c72b0",
            alpha=0.75,
            marker="o",
            label="Customer",
            zorder=1,
        )
        self.ax.scatter(
            depot_x,
            depot_y,
            s=190,
            c="#c44e52",
            marker="*",
            edgecolors="black",
            linewidths=0.7,
            label="Depot",
            zorder=3,
        )

        self.ax.set_title(f"Simulation Log Animator - {self.instance_name}", fontsize=13)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(alpha=0.25)
        self.ax.set_aspect("equal", adjustable="box")

        all_x = customer_x + depot_x
        all_y = customer_y + depot_y
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            margin_x = max(5.0, (max_x - min_x) * 0.08)
            margin_y = max(5.0, (max_y - min_y) * 0.08)
            self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
            self.ax.set_ylim(min_y - margin_y, max_y + margin_y)

        legend_handles = [
            Line2D([0], [0], color="#555555", lw=1.5, label="Planned route"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=8, label="Vehicle en-route"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#ff8c00", markeredgecolor="black", markersize=8, label="Vehicle servicing"),
            Line2D([0], [0], color="#b22222", lw=2.2, label="Blocked edge"),
        ]
        self.ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    def _init_artists(self) -> None:
        self.ax.add_collection(self.blocked_collection)

        for route_id in self.route_ids:
            color = self.route_colors[route_id]
            line, = self.ax.plot([], [], color=color, linewidth=1.5, alpha=0.55, zorder=2)
            marker, = self.ax.plot(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor="#2ca02c",
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=6,
            )
            self.route_lines[route_id] = line
            self.vehicle_markers[route_id] = marker

            if self.show_ids:
                label = self.ax.text(
                    0.0,
                    0.0,
                    str(route_id),
                    fontsize=7,
                    color="black",
                    ha="left",
                    va="bottom",
                    zorder=7,
                )
                self.vehicle_labels[route_id] = label

    def _reset_dynamic_state(self) -> None:
        self.event_index = 0
        self.blocked_edges.clear()
        for runtime in self.route_runtimes.values():
            runtime.visited_order.clear()
            runtime.visited_set.clear()
            runtime.timeline_cursor = 0
            runtime.plan.depot_index = runtime.base_depot_index
            runtime.plan.customers = list(runtime.base_customers)

    def set_time(self, new_time: float) -> None:
        bounded_time = min(max(0.0, new_time), self.total_time)
        if bounded_time + EPS >= self.sim_time:
            self.sim_time = bounded_time
            self._apply_events_until(self.sim_time)
            return

        self._reset_dynamic_state()
        self.sim_time = bounded_time
        self._apply_events_until(self.sim_time)

    def _apply_events_until(self, target_time: float) -> None:
        while self.event_index < len(self.events):
            event = self.events[self.event_index]
            if event.time_minutes > target_time + EPS:
                break

            if event.event_type == "edge_block":
                self._handle_edge_block(event)
            elif event.event_type == "arrival":
                self._handle_arrival(event)
            elif event.event_type in ROUTE_UPDATE_TYPES:
                self._handle_route_update(event)

            self.event_index += 1

    def _handle_edge_block(self, event: TimedEvent) -> None:
        node_a = _as_int(event.payload.get("node_a"))
        node_b = _as_int(event.payload.get("node_b"))
        if node_a is None or node_b is None:
            return
        self.blocked_edges.append(BlockedEdgeRecord(event.time_minutes, node_a, node_b))

    def _handle_arrival(self, event: TimedEvent) -> None:
        route_id = _as_int(event.payload.get("route_id"))
        node_index = _as_int(event.payload.get("node_index"))
        depot_index = _as_int(event.payload.get("depot_index"))
        if route_id is None or node_index is None:
            return

        runtime = self.route_runtimes.get(route_id)
        if runtime is None:
            return

        if depot_index is not None:
            runtime.plan.depot_index = depot_index

        if node_index == runtime.plan.depot_index:
            return

        if node_index not in runtime.visited_set:
            runtime.visited_set.add(node_index)
            runtime.visited_order.append(node_index)

    def _handle_route_update(self, event: TimedEvent) -> None:
        for update in extract_route_updates(event.payload):
            runtime = self.route_runtimes.get(update.route_id)
            if runtime is None:
                continue

            if update.depot_index is not None and update.depot_index in self.positions:
                runtime.plan.depot_index = update.depot_index

            sanitized = [
                customer
                for customer in update.customers
                if customer in self.positions and customer != runtime.plan.depot_index
            ]

            if update.future_only:
                prefix = [customer for customer in runtime.visited_order if customer != runtime.plan.depot_index]
                prefix_set = set(prefix)
                runtime.plan.customers = prefix + [
                    customer for customer in sanitized if customer not in prefix_set
                ]
            else:
                runtime.plan.customers = sanitized

    def _vehicle_pose(self, runtime: RouteRuntime) -> Pose:
        depot_index = runtime.plan.depot_index
        depot_pos = self.positions.get(depot_index, (0.0, 0.0))
        timeline = runtime.timeline

        if not timeline:
            return Pose(depot_pos[0], depot_pos[1], "idle", depot_index, None)

        while (
            runtime.timeline_cursor < len(timeline) - 1
            and self.sim_time > timeline[runtime.timeline_cursor].end + EPS
        ):
            runtime.timeline_cursor += 1

        segment = timeline[runtime.timeline_cursor]

        if runtime.timeline_cursor == len(timeline) - 1 and self.sim_time > segment.end + EPS:
            if segment.kind == "travel":
                node_index = segment.to_node if segment.to_node is not None else depot_index
            else:
                node_index = segment.node if segment.node is not None else depot_index
            pos = self.positions.get(node_index, depot_pos)
            return Pose(pos[0], pos[1], "idle", node_index, None)

        if self.sim_time < segment.start - EPS:
            if runtime.timeline_cursor == 0:
                return Pose(depot_pos[0], depot_pos[1], "idle", depot_index, None)

            previous = timeline[runtime.timeline_cursor - 1]
            if previous.kind == "travel":
                node_index = previous.to_node if previous.to_node is not None else depot_index
            else:
                node_index = previous.node if previous.node is not None else depot_index
            pos = self.positions.get(node_index, depot_pos)
            return Pose(pos[0], pos[1], "idle", node_index, None)

        if segment.kind == "travel":
            from_node = segment.from_node if segment.from_node is not None else depot_index
            to_node = segment.to_node if segment.to_node is not None else depot_index
            start_pos = self.positions.get(from_node, depot_pos)
            end_pos = self.positions.get(to_node, depot_pos)
            duration = max(EPS, segment.end - segment.start)
            ratio = max(0.0, min(1.0, (self.sim_time - segment.start) / duration))
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio
            return Pose(x, y, "travel", from_node, to_node)

        node_index = segment.node if segment.node is not None else depot_index
        pos = self.positions.get(node_index, depot_pos)
        return Pose(pos[0], pos[1], "service", node_index, None)

    def _route_polyline(self, runtime: RouteRuntime, pose: Pose) -> tuple[list[float], list[float]]:
        remaining = [
            customer
            for customer in runtime.plan.customers
            if customer not in runtime.visited_set and customer in self.positions
        ]

        points: list[tuple[float, float]] = [(pose.x, pose.y)]
        for customer in remaining:
            points.append(self.positions[customer])

        depot_pos = self.positions.get(runtime.plan.depot_index)
        if depot_pos is not None:
            if remaining or pose.current_node != runtime.plan.depot_index:
                points.append(depot_pos)

        compact: list[tuple[float, float]] = []
        for point in points:
            if not compact:
                compact.append(point)
                continue
            prev = compact[-1]
            if math.hypot(point[0] - prev[0], point[1] - prev[1]) > 1e-9:
                compact.append(point)

        if len(compact) < 2:
            return [], []
        xs = [point[0] for point in compact]
        ys = [point[1] for point in compact]
        return xs, ys

    def _refresh_blocked_edges(self) -> None:
        cutoff = self.sim_time - self.blocked_edge_ttl
        while self.blocked_edges and self.blocked_edges[0].time_minutes < cutoff:
            self.blocked_edges.popleft()
        while len(self.blocked_edges) > self.max_blocked_edges:
            self.blocked_edges.popleft()

        segments: list[list[tuple[float, float]]] = []
        colors: list[tuple[float, float, float, float]] = []
        for blocked in self.blocked_edges:
            pos_a = self.positions.get(blocked.node_a)
            pos_b = self.positions.get(blocked.node_b)
            if pos_a is None or pos_b is None:
                continue

            age = max(0.0, self.sim_time - blocked.time_minutes)
            alpha = max(0.08, 1.0 - (age / self.blocked_edge_ttl))
            segments.append([pos_a, pos_b])
            colors.append((0.70, 0.10, 0.10, alpha))

        self.blocked_collection.set_segments(segments)
        self.blocked_collection.set_colors(colors)

    def _refresh_artists(self) -> None:
        for route_id, runtime in self.route_runtimes.items():
            pose = self._vehicle_pose(runtime)
            xs, ys = self._route_polyline(runtime, pose)

            self.route_lines[route_id].set_data(xs, ys)

            marker = self.vehicle_markers[route_id]
            marker.set_data([pose.x], [pose.y])
            if pose.mode == "service":
                marker.set_marker("s")
                marker.set_markerfacecolor("#ff8c00")
            elif pose.mode == "travel":
                marker.set_marker("o")
                marker.set_markerfacecolor("#2ca02c")
            else:
                marker.set_marker("o")
                marker.set_markerfacecolor("#b0b0b0")

            if self.show_ids and route_id in self.vehicle_labels:
                self.vehicle_labels[route_id].set_position((pose.x + 1.0, pose.y + 1.0))

        self._refresh_blocked_edges()

        state = "PAUSED" if self.paused else "RUNNING"
        self.status_text.set_text(
            "\n".join(
                [
                    (
                        f"t = {self.sim_time:8.2f} min / {self.total_time:.2f}  "
                        f"|  {state}  |  speed = {self.speed:.2f} min/s"
                    ),
                    (
                        f"events = {self.event_index}/{len(self.events)}  "
                        f"|  blocked visible = {len(self.blocked_edges)}"
                    ),
                    "keys: space play/pause, up/down speed, left/right seek",
                ]
            )
        )

    def _on_key_press(self, event: Any) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == " ":
            self.paused = not self.paused
            return
        if key == "up":
            self.speed = min(500.0, self.speed * 1.4)
            return
        if key == "down":
            self.speed = max(0.1, self.speed / 1.4)
            return
        if key == "right":
            self.paused = True
            self.set_time(self.sim_time + 5.0)
            self._refresh_artists()
            self.fig.canvas.draw_idle()
            return
        if key == "left":
            self.paused = True
            self.set_time(self.sim_time - 5.0)
            self._refresh_artists()
            self.fig.canvas.draw_idle()

    def _update(self, _frame_index: int) -> list[Any]:
        now = time.perf_counter()
        dt = max(0.0, now - self._last_wall_time)
        self._last_wall_time = now

        if not self.paused:
            next_time = min(self.total_time, self.sim_time + dt * self.speed)
            self.sim_time = next_time
            self._apply_events_until(self.sim_time)
            if self.sim_time >= self.total_time - EPS:
                self.paused = True

        self._refresh_artists()
        return []

    def run(self) -> None:
        interval_ms = int(1000 / self.fps)
        self._animation = FuncAnimation(
            self.fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate MDVRP simulation logs.")
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to log JSON (e.g., data/processed/simulation_logs/p22_log.json).",
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        default=None,
        help="Path to Cordeau instance file. If omitted, inferred from log metadata.",
    )
    parser.add_argument(
        "--routes-file",
        type=Path,
        default=None,
        help="Path to initial routes JSON. If omitted, inferred as data/processed/results/<instance>_routes.json.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Render FPS (default: 30).")
    parser.add_argument(
        "--speed",
        type=float,
        default=8.0,
        help="Initial simulation speed in minutes/second (default: 8.0).",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Initial simulation time in minutes.",
    )
    parser.add_argument(
        "--blocked-edge-ttl",
        type=float,
        default=45.0,
        help="Minutes a blocked edge remains visible (default: 45).",
    )
    parser.add_argument(
        "--max-blocked-edges",
        type=int,
        default=600,
        help="Max blocked edges rendered at once (default: 600).",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Show route id labels next to vehicle markers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and print summary without opening the animation window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_file = args.log_file.resolve()
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    metadata, events = load_event_log(log_file)
    instance_name, instance_file, routes_file = resolve_paths(
        log_file=log_file,
        metadata=metadata,
        instance_file=args.instance_file.resolve() if args.instance_file is not None else None,
        routes_file=args.routes_file.resolve() if args.routes_file is not None else None,
    )

    visualizer = Visualizer.from_files(
        instance_name=instance_name,
        instance_file=instance_file,
        routes_file=routes_file,
        events=events,
        fps=args.fps,
        initial_speed=args.speed,
        blocked_edge_ttl=args.blocked_edge_ttl,
        max_blocked_edges=args.max_blocked_edges,
        show_ids=args.show_ids,
    )

    if args.start_time > 0.0:
        visualizer.set_time(args.start_time)

    if args.dry_run:
        print(f"instance={instance_name}")
        print(f"events={len(events)}")
        print(f"routes={len(visualizer.route_runtimes)}")
        print(f"time_horizon_minutes={visualizer.total_time:.3f}")
        print(f"instance_file={instance_file}")
        print(f"routes_file={routes_file}")
        return 0

    visualizer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
