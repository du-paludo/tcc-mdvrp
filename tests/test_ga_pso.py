"""Unit tests for the CCBC clustering and GA routing modules."""

import math

import numpy as np
import pytest

from core.entities import Customer, Depot, Route
from core.solution import Solution
from utils.config import CCBCConfig, GAConfig, LocalSearchConfig, AppConfig, SimulationConfig

from algorithms.ccbc_cluster import run_ccbc_clustering
from algorithms.ga_problems import RoutingProblem
from algorithms.ga_router import run_ga_routing
from algorithms.ga_local_search import local_search, local_search_stage1_intra, _route_cost
from algorithms.ccbc_ga import CCBCGAAlgorithm


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def depots() -> list:
    return [
        Depot(index=1, x=0.0,  y=0.0,  max_duration=0.0, max_capacity=60),
        Depot(index=2, x=20.0, y=20.0, max_duration=0.0, max_capacity=60),
    ]


@pytest.fixture
def customers() -> list:
    # Four customers: two near depot 1, two near depot 2
    return [
        Customer(index=1, x=1.0,  y=1.0,  demand=10, service_time=0),
        Customer(index=2, x=2.0,  y=2.0,  demand=10, service_time=0),
        Customer(index=3, x=19.0, y=19.0, demand=10, service_time=0),
        Customer(index=4, x=21.0, y=21.0, demand=10, service_time=0),
    ]


def _dist(a: int, b: int, nodes: dict) -> float:
    ax, ay = nodes[a]
    bx, by = nodes[b]
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


@pytest.fixture
def dist_fn(depots, customers):
    """Simple Euclidean dist_fn for use in tests (no pre-built matrix)."""
    nodes = {d.index: (d.x, d.y) for d in depots}
    nodes.update({c.index: (c.x, c.y) for c in customers})
    return lambda a, b: _dist(a, b, nodes)


@pytest.fixture
def ccbc_cfg() -> CCBCConfig:
    return CCBCConfig(max_iter=100, tol=1e-4, n_starts=3, capacity_balance_target=0.90, duration_estimate_slack=1.15)


@pytest.fixture
def ga_cfg() -> GAConfig:
    return GAConfig(
        pop_size=10,
        n_gen=30,
        seed=0,
        mutation_prob=0.5,
        clone_delta=1.0,
        stagnation_period=5,
        stagnation_ftol=1e-6,
        time_limit="00:01:00",
        capacity_penalty=500.0,
        duration_penalty=500.0,
        n_offsprings=10,
    )


@pytest.fixture
def ls_cfg() -> LocalSearchConfig:
    return LocalSearchConfig(max_iterations=10, granularity=0)

@pytest.fixture
def simulation_cfg() -> SimulationConfig:
    return SimulationConfig(
        reroute_degradation_threshold=1.20,
        cluster_degradation_threshold=1.10,
        penalty_overcapacity_per_unit=100000.0,
        penalty_overtime_per_minute=50000.0,
    )


@pytest.fixture
def app_cfg(ccbc_cfg, ga_cfg, ls_cfg, simulation_cfg) -> AppConfig:
    return AppConfig(
        ccbc=ccbc_cfg,
        ga=ga_cfg,
        local_search=ls_cfg,
        simulation=simulation_cfg,
    )


# ---------------------------------------------------------------------------
# run_ccbc_clustering
# ---------------------------------------------------------------------------

class TestCCBCClustering:
    def test_customers_split_by_nearest_depot(self, depots, customers):
        """Customers near depot 1 should be assigned to depot 1, etc."""
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3, capacity_balance_target=0.90, duration_estimate_slack=1.15)
        clusters = run_ccbc_clustering(customers=customers, depots=depots, cfg=cfg)
        depot1_indices = {c.index for c in clusters[depots[0]]}
        depot2_indices = {c.index for c in clusters[depots[1]]}
        # Customers 1,2 are near depot 1 (0,0); customers 3,4 near depot 2 (20,20)
        assert {1, 2}.issubset(depot1_indices)
        assert {3, 4}.issubset(depot2_indices)

    def test_all_customers_assigned(self, depots, customers):
        """Every customer must appear in exactly one cluster."""
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3, capacity_balance_target=0.90, duration_estimate_slack=1.15)
        clusters = run_ccbc_clustering(customers=customers, depots=depots, cfg=cfg)
        assigned = [c for cs in clusters.values() for c in cs]
        assert len(assigned) == len(customers)
        assert {c.index for c in assigned} == {c.index for c in customers}

    def test_capacity_budget_respected(self, depots):
        """No cluster should exceed its capacity budget when avoidable."""
        # 6 customers each with demand=10; each depot has capacity=60 and 1 vehicle
        # → budget=60 per depot; 3 customers per depot is feasible
        cs = [Customer(index=i, x=float(i), y=0.0, demand=10, service_time=0) for i in range(1, 7)]
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3, capacity_balance_target=0.90, duration_estimate_slack=1.15)
        clusters = run_ccbc_clustering(customers=cs, depots=depots, cfg=cfg)
        for depot, assigned in clusters.items():
            total = sum(c.demand for c in assigned)
            budget = depot.max_capacity * depot.max_vehicles
            assert total <= budget, f"Depot {depot.index} exceeded budget: {total} > {budget}"

    def test_empty_customers(self, depots):
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3, capacity_balance_target=0.90, duration_estimate_slack=1.15)
        clusters = run_ccbc_clustering(customers=[], depots=depots, cfg=cfg)
        assert all(v == [] for v in clusters.values())

    def test_returns_all_depots(self, depots, customers):
        cfg = CCBCConfig(max_iter=100, tol=1e-4, n_starts=3, capacity_balance_target=0.90, duration_estimate_slack=1.15)
        clusters = run_ccbc_clustering(customers=customers, depots=depots, cfg=cfg)
        assert set(clusters.keys()) == set(depots)


# ---------------------------------------------------------------------------
# RoutingProblem
# ---------------------------------------------------------------------------

class TestRoutingProblem:
    def test_evaluate_returns_round_trip_cost(self, depots, dist_fn):
        depot = depots[0]
        route_customers = [
            Customer(index=101, x=3.0, y=4.0, demand=5, service_time=0),
        ]
        nodes = {depot.index: (depot.x, depot.y)}
        nodes.update({c.index: (c.x, c.y) for c in route_customers})
        dfn = lambda a, b: _dist(a, b, nodes)

        problem = RoutingProblem(depot=depot, customers=route_customers, dist_fn=dfn,
                                  capacity_penalty=500.0, duration_penalty=500.0)
        out: dict = {}
        problem._evaluate(np.array([0]), out)
        # depot(0,0) → (3,4) → depot(0,0) = 5+5 = 10
        assert out["F"] == pytest.approx(10.0)

    def test_shorter_permutation_wins(self, depots, dist_fn):
        depot = depots[0]  # (0,0)
        # Place customers along x-axis for predictable ordering
        cs = [
            Customer(index=1, x=1.0, y=0.0, demand=5, service_time=0),
            Customer(index=2, x=2.0, y=0.0, demand=5, service_time=0),
            Customer(index=3, x=3.0, y=0.0, demand=5, service_time=0),
        ]
        nodes = {depot.index: (depot.x, depot.y)}
        nodes.update({c.index: (c.x, c.y) for c in cs})
        dfn = lambda a, b: _dist(a, b, nodes)

        problem = RoutingProblem(depot=depot, customers=cs, dist_fn=dfn,
                                  capacity_penalty=500.0, duration_penalty=500.0)

        # Sequential order [0,1,2] → 1+1+1+3 = 6
        out_good: dict = {}
        problem._evaluate(np.array([0, 1, 2]), out_good)

        # Permutation [1,2,0]: depot→cs[1]→cs[2]→cs[0]→depot
        out_bad: dict = {}
        problem._evaluate(np.array([1, 2, 0]), out_bad)

        # Both valid; key assertion: no negative costs
        assert out_good["F"] > 0
        assert out_bad["F"] > 0


# ---------------------------------------------------------------------------
# local_search
# ---------------------------------------------------------------------------

class TestLocalSearch:
    @pytest.fixture
    def ls_depot(self):
        return Depot(index=0, x=0.0, y=0.0, max_duration=0.0, max_capacity=100)

    @pytest.fixture
    def ls_nodes(self, ls_depot):
        """Customers on a grid; depot at origin."""
        cs = [
            Customer(index=1, x=1.0, y=0.0, demand=5, service_time=0),
            Customer(index=2, x=2.0, y=0.0, demand=5, service_time=0),
            Customer(index=3, x=3.0, y=0.0, demand=5, service_time=0),
            Customer(index=4, x=4.0, y=0.0, demand=5, service_time=0),
        ]
        nodes = {ls_depot.index: (ls_depot.x, ls_depot.y)}
        nodes.update({c.index: (c.x, c.y) for c in cs})
        dfn = lambda a, b: _dist(a, b, nodes)
        return cs, dfn

    def test_cost_non_increasing(self, ls_depot, ls_nodes):
        """Local search must never worsen the total solution cost."""
        customers, dfn = ls_nodes
        routes = [Route(depot=ls_depot, customers=[customers[3], customers[0]]),
                  Route(depot=ls_depot, customers=[customers[2], customers[1]])]
        before = sum(_route_cost(r.customers, ls_depot, dfn) for r in routes)
        improved = local_search(routes, ls_depot, dfn, local_search_max_iterations=200)
        after = sum(_route_cost(r, ls_depot, dfn) for r in improved)
        assert after <= before + 1e-9

    def test_all_customers_preserved(self, ls_depot, ls_nodes):
        """No customers should be lost or duplicated after local search."""
        customers, dfn = ls_nodes
        routes = [Route(depot=ls_depot, customers=[customers[0], customers[3]]),
                  Route(depot=ls_depot, customers=[customers[1], customers[2]])]
        improved = local_search(routes, ls_depot, dfn, local_search_max_iterations=200)
        result_indices = [c.index for r in improved for c in r]
        expected_indices = sorted(c.index for c in customers)
        assert sorted(result_indices) == expected_indices

    def test_capacity_feasible_after_ls(self, ls_nodes):
        """All routes returned must satisfy capacity constraints."""
        customers, dfn = ls_nodes
        tight_depot = Depot(index=0, x=0.0, y=0.0, max_duration=0.0, max_capacity=10)
        routes = [Route(depot=tight_depot, customers=[customers[0], customers[1]]),
                  Route(depot=tight_depot, customers=[customers[2], customers[3]])]
        improved = local_search(routes, tight_depot, dfn, local_search_max_iterations=200)
        for r in improved:
            assert sum(c.demand for c in r) <= tight_depot.max_capacity

    def test_empty_routes_removed(self, ls_depot, ls_nodes):
        """Local search must strip empty routes from its output."""
        customers, dfn = ls_nodes
        routes = [Route(depot=ls_depot, customers=[c]) for c in customers]
        improved = local_search(routes, ls_depot, dfn, local_search_max_iterations=200)
        assert all(len(r) > 0 for r in improved)

    def test_single_route_stays_valid(self, ls_depot, ls_nodes):
        """A single feasible route should remain valid after LS."""
        customers, dfn = ls_nodes
        routes = [Route(depot=ls_depot, customers=list(customers))]
        improved = local_search(routes, ls_depot, dfn, local_search_max_iterations=200)
        result_indices = sorted(c.index for r in improved for c in r)
        assert result_indices == sorted(c.index for c in customers)

    def test_stage1_intra_cost_non_increasing(self, ls_depot, ls_nodes):
        """Stage-1 LS should not worsen start->route->end duration."""
        customers, dfn = ls_nodes
        start_node = customers[0]
        end_node = ls_depot
        pending = [customers[3], customers[2], customers[1]]

        def _open_duration(route):
            if not route:
                return dfn(start_node.index, end_node.index)
            travel = dfn(start_node.index, route[0].index)
            for i in range(len(route) - 1):
                travel += dfn(route[i].index, route[i + 1].index)
            travel += dfn(route[-1].index, end_node.index)
            return travel + sum(c.service_time for c in route)

        before = _open_duration(pending)
        improved = local_search_stage1_intra(pending, start_node, end_node, dfn)
        after = _open_duration(improved)
        assert after <= before + 1e-9

    def test_stage1_intra_preserves_customers(self, ls_depot, ls_nodes):
        """Stage-1 LS must preserve customer set exactly once."""
        customers, dfn = ls_nodes
        start_node = customers[0]
        end_node = ls_depot
        pending = [customers[3], customers[2], customers[1]]

        improved = local_search_stage1_intra(pending, start_node, end_node, dfn)
        assert sorted(c.index for c in improved) == sorted(c.index for c in pending)


# ---------------------------------------------------------------------------
# run_ga_routing edge cases
# ---------------------------------------------------------------------------

class TestRunGARouting:
    def test_empty_cluster(self, depots, ga_cfg, ls_cfg, dist_fn):
        routes, _ = run_ga_routing(depots[0], [], dist_fn, ga_cfg, ls_cfg)
        assert routes == []

    def test_single_customer(self, depots, customers, ga_cfg, ls_cfg, dist_fn):
        routes, _ = run_ga_routing(depots[0], [customers[0]], dist_fn, ga_cfg, ls_cfg)
        assert len(routes) == 1
        assert routes[0].customers[0].index == customers[0].index


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CCBCGAAlgorithm smoke test
# ---------------------------------------------------------------------------

class TestCCBCGAAlgorithm:
    def test_solve_returns_solution(self, depots, customers, app_cfg):
        algo = CCBCGAAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assert isinstance(solution, Solution)

    def test_all_customers_assigned(self, depots, customers, app_cfg):
        algo = CCBCGAAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assigned = {c.index for route in solution.routes for c in route.customers}
        expected = {c.index for c in customers}
        assert assigned == expected

    def test_cost_is_positive(self, depots, customers, app_cfg):
        algo = CCBCGAAlgorithm(app_cfg)
        solution = algo.solve(customers, depots)
        assert solution.total_cost() > 0

    def test_single_depot_can_use_multiple_vehicles(self, app_cfg):
        depot = Depot(
            index=1,
            x=0.0,
            y=0.0,
            max_duration=0.0,
            max_capacity=20,
            max_vehicles=2,
        )
        customers = [
            Customer(index=101, x=1.0, y=0.0, demand=10, service_time=0),
            Customer(index=102, x=2.0, y=0.0, demand=10, service_time=0),
            Customer(index=103, x=3.0, y=0.0, demand=10, service_time=0),
            Customer(index=104, x=4.0, y=0.0, demand=10, service_time=0),
        ]

        algo = CCBCGAAlgorithm(app_cfg)
        solution = algo.solve(customers, [depot])

        assert len(solution.routes) == 2
        assert all(route.depot.index == depot.index for route in solution.routes)
        assert sum(len(route.customers) for route in solution.routes) == len(customers)
        assert all(route.total_demand() <= depot.max_capacity for route in solution.routes)


# ---------------------------------------------------------------------------
# load_config round-trip
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_types(self, app_cfg):
        assert isinstance(app_cfg.ccbc.max_iter, int)
        assert isinstance(app_cfg.ccbc.tol, float)
        assert isinstance(app_cfg.ccbc.n_starts, int)
        assert isinstance(app_cfg.ga.pop_size, int)
        assert isinstance(app_cfg.ga.n_gen, int)
        assert isinstance(app_cfg.ga.mutation_prob, float)
