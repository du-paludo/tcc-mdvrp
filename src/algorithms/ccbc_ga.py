"""
CCBC+GA Cluster-first, Route-second algorithm for MDVRP.

Phase 1 — Clustering (CCBC):
    Each customer is assigned to a depot via Constrained Centroid-Based
    Clustering (CCBC) — an augmented capacitated k-means with vehicle-level
    slots, multi-start Voronoi initialisation, and two-phase boundary
    resolution.

Phase 2 — Routing (GA + Linear split):
    Within each depot's cluster a GA optimises the giant-tour order via SPV
    encoding.  For each candidate permutation the Linear split finds the
    optimal vehicle partition given the depot's capacity.

Usage
-----
    cfg = load_config()
    algorithm = CCBCGAAlgorithm(cfg)
    solution = algorithm.solve(customers, depots)
"""

from typing import Dict, List

from algorithms.base import ClusterFirstAlgorithm
from algorithms.ccbc_cluster import run_ccbc_clustering
from algorithms.ga_router import run_ga_routing, GADepotHistory
from core.entities import Customer, Depot
from core.solution import Solution
from utils.config import AppConfig, load_config
from utils.reporting import print_cluster_summary
from concurrent.futures import ProcessPoolExecutor

_SEP = "-" * 78

class CCBCGAAlgorithm(ClusterFirstAlgorithm):
    """
    Cluster-first, route-second MDVRP solver using CCBC and GA.

    The distance matrix is built once by ``ClusterFirstAlgorithm.solve()``
    before either phase runs, so ``route()`` can use ``self._dist()`` for
    O(1) lookups.  The CCBC phase uses raw (x, y) coordinates directly
    for centroid arithmetic.

    Parameters
    ----------
    cfg:
        Full application config loaded from config.yaml.
        If omitted, config.yaml is loaded from the project root.
    """

    def __init__(self, cfg: AppConfig | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.last_clusters: Dict[int, List[int]] = {}
        self.last_ga_history: List[GADepotHistory] = []


    def cluster(
        self, customers: List[Customer], depots: List[Depot]
    ) -> Dict[Depot, List[Customer]]:
        """Phase 1: assign customers to depots via CCBC."""
        clusters = run_ccbc_clustering(
            customers=customers,
            depots=depots,
            cfg=self.cfg.ccbc,
        )
        self.last_clusters = {
            depot.index: [customer.index for customer in assigned]
            for depot, assigned in clusters.items()
        }
        # self._print_cluster_summary(clusters)
        return clusters

    def _print_cluster_summary(self, clusters: Dict[Depot, List[Customer]]) -> None:
        print_cluster_summary(clusters, self._dist)

    def route(self, clusters: Dict[Depot, List[Customer]]) -> Solution:
        """Phase 2: optimise visiting order and vehicle split per depot via GA."""
        # print(f"Starting GA routing for {len(clusters)} depots...")
        # print(_SEP)
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_ga_routing, depot, customers, self._dist, self.cfg.ga, self.cfg.local_search)
                for depot, customers in clusters.items()
            ]
            results = [f.result() for f in futures]
        routes = [r for depot_routes, _ in results for r in depot_routes]
        self.last_ga_history = [hist for _, hist in results]
        return Solution(routes=routes)

    def __repr__(self) -> str:
        return (
            f"CCBCGAAlgorithm("
            f"ccbc_iter={self.cfg.ccbc.max_iter}, ccbc_starts={self.cfg.ccbc.n_starts}, "
            f"ga_pop={self.cfg.ga.pop_size}, ga_gen={self.cfg.ga.n_gen})"
        )


