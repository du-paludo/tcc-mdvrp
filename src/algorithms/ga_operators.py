"""
pymoo operator implementations for GA-based VRP routing.

Provides:
- HeuristicSampling: population initialisation seeded with nearest-neighbour tours
- LSMutation: Prins (2004) local-search mutation applied after crossover
"""

from typing import Callable, List

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival

from core.entities import Customer, Depot
from algorithms.ga_split import linear_split
from algorithms.ga_local_search import local_search, _route_cost


def _is_infeasible(
    segs: List[List[Customer]],
    depot: Depot,
    dist_fn: Callable[[int, int], float],
) -> bool:
    """Return True if any route violates capacity, duration, or fleet-size constraints."""
    if len(segs) > depot.max_vehicles:
        return True
    for seg in segs:
        if sum(c.demand for c in seg) > depot.max_capacity:
            return True
        if depot.max_duration > 0:
            travel = _route_cost(seg, depot, dist_fn)
            service = sum(c.service_time for c in seg)
            if travel + service > depot.max_duration:
                return True
    return False


def _nearest_neighbor_permutation(
    customers: List[Customer],
    start_node,
    dist_fn: Callable[[int, int], float],
    first_pos: int | None = None,
) -> np.ndarray:
    """
    Greedy nearest-neighbour giant tour.

    Visits every customer exactly once by always choosing the nearest
    unvisited customer.  Returns an integer permutation array of shape
    ``(n,)`` representing visit order (indices into ``customers``).

    Parameters
    ----------
    customers:
        Customer list (same order as chromosome encoding).
    start_node:
        Node from which the tour begins (Depot or Customer).
    dist_fn:
        O(1) distance callable.
    first_pos:
        Index into ``customers`` for the first customer to visit.
        If None, the customer nearest to ``start_node`` is chosen.
    """
    n = len(customers)
    remaining = list(range(n))
    tour: List[int] = []

    if first_pos is None:
        current_pos = min(remaining, key=lambda i: dist_fn(start_node.index, customers[i].index))
    else:
        current_pos = first_pos % n  # clamp to valid range

    remaining.remove(current_pos)
    tour.append(current_pos)
    current_node = customers[current_pos].index

    while remaining:
        nearest = min(remaining, key=lambda i: dist_fn(current_node, customers[i].index))
        tour.append(nearest)
        current_node = customers[nearest].index
        remaining.remove(nearest)

    return np.array(tour, dtype=int)


class HeuristicSampling(Sampling):
    """
    Population initialisation seeded with nearest-neighbour heuristic tours.

    Generates ``n_heuristic`` individuals using the greedy nearest-neighbour
    heuristic—one deterministic (nearest to ``start_node``) and the rest with
    random starting customers for diversity—then fills the remainder of the
    population with random permutations.

    This mirrors Prins (2004) K4, which seeds one individual with the
    Clarke-Wright savings heuristic and randomises the rest.

    Parameters
    ----------
    customers:
        Customer list (defines the permutation domain).
    start_node:
        Starting node for distance evaluation (Depot or Customer).
    dist_fn:
        O(1) distance callable.
    n_heuristic:
        Number of heuristic-seeded individuals (capped at pop_size).
    """

    def __init__(
        self,
        customers: List[Customer],
        start_node,
        dist_fn: Callable[[int, int], float],
        n_heuristic: int = 1,
    ) -> None:
        super().__init__()
        self.customers = customers
        self.start_node = start_node
        self.dist_fn = dist_fn
        self.n_heuristic = n_heuristic

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        n = len(self.customers)
        X = np.empty((n_samples, n), dtype=int)
        rng = np.random.default_rng()
        n_heuristic = min(self.n_heuristic, n_samples)

        # First: deterministic NN from start_node (nearest customer first)
        if n_heuristic >= 1:
            X[0] = _nearest_neighbor_permutation(
                self.customers, self.start_node, self.dist_fn, first_pos=None
            )

        # Remaining heuristic individuals: NN with random starting customers
        for i in range(1, n_heuristic):
            first_pos = int(rng.integers(0, n))
            X[i] = _nearest_neighbor_permutation(
                self.customers, self.start_node, self.dist_fn, first_pos=first_pos
            )

        # Fill the rest with random permutations
        for i in range(n_heuristic, n_samples):
            X[i] = rng.permutation(n)

        return X


class LSMutation(Mutation):
    """
    Prins (2004) local-search mutation operator for pymoo GA.

    Applied with probability ``prob`` to each child chromosome after crossover.
    The chromosome is decoded to routes via linear_split, improved by
    ``local_search``, then re-encoded back to an integer permutation so pymoo
    can continue operating on it.

    Parameters
    ----------
    depot:
        The depot used for route evaluation.
    customers:
        All customers in the current cluster (defines the permutation encoding).
    dist_fn:
        O(1) distance callable.
    prob:
        Per-individual mutation probability (passed to pymoo Mutation base).
    """

    def __init__(
        self,
        depot: Depot,
        customers: List[Customer],
        dist_fn: Callable[[int, int], float],
        prob: float,
        local_search_max_iterations: int,
        capacity_penalty: float,
        duration_penalty: float,
        repair_prob: float = 0.0,
        granularity: int = 0,
    ) -> None:
        super().__init__(prob=prob)
        self.depot = depot
        self.customers = customers
        self.dist_fn = dist_fn
        self.local_search_max_iterations = local_search_max_iterations
        self.capacity_penalty = capacity_penalty
        self.duration_penalty = duration_penalty
        self.repair_prob = repair_prob
        self.granularity = granularity
        # Map customer object → position in customers list for re-encoding
        self._customer_pos = {c: i for i, c in enumerate(customers)}

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        X = X.copy()
        n = len(self.customers)
        if n <= 1:
            return X

        rng = np.random.default_rng()
        for k in range(len(X)):
            if rng.random() >= self.prob.value:
                continue

            ordered = [self.customers[i] for i in X[k]]
            segments = linear_split(ordered, self.depot, self.dist_fn,
                                     capacity_penalty=self.capacity_penalty,
                                     duration_penalty=self.duration_penalty)
            improved_segs = local_search(segments, self.depot, self.dist_fn,
                                         self.local_search_max_iterations,
                                         granularity=self.granularity,
                                         self.capacity_penalty, self.duration_penalty)

            # Vidal (2011) §4.5 Repair: if offspring is infeasible, re-run split+LS
            # with escalated penalties (×10, then ×100) to push toward feasibility.
            if (
                self.repair_prob > 0.0
                and _is_infeasible(improved_segs, self.depot, self.dist_fn)
                and rng.random() < self.repair_prob
            ):
                repair_order = [c for route in improved_segs for c in route]
                cap10 = self.capacity_penalty * 10.0
                dur10 = self.duration_penalty * 10.0
                repaired = linear_split(repair_order, self.depot, self.dist_fn,
                                        capacity_penalty=cap10, duration_penalty=dur10)
                repaired = local_search(repaired, self.depot, self.dist_fn,
                                        self.local_search_max_iterations, cap10, dur10)
                if _is_infeasible(repaired, self.depot, self.dist_fn):
                    cap100 = self.capacity_penalty * 100.0
                    dur100 = self.duration_penalty * 100.0
                    repaired = linear_split(repair_order, self.depot, self.dist_fn,
                                            capacity_penalty=cap100, duration_penalty=dur100)
                    repaired = local_search(repaired, self.depot, self.dist_fn,
                                            self.local_search_max_iterations, cap100, dur100)
                improved_segs = repaired

            # Re-encode: flatten improved segments → integer permutation
            new_order = [c for route in improved_segs for c in route]
            if len(new_order) != n:
                # Fallback: keep original if LS dropped customers (shouldn't happen)
                continue

            X[k] = np.array([self._customer_pos[c] for c in new_order], dtype=int)

        return X


class WellSpacedSurvival(Survival):
    """
    Vidal (2012) well-spaced population survival selection.

    Sorts the merged population by fitness, then greedily retains individuals
    whose cost bucket floor(F/delta) has not yet been occupied. This enforces
    the well-spaced condition from Prins (2004): |F(P1) - F(P2)| >= delta
    for all pairs in the surviving population.

    If fewer than n_survive well-spaced individuals exist, the remainder is
    filled with the best remaining individuals (least-bad clones), matching
    the paper's truncation behaviour.

    Parameters
    ----------
    delta:
        Minimum fitness spacing. delta=1.0 enforces distinct integer costs.
    """

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(filter_infeasible=False)
        self.delta = delta
        self.eliminated_count: int = 0

    def _do(self, problem, pop, n_survive, **kwargs):
        F = pop.get("F").flatten()
        sorted_idx = np.argsort(F)

        kept = []
        clones = []
        occupied: set = set()

        for i in sorted_idx:
            bucket = int(F[i] / self.delta)
            if bucket not in occupied:
                kept.append(i)
                occupied.add(bucket)
            else:
                clones.append(i)

        # Clones that are truly discarded (not needed as fill)
        n_fill = max(0, n_survive - len(kept))
        self.eliminated_count += len(clones) - n_fill

        if n_fill > 0:
            kept.extend(clones[:n_fill])

        return pop[kept[:n_survive]]
