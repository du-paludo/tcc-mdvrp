"""
Configuration loader for the MDVRP solver.

Reads config.yaml from the project root and returns typed dataclasses
so all algorithm modules receive validated, IDE-completable parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

@dataclass
class CCBCConfig:
    max_iter: int
    tol: float
    n_starts: int
    capacity_balance_target: float
    duration_estimate_slack: float


@dataclass
class LocalSearchConfig:
    max_iterations: int
    granularity: int


@dataclass
class GAConfig:
    pop_size: int
    n_gen: int
    seed: int
    mutation_prob: float
    clone_delta: float
    stagnation_period: int
    stagnation_ftol: float
    time_limit: str
    capacity_penalty: float
    duration_penalty: float
    n_offsprings: int
    repair_prob: float
    feasibility_target: float
    penalty_adjustment_period: int


@dataclass
class SimulationConfig:
    reroute_degradation_threshold: float
    cluster_degradation_threshold: float
    penalty_overcapacity_per_unit: float
    penalty_overtime_per_minute: float


@dataclass
class AppConfig:
    ccbc: CCBCConfig
    ga: GAConfig
    local_search: LocalSearchConfig
    simulation: SimulationConfig


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load and parse config.yaml into an AppConfig dataclass.

    Args:
        path: Path to the YAML file. Defaults to config.yaml in the
              project root (two levels above this file).

    Returns:
        AppConfig with fully typed ccbc and ga sub-configs.
    """
    if path is None:
        path = str(Path(__file__).parent.parent.parent / "config.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    ccbc_raw = raw["ccbc"]
    ga_raw = raw["ga"]
    ls_raw = raw["local_search"]
    simulation_raw = raw.get("simulation", {})

    return AppConfig(
        ccbc=CCBCConfig(
            max_iter=int(ccbc_raw["max_iter"]),
            tol=float(ccbc_raw["tol"]),
            n_starts=int(ccbc_raw["n_starts"]),
            capacity_balance_target=float(ccbc_raw["capacity_balance_target"]),
            duration_estimate_slack=float(ccbc_raw["duration_estimate_slack"]),
        ),
        ga=GAConfig(
            pop_size=int(ga_raw["pop_size"]),
            n_gen=int(ga_raw["n_gen"]),
            seed=int(ga_raw["seed"]),
            mutation_prob=float(ga_raw["mutation_prob"]),
            clone_delta=float(ga_raw["clone_delta"]),
            stagnation_period=int(ga_raw["stagnation_period"]),
            stagnation_ftol=float(ga_raw["stagnation_ftol"]),
            time_limit=str(ga_raw["time_limit"]),
            capacity_penalty=float(ga_raw["capacity_penalty"]),
            duration_penalty=float(ga_raw["duration_penalty"]),
            repair_prob=float(ga_raw["repair_prob"]),
            feasibility_target=float(ga_raw["feasibility_target"]),
            penalty_adjustment_period=int(ga_raw["penalty_adjustment_period"]),
            n_offsprings=int(ga_raw["n_offsprings"]),
        ),
        local_search=LocalSearchConfig(
            max_iterations=int(ls_raw["max_iterations"]),
            granularity=int(ls_raw["granularity"]),
        ),
        simulation=SimulationConfig(
            reroute_degradation_threshold=float(simulation_raw["reroute_degradation_threshold"]),
            cluster_degradation_threshold=float(simulation_raw["cluster_degradation_threshold"]),
            penalty_overcapacity_per_unit=float(simulation_raw["penalty_overcapacity_per_unit"]),
            penalty_overtime_per_minute=float(simulation_raw["penalty_overtime_per_minute"]),
        ),
    )
