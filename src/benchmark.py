"""
Benchmark runner for the MDVRP solver.

Runs a chosen algorithm across all (or selected) Cordeau benchmark instances
and prints a results table comparing algorithm cost against the reference
solution.

Usage
-----
    # Run all instances (default)
    python src/benchmark.py

    # Run specific instances
    python src/benchmark.py --instances p01 p02 p05

    # Run only p- or pr-instances
    python src/benchmark.py --set p
    python src/benchmark.py --set pr
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from algorithms.ccbc_ga import CCBCGAAlgorithm
from utils.config import load_config
from utils.converter import load_instance

_P_INSTANCES = [f"p{i:02d}" for i in range(1, 24)]
_PR_INSTANCES = [f"pr{i:02d}" for i in range(1, 11)]

# Column widths:  Instance  Cust  Dep  Routes  Reference   Algorithm   Gap %   Feasible  Time(s)
_COL_W = (10, 6, 4, 7, 12, 12, 7, 9, 8)
_HEADERS = ("Instance", "Cust", "Dep", "Routes", "Reference", "Algorithm", "Gap %", "Feasible", "Time(s)")


@dataclass
class _InstanceResult:
    name: str
    n_customers: int
    n_depots: int
    n_routes: int
    ref: float | None
    cost: float
    feasible: bool
    elapsed: float

    @property
    def gap(self) -> float | None:
        if self.ref is None:
            return None
        return (self.cost - self.ref) / self.ref * 100.0


def _header_line() -> str:
    return "  ".join(h.ljust(w) for h, w in zip(_HEADERS, _COL_W))


def _separator() -> str:
    return "  ".join("-" * w for w in _COL_W)


def _row(r: _InstanceResult) -> str:
    ref_str = f"{r.ref:.2f}" if r.ref is not None else "N/A"
    gap_str = f"{r.gap:.1f}" if r.gap is not None else "N/A"
    return "  ".join([
        r.name.ljust(_COL_W[0]),
        str(r.n_customers).ljust(_COL_W[1]),
        str(r.n_depots).ljust(_COL_W[2]),
        str(r.n_routes).ljust(_COL_W[3]),
        ref_str.ljust(_COL_W[4]),
        f"{r.cost:.2f}".ljust(_COL_W[5]),
        gap_str.ljust(_COL_W[6]),
        str(r.feasible).ljust(_COL_W[7]),
        f"{r.elapsed:.1f}".ljust(_COL_W[8]),
    ])


def run_benchmark(instance_names: list[str]) -> None:
    cfg = load_config()
    algorithm = CCBCGAAlgorithm(cfg)

    print(f"\nAlgorithm : {algorithm}")
    print(f"Instances : {len(instance_names)}\n")
    print(_header_line())
    print(_separator())

    results: list[_InstanceResult] = []

    for name in instance_names:
        try:
            loaded = load_instance(name)
        except FileNotFoundError as exc:
            print(f"  {name:<10}  SKIPPED  ({exc})")
            continue

        t0 = time.perf_counter()
        solution = algorithm.solve(loaded.customers, loaded.depots)
        elapsed = time.perf_counter() - t0

        result = _InstanceResult(
            name=name,
            n_customers=len(loaded.customers),
            n_depots=len(loaded.depots),
            n_routes=len(solution.routes),
            ref=loaded.reference.objective if loaded.reference is not None else None,
            cost=solution.total_cost(),
            feasible=solution.is_feasible(),
            elapsed=elapsed,
        )
        results.append(result)
        print(_row(result))

    print(_separator())
    _print_summary(results)


def _print_summary(results: list[_InstanceResult]) -> None:
    if not results:
        return

    gaps = [r.gap for r in results if r.gap is not None]
    feasible_count = sum(1 for r in results if r.feasible)
    total_elapsed = sum(r.elapsed for r in results)

    print(f"\n{'Summary':=<50}")
    print(f"  Instances run   : {len(results)}")
    print(f"  Feasible        : {feasible_count} / {len(results)}")
    print(f"  Total time      : {total_elapsed:.1f}s")

    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        best_gap = min(gaps)
        worst_gap = max(gaps)
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        std_gap = variance ** 0.5

        print(f"\n  Gap vs reference ({len(gaps)} instances with known optimum):")
        print(f"    Mean  : {mean_gap:+.2f}%")
        print(f"    Best  : {best_gap:+.2f}%  ({next(r.name for r in results if r.gap == best_gap)})")
        print(f"    Worst : {worst_gap:+.2f}%  ({next(r.name for r in results if r.gap == worst_gap)})")
        print(f"    Std   : {std_gap:.2f}%")

        # Per-instance gap sorted best → worst
        print(f"\n  Gap ranking (best → worst):")
        ranked = sorted((r for r in results if r.gap is not None), key=lambda r: r.gap)
        for r in ranked:
            bar_len = max(0, min(40, int(abs(r.gap) / max(abs(worst_gap), 1e-9) * 20)))
            bar = ("+" if r.gap >= 0 else "-") * bar_len
            print(f"    {r.name:<10}  {r.gap:+6.1f}%  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MDVRP solver on Cordeau instances.")
    parser.add_argument(
        "--instances",
        nargs="+",
        metavar="NAME",
        help="Specific instance names to run, e.g. p01 p02 pr03.",
    )
    parser.add_argument(
        "--set",
        choices=["p", "pr", "all"],
        default="all",
        help="Which instance set to run when --instances is not provided (default: all).",
    )
    args = parser.parse_args()

    if args.instances:
        names = args.instances
    elif args.set == "pr":
        names = _PR_INSTANCES
    elif args.set == "p":
        names = _P_INSTANCES
    else:
        names = _P_INSTANCES + _PR_INSTANCES

    run_benchmark(names)


if __name__ == "__main__":
    main()
