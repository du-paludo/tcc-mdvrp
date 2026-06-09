import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from algorithms.ccbc_ga import CCBCGAAlgorithm
from utils.config import load_config
from utils.converter import load_instance
from utils.data_loader import read_cordeau_data_file, read_cordeau_solution_file, read_failures_file
from utils.reporting import print_solution_summary, print_run_summary, print_simulation_validation
from utils.results_io import load_routing_result_as_solution, save_clustering_result, save_routing_result
from utils.visualizer import visualize_comparison, visualize_solution, visualize_ga_convergence
from scenario.simulator import SIMULATION_LOG_DIR, run_simulation
from tools.validate_simulation_log import validate_simulation_log


def main() -> int:
    base_dir = Path(__file__).parent.parent
    default_failures_file = None

    parser = argparse.ArgumentParser(description="Run and visualize the MDVRP solver on one instance.")
    parser.add_argument("--instance", default="p20", metavar="NAME", help="Instance name (default: p01).")
    parser.add_argument(
        "--failures-file",
        default=default_failures_file,
        metavar="PATH",
        help="Path to the failures JSON file (default: auto-detected for the selected instance).",
    )
    parser.add_argument(
        "--routes-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a precomputed routes JSON (same format as data/processed/results/*_routes.json). "
            "If provided, skips clustering/routing and runs only simulation on these routes."
        ),
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        default=False,
        help="Skip the simulation phase (default: run simulation if a failures file is found).",
    )
    args = parser.parse_args()

    loaded = load_instance(args.instance)
    customers = loaded.customers
    depots = loaded.depots
    reference_solution = loaded.reference

    """Load the selected instance, run the algorithm and visualize the result."""
    data_file = base_dir / "data" / "raw" / "cordeau" / args.instance
    solution_file = base_dir / "data" / "raw" / "cordeau_sol" / f"{args.instance}.res"
    failures_dir = base_dir / "data" / "processed" / "failures"
    results_dir = base_dir / "data" / "processed" / "results"

    if args.routes_file is not None:
        provided_routes = Path(args.routes_file)
        if provided_routes.is_absolute():
            routes_file = provided_routes
        elif provided_routes.exists():
            routes_file = provided_routes
        elif (base_dir / provided_routes).exists():
            routes_file = base_dir / provided_routes
        else:
            routes_file = results_dir / provided_routes.name
        if not routes_file.exists():
            raise FileNotFoundError(f"Routes file not found: {routes_file}")
    else:
        routes_file = None

    if args.failures_file is not None:
        provided_failures = Path(args.failures_file)
        if provided_failures.is_absolute():
            failures_file = provided_failures
        elif provided_failures.exists():
            failures_file = provided_failures
        elif (base_dir / provided_failures).exists():
            failures_file = base_dir / provided_failures
        else:
            failures_file = failures_dir / provided_failures.name
    else:
        default_failure_candidates = sorted(failures_dir.glob(f"{args.instance}_*.json"))
        if not default_failure_candidates:
            failures_file = None
        else:
            failures_file = default_failure_candidates[-1]

    # Load raw instance and reference solution
    instance = read_cordeau_data_file(str(data_file))
    if solution_file.exists():
        reference_solution = read_cordeau_solution_file(str(solution_file), instance)
    else:
        reference_solution = None
    failures = read_failures_file(str(failures_file)) if failures_file is not None else None

    cfg = load_config()

    # Always instantiate the algorithm because simulation rerouting depends on it.
    algorithm = CCBCGAAlgorithm(cfg, debug=True)

    if routes_file is not None:
        print(f"Using precomputed routes from: {routes_file}")
        t_start = time.perf_counter()
        solution = load_routing_result_as_solution(routes_file, customers, depots)
        elapsed = time.perf_counter() - t_start
        ga_history = []
        algorithm_repr = f"{algorithm} [precomputed-routes]"
        clustering_file = "(skipped)"
        routing_file = str(routes_file)
    else:
        # Run CCBC+GA algorithm
        t_start = time.perf_counter()
        solution = algorithm.solve(customers, depots)
        elapsed = time.perf_counter() - t_start

        ga_history = algorithm.last_ga_history
        algorithm_repr = str(algorithm)

        visualize_ga_convergence(ga_history)

        clustering_path = results_dir / f"{data_file.name}_clusters.json"
        routing_path = results_dir / f"{data_file.name}_routes.json"

        save_clustering_result(
            output_path=str(clustering_path),
            instance_name=data_file.name,
            algorithm_name=str(algorithm),
            clusters=algorithm.last_clusters,
        )
        save_routing_result(
            output_path=str(routing_path),
            instance_name=data_file.name,
            algorithm_name=str(algorithm),
            solution=solution,
        )

        clustering_file = str(clustering_path)
        routing_file = str(routing_path)

    print_solution_summary(solution)

    print_run_summary(
        solution=solution,
        elapsed=elapsed,
        ga_history=ga_history,
        clone_delta=cfg.ga.clone_delta,
        reference_cost=reference_solution.objective if reference_solution else None,
        algorithm_repr=algorithm_repr,
        clustering_file=clustering_file,
        routing_file=routing_file,
    )

    # Visualize
    if reference_solution is not None:
        visualize_comparison(
            instance,
            [reference_solution, solution],
            titles=[
                f"Reference (obj: {reference_solution.objective:.2f})",
                f"CCBC+GA (cost: {solution.total_cost():.2f})",
            ],
        )
    else:
        visualize_solution(
            instance,
            solution,
            title=f"CCBC+GA solution (cost: {solution.total_cost():.2f})",
        )
    
    if failures is not None and not args.no_simulate:
        simulated_solution, history_log = run_simulation(
            # instance=instance,
            initial_solution=solution,
            failures=failures,
            instance_name=data_file.name,
            algorithm=algorithm,
            cfg=cfg,
            # output_dir=base_dir / "data" / "processed" / "simulations" / data_file.name,
        )

        if reference_solution is not None:
            visualize_comparison(
                instance,
                [reference_solution, simulated_solution],
                titles=[
                    f"Reference (obj: {reference_solution.objective:.2f})",
                    f"CCBC+GA after simulation (cost: {simulated_solution.total_cost():.2f})",
                ],
            )
        else:
            visualize_solution(
                instance,
                simulated_solution,
                title=f"CCBC+GA after simulation (cost: {simulated_solution.total_cost():.2f})",
            )

        log_path = SIMULATION_LOG_DIR / f"{data_file.name}_log.json"
        validation_result = validate_simulation_log(log_path)
        blocked_edge_violations = validation_result["blocked_edge_violations"]
        unserved_customers = validation_result["unserved_customers"]

        try:
            log_path_for_cmd = Path(".") / log_path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            log_path_for_cmd = log_path
        log_arg = str(log_path_for_cmd).replace("/", "\\")

        print("\nTo animate the simulation log, run the following command:")
        print(
            f"python3 .\\src\\tools\\animate_simulation_log.py --log-file {log_arg}"
        )

        print_simulation_validation(validation_result, log_path)

        if blocked_edge_violations or unserved_customers:
            return 1
    else:
        if args.no_simulate:
            print("Simulation skipped (--no-simulate).")
        else:
            print("No failures file found; skipping simulation.")

    plt.show()  # keep all non-blocking windows open until manually closed
    return 0


if __name__ == "__main__":
    main()
