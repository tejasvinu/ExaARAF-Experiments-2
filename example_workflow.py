#!/usr/bin/env python3
"""
Example workflow script for MCTS scheduling optimization experiments.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add scripts directory to sys.path
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir / "scripts"
sys.path.append(str(scripts_dir))

from experiment_manager import ExperimentManager

def run_example_experiment(base_dir, problem_file, simulations=10000):
    """
    Run an example experiment workflow.
    
    Args:
        base_dir: Base directory for experiments
        problem_file: Path to problem instance file
        simulations: Number of MCTS simulations
    """
    print(f"Starting example experiment workflow...")
    
    # Initialize experiment manager
    manager = ExperimentManager(base_dir)
    
    # Prepare settings
    mcts_settings = {
        "simulations": simulations,
        "exploration": 1.0
    }
    
    build_settings = {
        "parallelization": "treeMPI",
        "compiler": "gcc",  # Use what's available on your system
        "optimization": "O3"
    }
    
    parallel_settings = {
        "processes": 2,  # Adjust based on your system
        "omp_threads": 2  # Adjust based on your system
    }
    
    # Setup experiment
    print("\n=== Setting up experiment ===")
    run_id = manager.setup_experiment(
        problem_instance=problem_file,
        mcts_settings=mcts_settings,
        build_settings=build_settings,
        parallel_settings=parallel_settings,
        tracing=False
    )
    
    print(f"Experiment set up with run_id: {run_id}")
    
    # Build executable (in a real scenario, this would compile the code)
    print("\n=== Building executable ===")
    executable_path = manager.build_executable(
        run_id=run_id,
        build_settings=build_settings,
        parallel_settings=parallel_settings,
        tracing=False
    )
    
    print(f"Executable would be built at: {executable_path}")
    
    # Generate Slurm script
    print("\n=== Generating Slurm script ===")
    slurm_script_path = manager.generate_slurm_script(
        run_id=run_id,
        parallel_settings=parallel_settings
    )
    
    print(f"Slurm script generated at: {slurm_script_path}")
    
    # Instead of submitting to Slurm, let's run a local simulation
    print("\n=== Running local simulation ===")
    run_dir = Path(base_dir) / "experiment_results" / run_id
    
    # Create a dummy solution file
    solution_file = run_dir / "solution.json"
    with open(solution_file, 'w') as f:
        f.write("""
{
  "makespan": 42,
  "execution_time": 5.67,
  "nodes_explored": 1234,
  "simulations_performed": 5000,
  "task_assignments": [
    {"task": 0, "machine": 0},
    {"task": 1, "machine": 1},
    {"task": 2, "machine": 2},
    {"task": 3, "machine": 0},
    {"task": 4, "machine": 1}
  ],
  "schedule": [0, 1, 2, 3, 4]
}
""")
    
    # Create a dummy job output log
    output_log = run_dir / "job_output.log"
    with open(output_log, 'w') as f:
        f.write("""
Starting job for run_id: sample_problem01_sims10k_treeMPI_gcc_O3_p2_omp2_traceOff_20250505123456_abcd
Timestamp: Mon May 5 12:34:56 UTC 2025
Nodes: 1, Processes: 2, OMP Threads: 2

Loaded problem with 10 tasks and 3 machines
Running MCTS with 10000 simulations, 2 MPI processes and 2 OMP threads per process
Rank 0: 0 / 5000 simulations completed
Rank 0: 1000 / 5000 simulations completed
Rank 0: 2000 / 5000 simulations completed
Rank 0: 3000 / 5000 simulations completed
Rank 0: 4000 / 5000 simulations completed
MCTS search completed in 5.67 seconds
Makespan: 42
MCTS nodes: 1234
MCTS simulations: 5000
Execution time: 5.67 seconds
Makespan: 42
Solution written to [solution path]

Job completed at Mon May 5 12:35:02 UTC 2025
""")
    
    # Create dummy system metrics
    metrics_file = run_dir / "system_metrics.pid12345.log"
    with open(metrics_file, 'w') as f:
        f.write("""timestamp,cpu_pct,mem_rss_kb,mem_vms_kb,io_read_kb,io_write_kb,cpu_user_time,cpu_system_time
1714906496,125.6,245678,356789,1234,5678,1234,567
1714906501,156.4,278901,389012,2345,6789,2345,678
1714906506,184.2,312345,423456,3456,7890,3456,789
1714906511,198.7,345678,456789,4567,8901,4567,890
1714906516,172.3,312345,423456,5678,9012,5678,901
1714906521,143.5,278901,389012,6789,10123,6789,1012
""")
    
    # Wait a bit to simulate job execution
    print("Simulating job execution...")
    time.sleep(2)
    
    # Parse results (this would normally be done separately)
    print("\n=== Parsing results ===")
    results_parser_path = scripts_dir / "results_parser.py"
    output_csv = run_dir / "results.csv"
    
    # Construct command to run results parser
    cmd = [
        sys.executable,
        str(results_parser_path),
        "--base-dir", str(base_dir),
        "--output", str(output_csv),
        "--run-ids", run_id
    ]
    
    # In a real scenario, we would run this command
    # For demonstration, we'll just print it
    print(f"Would run: {' '.join(cmd)}")
    
    print("\nExample workflow completed.")
    print(f"Results directory: {run_dir}")


if __name__ == "__main__":
    # Get base directory from current file location
    base_dir = str(Path(__file__).resolve().parent)
    problem_file = os.path.join(base_dir, "problem_instances", "sample_problem01.txt")
    
    # Run example workflow
    run_example_experiment(base_dir, problem_file)
