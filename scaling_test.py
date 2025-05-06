#!/usr/bin/env python3
"""
Multi-Node Scaling Test Script for MCTS Scheduling Optimization

This script performs a comprehensive multi-node scaling analysis for Monte Carlo
Tree Search scheduling optimization. It systematically evaluates performance
across different node counts, MPI process counts, and OpenMP thread configurations.
"""

import os
import sys
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add necessary paths to import experiment_manager
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir / "scripts"))
from experiment_manager import ExperimentManager

def run_multi_node_scaling_test(
        problem_instance,
        process_counts,
        node_counts=None,
        omp_threads=[1, 2, 4],
        simulations=50000,
        compiler="gcc",
        parallelization="treeMPI",
        time_limit="04:00:00",
        cores_per_node=40,
        base_dir=None,
        wait_for_completion=True):
    """
    Run a comprehensive multi-node scaling test.
    
    Args:
        problem_instance: Path to problem instance file
        process_counts: List of MPI process counts to test
        node_counts: List of node counts (one per process count), or None for auto
        omp_threads: List of OpenMP thread counts to test
        simulations: Number of MCTS simulations
        compiler: Compiler to use (gcc, oneapi)
        parallelization: Parallelization strategy (treeMPI, rootMPI)
        time_limit: Job time limit
        cores_per_node: CPU cores per compute node
        base_dir: Base directory for experiments
        wait_for_completion: Whether to wait for jobs to complete
    
    Returns:
        List of run_ids for the submitted jobs
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    manager = ExperimentManager(base_dir)
    run_ids = []
    job_ids = []
    
    # Set up automatic node allocation if not provided
    if node_counts is None:
        node_counts = [max(1, (p + cores_per_node - 1) // cores_per_node) for p in process_counts]
    elif len(node_counts) != len(process_counts):
        raise ValueError(f"Number of node counts ({len(node_counts)}) must match number of process counts ({len(process_counts)})")
    
    # Create combination of process counts, nodes, and threads
    combinations = []
    for i, (processes, nodes) in enumerate(zip(process_counts, node_counts)):
        for threads in omp_threads:
            combinations.append((processes, nodes, threads))
    
    print(f"Planning {len(combinations)} scaling experiments:")
    for processes, nodes, threads in combinations:
        print(f"  {processes} processes on {nodes} node(s) with {threads} thread(s) per process")
    
    # Set up and submit experiments
    for processes, nodes, threads in combinations:
        # Prepare experiment configuration
        mcts_settings = {
            "simulations": simulations,
            "exploration": 1.0  # Default exploration parameter
        }
        
        build_settings = {
            "parallelization": parallelization,
            "compiler": compiler,
            "optimization": "O3"
        }
        
        parallel_settings = {
            "processes": processes,
            "omp_threads": threads
        }
        
        # Set up experiment
        try:
            run_id = manager.setup_experiment(
                problem_instance=problem_instance,
                mcts_settings=mcts_settings,
                build_settings=build_settings,
                parallel_settings=parallel_settings,
                nodes=nodes,
                cores_per_node=cores_per_node
            )
            
            run_ids.append(run_id)
            
            # Build executable
            executable_path = manager.build_executable(
                run_id=run_id,
                build_settings=build_settings,
                parallel_settings=parallel_settings,
                tracing=False
            )
            
            # Generate and submit job
            slurm_script_path = manager.generate_slurm_script(
                run_id=run_id,
                parallel_settings=parallel_settings,
                time_limit=time_limit,
                nodes=nodes,
                cores_per_node=cores_per_node
            )
            
            job_id = manager.submit_job(run_id)
            if job_id:
                job_ids.append(job_id)
                print(f"Submitted job {job_id} for run_id: {run_id} ({processes} processes on {nodes} node(s) with {threads} thread(s))")
            
        except Exception as e:
            print(f"Error setting up experiment: {e}")
    
    # Wait for jobs to complete if requested
    if wait_for_completion and job_ids:
        print(f"\nWaiting for {len(job_ids)} jobs to complete...")
        
        check_interval = 60  # seconds
        max_wait_time = 7200  # 2 hours
        start_time = time.time()
        
        time.sleep(check_interval)  # Initial wait
        
        while True:
            try:
                import subprocess
                jobs_str = ','.join(job_ids)
                result = subprocess.run(['squeue', '-h', '-j', jobs_str], capture_output=True, text=True)
                
                running_jobs = result.stdout.strip().count('\n') + (1 if result.stdout.strip() else 0)
                
                if not result.stdout.strip():
                    print("All jobs finished.")
                    break
                
                print(f"{running_jobs} jobs still running. Waiting...")
                
                if time.time() - start_time > max_wait_time:
                    print(f"Maximum wait time ({max_wait_time}s) exceeded. Continuing...")
                    break
                    
            except Exception as e:
                print(f"Error checking job status: {e}")
                print("Assuming jobs finished.")
                break
                
            time.sleep(check_interval)
    
    return run_ids

def analyze_scaling_results(run_ids, base_dir=None, output_dir="scaling_analysis"):
    """
    Analyze results from multi-node scaling tests.
    
    Args:
        run_ids: List of experiment run IDs
        base_dir: Base directory for experiments
        output_dir: Directory for output plots
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    experiment_results_dir = os.path.join(base_dir, "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Parse results from each run
    for run_id in run_ids:
        run_dir = os.path.join(experiment_results_dir, run_id)
        config_path = os.path.join(run_dir, "run_config.json")
        output_path = os.path.join(run_dir, "job_output.log")
        
        if not os.path.exists(config_path) or not os.path.exists(output_path):
            print(f"Warning: Missing files for run_id: {run_id}")
            continue
        
        # Read config
        try:
            import json
            import re
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract parameters
            processes = config.get("parallel_settings", {}).get("processes", 0)
            omp_threads = config.get("parallel_settings", {}).get("omp_threads", 0)
            total_cores = processes * omp_threads
            simulations = config.get("mcts_settings", {}).get("simulations", 0)
            parallelization = config.get("build_settings", {}).get("parallelization", "unknown")
            compiler = config.get("build_settings", {}).get("compiler", "unknown")
            nodes = config.get("nodes", None)
            
            # Read performance data
            makespan = None
            execution_time = None
            mcts_nodes = None
            
            # Regex patterns
            makespan_pattern = re.compile(r"Makespan:\s+(\d+\.?\d*)")
            time_pattern = re.compile(r"Execution time:\s+(\d+\.?\d*)\s+seconds")
            nodes_pattern = re.compile(r"MCTS nodes:\s+(\d+)")
            
            with open(output_path, 'r') as f:
                content = f.read()
                
                makespan_match = makespan_pattern.search(content)
                time_match = time_pattern.search(content)
                nodes_match = nodes_pattern.search(content)
                
                if makespan_match:
                    makespan = float(makespan_match.group(1))
                if time_match:
                    execution_time = float(time_match.group(1))
                if nodes_match:
                    mcts_nodes = int(nodes_match.group(1))
            
            # Add to results
            results.append({
                'run_id': run_id,
                'processes': processes,
                'omp_threads': omp_threads,
                'total_cores': total_cores,
                'nodes': nodes,
                'simulations': simulations,
                'parallelization': parallelization,
                'compiler': compiler,
                'makespan': makespan,
                'execution_time': execution_time,
                'mcts_nodes': mcts_nodes
            })
            
        except Exception as e:
            print(f"Error parsing results for run_id {run_id}: {e}")
    
    # Convert to DataFrame
    if not results:
        print("No valid results found.")
        return None
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "scaling_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Generate scaling plots
    if not df.empty:
        plt.figure(figsize=(10, 6))
        
        # Group by node count and threads
        for (threads, nodes), group in df.groupby(['omp_threads', 'nodes']):
            if nodes is not None:  # Skip if nodes is None
                group = group.sort_values('processes')
                plt.plot(group['processes'], group['execution_time'], 
                         marker='o', linestyle='-', 
                         label=f"{threads} thread(s), {nodes} node(s)")
        
        plt.xlabel("MPI Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.title("MCTS Scaling: Execution Time vs. MPI Processes")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "execution_time_vs_processes.png"))
        
        # Create efficiency plots
        plt.figure(figsize=(10, 6))
        
        for (threads,), group in df.groupby(['omp_threads']):
            # For each thread count, find base performance (single node)
            group = group.sort_values(['nodes', 'processes'])
            if 'nodes' in group.columns and not group['nodes'].isna().all():
                # Get baseline for single node
                base_group = group[group['nodes'] == 1]
                if not base_group.empty:
                    base_time = base_group['execution_time'].iloc[0]
                    base_cores = base_group['total_cores'].iloc[0]
                    
                    # Calculate efficiency: (base_time / time) / (cores / base_cores)
                    efficiency = (base_time / group['execution_time']) / (group['total_cores'] / base_cores)
                    
                    plt.plot(group['nodes'], efficiency * 100, 
                             marker='o', linestyle='-', 
                             label=f"{threads} thread(s)")
        
        plt.xlabel("Number of Nodes")
        plt.ylabel("Parallel Efficiency (%)")
        plt.title("MCTS Scaling: Node Scaling Efficiency")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "node_scaling_efficiency.png"))
        
        print(f"Plots saved to {output_dir}")
    
    return df

def main():
    """Main function for multi-node scaling tests."""
    parser = argparse.ArgumentParser(description="Multi-Node Scaling Test for MCTS Scheduling Optimization")
    
    parser.add_argument("--problem", required=True, 
                        help="Path to problem instance file")
    parser.add_argument("--processes", nargs='+', type=int, required=True,
                        help="MPI process counts to test")
    parser.add_argument("--nodes", nargs='+', type=int,
                        help="Node counts for each process count (optional, auto-calculated if not provided)")
    parser.add_argument("--omp-threads", nargs='+', type=int, default=[1, 2, 4],
                        help="OpenMP thread counts per process to test")
    parser.add_argument("--simulations", type=int, default=50000,
                        help="Number of MCTS simulations")
    parser.add_argument("--compiler", choices=["gcc", "oneapi"], default="gcc",
                        help="Compiler to use")
    parser.add_argument("--parallelization", choices=["treeMPI", "rootMPI"], default="treeMPI",
                        help="Parallelization strategy")
    parser.add_argument("--time-limit", type=str, default="04:00:00",
                        help="Job time limit in HH:MM:SS format")
    parser.add_argument("--cores-per-node", type=int, default=40,
                        help="Number of CPU cores per compute node")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for jobs to complete")
    parser.add_argument("--output-dir", type=str, default="scaling_analysis",
                        help="Directory for output plots and results")
    
    args = parser.parse_args()
    
    # Run multi-node scaling test
    run_ids = run_multi_node_scaling_test(
        problem_instance=args.problem,
        process_counts=args.processes,
        node_counts=args.nodes,
        omp_threads=args.omp_threads,
        simulations=args.simulations,
        compiler=args.compiler,
        parallelization=args.parallelization,
        time_limit=args.time_limit,
        cores_per_node=args.cores_per_node,
        wait_for_completion=not args.no_wait
    )
    
    # Analyze results if jobs completed
    if not args.no_wait:
        analyze_scaling_results(run_ids, output_dir=args.output_dir)
    else:
        print(f"\nSubmitted {len(run_ids)} scaling test jobs.")
        print("Run the following command later to analyze results:")
        run_ids_str = " ".join(run_ids)
        print(f"python scaling_test.py --analyze-only --run-ids {run_ids_str}")

if __name__ == "__main__":
    main()
