#!/usr/bin/env python3
"""
MCTS Scheduling Experiments Automation Script

This script automates building, submitting, and analyzing MCTS scheduling optimization
experiments across different configurations (problem sizes, cores, compilers, etc.)
"""

import os
import subprocess
import itertools
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path
import json
import sys

# Add scripts directory to sys.path to import experiment_manager
script_dir = Path(__file__).resolve().parent
if Path(script_dir / "scripts").exists():
    sys.path.append(str(script_dir / "scripts"))
    from experiment_manager import ExperimentManager

# --- Configuration ---
COMPILERS = {
    'gcc': {
        'modules': ['gcc/8.2.0', 'openmpi/4.1.4'],
        'command': 'mpicxx',
        'flags': ['-O3', '-march=native', '-fopenmp']
    },
    'oneapi': {
        'modules': [
            'oneapi_2024/tbb/2021.11',
            'oneapi_2024/compiler-rt/2024.0.0',
            'oneapi_2024/compiler/2024.0.0',
            'oneapi_2024/mpi/2021.11',
        ],
        'command': 'mpicxx',
        'flags': [
            '-O3',
            '-xHost',
            '-fiopenmp'
        ]
    }
}

# Default source file and directories
SOURCE_DIR = "src/mcts"
SOURCE_FILE = "main.cpp"
EXECUTABLE_BASE = "mcts_scheduler"
RESULTS_DIR = "experiment_results"
CLUSTER_PARTITION = "standard"  # Partition for Slurm
MAX_JOB_TIME = "01:00:00"  # Default max time for each job

# --- Helper Functions ---

def run_command(command, cwd=None, env=None):
    """Runs a shell command and returns output."""
    # Make a copy to modify if necessary
    processed_command = list(command)
    # If the command is to run a bash script, e.g., ['bash', 'script.sh'],
    # execute it as a login shell to ensure proper environment initialization (e.g., for modules).
    if processed_command[0] == 'bash' and len(processed_command) == 2 and \
       not processed_command[1].startswith('-') and processed_command[1].endswith('.sh'):
        processed_command.insert(1, '-l')

    print(f"Running: {' '.join(processed_command)}")
    try:
        result = subprocess.run(processed_command, cwd=cwd, env=env, text=True,
                               capture_output=True, check=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        raise  # Re-raise the exception

def generate_bash_script(commands, filename="temp_script.sh"):
    """Generates a bash script file containing the given commands with enhanced debugging."""
    with open(filename, "w") as f:
        f.write("#!/bin/bash\\n")
        # set -e will exit on error, set -x will trace commands
        f.write("set -ex\\n") 
        
        f.write("echo '--- Debug: Script Start ---' >&2\\n")
        f.write("echo '--- Debug: Initial PATH ---' >&2\\n")
        f.write("echo \"$PATH\" >&2\\n")
        f.write("echo '--- Debug: Current User ---' >&2\\n")
        f.write("whoami >&2\\n")
        f.write("echo '--- Debug: Initial Environment Variables (env) ---' >&2\\n")
        f.write("env >&2\\n")

        f.write("echo '--- Debug: Attempting to source /etc/profile ---' >&2\\n")
        # Try to source /etc/profile and report if it fails
        f.write("source /etc/profile || echo 'WARNING: Failed to source /etc/profile (exit status $?)' >&2\\n")
        f.write("echo '--- Debug: PATH after sourcing /etc/profile ---' >&2\\n")
        f.write("echo \"$PATH\" >&2\\n")
        f.write("echo '--- Debug: Environment Variables after sourcing /etc/profile ---' >&2\\n")
        f.write("env >&2\\n")
        
        f.write("echo '--- Debug: Checking for module command availability ---' >&2\\n")
        f.write("type module >&2 || echo 'WARNING: module command/function not found by type (exit status $?)' >&2\\n")
        f.write("command -v module >&2 || echo 'WARNING: module command not found by command -v (exit status $?)' >&2\\n")
        f.write("which module >&2 || echo 'WARNING: module command not found by which (exit status $?)' >&2\\n")

        f.write("echo '--- Debug: Executing provided commands ---' >&2\\n")
        for cmd_list in commands:
            # Properly quote arguments for the shell script
            quoted_cmd = [f"'{arg}'" if ' ' in arg else arg for arg in cmd_list]
            command_str = " ".join(quoted_cmd)
            
            f.write(f"echo 'Debug: Executing: {command_str}' >&2\\n")
            # Execute the command. If set -e is active, script will exit here on failure.
            f.write(f"{command_str}\\n")
            # This line will only be reached if the command_str succeeded or set -e is not active/triggered
            f.write(f"echo 'Debug: Command [{command_str}] finished with status: $?' >&2\\n")

        f.write("echo '--- Debug: Final PATH before script exit ---' >&2\\n")
        f.write("echo \"$PATH\" >&2\\n")
        f.write("echo '--- Debug: Bash script execution finished successfully ---' >&2\\n")
        
    os.chmod(filename, 0o755)  # Make executable
    return filename

def build_target(compiler_name, problem_size, simulations, parallelization, omp_threads, tracing):
    """Compiles the MCTS code using the specified compiler and settings."""
    config = COMPILERS[compiler_name]
    
    # Create executable name that encodes parameters
    opt_level = "O3" if "-O3" in config['flags'] else "O2"
    if "-xHost" in config['flags']:
        opt_level += "xHost"
    
    executable_name = f"{EXECUTABLE_BASE}_{problem_size}_{simulations//1000}k_{parallelization}_{compiler_name}_{opt_level}_omp{omp_threads}"
    if tracing:
        executable_name += "_trace"
    
    print(f"\n--- Building target: {executable_name} ---")
    
    module_load_cmds = [['module', 'purge']]
    module_load_cmds.extend([['module', 'load', mod] for mod in config['modules']])
    
    # Define compile flags
    compile_flags = config['flags'].copy()
    
    # Add problem-specific defines
    compile_flags.extend([
        f'-DMCTS_SIMULATIONS={simulations}',
        f'-DMCTS_PARALLELIZATION="{parallelization}"',
        f'-DMCTS_OMP_THREADS={omp_threads}'
    ])
    
    # Add tracing flags if needed
    if tracing:
        if compiler_name == 'oneapi':
            compile_flags.append('-trace')
        else:
            compile_flags.append('-DSCOREP_USER_ENABLE')
    
    # Create compile command
    source_path = os.path.join(SOURCE_DIR, SOURCE_FILE)
    compile_cmd = [config['command'], source_path] + compile_flags + ['-o', executable_name]
    all_cmds = module_load_cmds + [compile_cmd]
    
    script_file = generate_bash_script(all_cmds, f"compile_{executable_name}.sh")
    try:
        run_command(['bash', script_file])
        print(f"Build successful: {executable_name}")
    except Exception as e:
        print(f"Build failed: {e}")
    finally:
        os.remove(script_file)  # Clean up script
    
    return executable_name

def run_experiments(problem_instances, processes_list, omp_threads_list, 
                  simulations_list, parallelization_list, compilers_to_run, 
                  nodes_list=['auto'], cores_per_node=40,
                  tracing=False, time_limit=MAX_JOB_TIME, partition=CLUSTER_PARTITION):
    """Generates and submits Slurm jobs for parameter combinations."""
    print("\n--- Running Experiments ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    job_ids = []
    run_ids = []
    
    # Initialize experiment manager
    manager = ExperimentManager(os.getcwd())

    # Generate all parameter combinations
    combinations = list(itertools.product(
        problem_instances, processes_list, omp_threads_list, 
        simulations_list, parallelization_list, compilers_to_run
    ))
    
    print(f"Planning to submit {len(combinations)} experiment configurations")
    
    # Process nodes_list - either 'auto' or a list of node counts matching processes_list
    if nodes_list == ['auto']:
        # Auto-calculate nodes for each process count
        nodes_dict = {p: max(1, (p + cores_per_node - 1) // cores_per_node) for p in processes_list}
    else:
        # If specific node counts are provided, map them to process counts
        if len(nodes_list) != len(processes_list):
            print(f"Warning: Number of node specifications ({len(nodes_list)}) doesn't match process counts ({len(processes_list)})")
            print("Using auto-calculation for missing values")
            
            # Fill in missing values with auto-calculated ones
            nodes_dict = {}
            for i, p in enumerate(processes_list):
                if i < len(nodes_list) and nodes_list[i] != 'auto':
                    try:
                        nodes_dict[p] = int(nodes_list[i])
                    except ValueError:
                        nodes_dict[p] = max(1, (p + cores_per_node - 1) // cores_per_node)
                else:
                    nodes_dict[p] = max(1, (p + cores_per_node - 1) // cores_per_node)
        else:
            # Map each process count to its node count
            nodes_dict = {}
            for i, p in enumerate(processes_list):
                if nodes_list[i] == 'auto':
                    nodes_dict[p] = max(1, (p + cores_per_node - 1) // cores_per_node)
                else:
                    try:
                        nodes_dict[p] = int(nodes_list[i])
                    except ValueError:
                        print(f"Warning: Invalid node count '{nodes_list[i]}' for {p} processes, using auto-calculation")
                        nodes_dict[p] = max(1, (p + cores_per_node - 1) // cores_per_node)
    
    # Print node allocation information
    print("\nNode allocation for process counts:")
    for p, n in sorted(nodes_dict.items()):
        print(f"  {p} processes: {n} node(s)")
    
    for problem, processes, omp_threads, simulations, parallelization, compiler in combinations:
        # Create a readable problem instance name
        problem_name = os.path.basename(problem)
        
        # Set up experiment configs
        mcts_settings = {
            "simulations": simulations,
            # Assuming exploration is a fixed value or needs to be added as an arg
            "exploration": 1.414  
        }
        
        build_settings = {
            "parallelization": parallelization,
            "compiler": compiler,
            # Assuming optimization level needs mapping or direct use
            "optimization": "O3xHost" if "-xHost" in COMPILERS[compiler]['flags'] else "O3" 
        }
        
        parallel_settings = {
            "processes": processes,
            "omp_threads": omp_threads
        }

        # Get the node count for this specific process configuration
        nodes_for_run = nodes_dict.get(processes)
        if nodes_for_run is None:
             print(f"Warning: Could not determine node count for {processes} processes. Defaulting to auto-calculation.")
             nodes_for_run = max(1, (processes + cores_per_node - 1) // cores_per_node)

        # Setup experiment using ExperimentManager
        try:
            run_id = manager.setup_experiment(
                problem_instance=problem,
                mcts_settings=mcts_settings,
                build_settings=build_settings,
                parallel_settings=parallel_settings,
                tracing=tracing,
                nodes=nodes_for_run, # Pass nodes here
                cores_per_node=cores_per_node
            )
            run_ids.append(run_id)
            print(f"  Set up experiment: {run_id}")

            # Build executable (optional, could be done once per build config)
            # For simplicity, let's assume build happens separately or is handled by setup
            # manager.build_executable(run_id, build_settings, parallel_settings, tracing)

            # Generate Slurm script
            slurm_script_path = manager.generate_slurm_script(
                run_id=run_id,
                parallel_settings=parallel_settings,
                time_limit=time_limit,
                cores_per_node=cores_per_node,
                nodes=nodes_for_run # Pass nodes here
            )
            print(f"    Generated Slurm script: {slurm_script_path}")

            # Submit job
            job_id = manager.submit_job(run_id)
            if job_id:
                job_ids.append(job_id)
                print(f"    Submitted job with ID: {job_id}")
            else:
                print(f"    Failed to submit job for {run_id}")

        except Exception as e:
            print(f"Error processing configuration: {problem_name}, {processes}p, {omp_threads}t, {compiler}")
            print(f"  Error details: {e}")

    return job_ids, run_ids

def wait_for_jobs(job_ids, check_interval=60, max_wait_time=None):
    """Wait for submitted jobs to complete."""
    if not job_ids:
        print("No jobs to wait for.")
        return
    
    print(f"Waiting for {len(job_ids)} jobs to complete...")
    start_time = time.time()
    jobs_str = ','.join(job_ids)
    
    time.sleep(check_interval)  # Initial wait
    
    while True:
        try:
            stdout, _ = run_command(['squeue', '-h', '-j', jobs_str])
            running_jobs = stdout.strip().count("\n") + (1 if stdout.strip() else 0)
            
            if not stdout.strip():
                print("All jobs finished.")
                break
            
            print(f"{running_jobs} jobs still running. Waiting...")
            
            if max_wait_time and (time.time() - start_time > max_wait_time):
                print(f"Maximum wait time ({max_wait_time}s) exceeded. Continuing...")
                break
                
        except Exception as e:
            print(f"Error checking job status: {e}")
            print("Assuming jobs finished.")
            break
            
        time.sleep(check_interval)

def parse_results(run_ids=None):
    """Parses experiment results using results_parser.py."""
    print("\n--- Parsing Results ---")
    
    # Get list of all run directories if not provided
    if run_ids is None:
        run_ids = [d for d in os.listdir(RESULTS_DIR) 
                  if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    if not run_ids:
        print("No experiment runs found.")
        return None
        
    results = []
    for run_id in run_ids:
        run_dir = os.path.join(RESULTS_DIR, run_id)
        config_path = os.path.join(run_dir, "run_config.json")
        output_path = os.path.join(run_dir, "job_output.log")
        
        if not os.path.exists(config_path) or not os.path.exists(output_path):
            print(f"Warning: Missing files for run_id: {run_id}")
            continue
            
        # Read config
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Extract key parameters
            problem_instance = os.path.basename(config.get("problem_instance", "unknown"))
            processes = config.get("parallel_settings", {}).get("processes", 0)
            omp_threads = config.get("parallel_settings", {}).get("omp_threads", 0)
            total_cores = processes * omp_threads
            simulations = config.get("mcts_settings", {}).get("simulations", 0)
            parallelization = config.get("build_settings", {}).get("parallelization", "unknown")
            compiler = config.get("build_settings", {}).get("compiler", "unknown")
            nodes = config.get("nodes", None)  # Get node count if available
            
            # Read job output to extract performance data
            makespan = None
            execution_time = None
            mcts_nodes = None
            
            # Regex patterns
            makespan_pattern = re.compile(r"Makespan:\s+(\d+\.?\d*)")
            time_pattern = re.compile(r"Execution time:\s+(\d+\.?\d*)\s+seconds")
            nodes_pattern = re.compile(r"MCTS nodes:\s+(\d+)")
            
            with open(output_path, 'r') as f:
                content = f.read()
                
                # Extract key metrics
                makespan_match = makespan_pattern.search(content)
                time_match = time_pattern.search(content)
                nodes_match = nodes_pattern.search(content)
                
                if makespan_match:
                    makespan = float(makespan_match.group(1))
                if time_match:
                    execution_time = float(time_match.group(1))
                if nodes_match:
                    mcts_nodes = int(nodes_match.group(1))
            
            # Collect results
            result = {
                'run_id': run_id,
                'problem': problem_instance,
                'processes': processes,
                'omp_threads': omp_threads,
                'total_cores': total_cores,
                'simulations': simulations,
                'parallelization': parallelization,
                'compiler': compiler,
                'makespan': makespan,
                'execution_time': execution_time,
                'mcts_nodes': mcts_nodes,
                'nodes': nodes  # Include node count in results
            }
            
            # Check if all key metrics were found
            if None not in [makespan, execution_time]:
                results.append(result)
            else:
                print(f"Warning: Missing performance data in {run_id}")
                
        except Exception as e:
            print(f"Error parsing results for run_id {run_id}: {e}")
    
    # Convert to DataFrame
    if not results:
        print("No valid results found.")
        return None
        
    df = pd.DataFrame(results)
    csv_path = "mcts_experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    return df

def plot_results(df):
    """Generate plots from the experiment results."""
    if df is None or df.empty:
        print("No data to plot.")
        return
        
    print("\n--- Plotting Results ---")
    
    # Sort data for consistent plot lines
    df = df.sort_values(by=['problem', 'compiler', 'parallelization', 'total_cores'])
    
    # Create 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Execution time vs number of cores for each problem/compiler/parallelization
    plt.figure(figsize=(12, 8))
    
    for (prob, comp, para), group in df.groupby(['problem', 'compiler', 'parallelization']):
        label = f"{prob}, {comp}, {para}"
        plt.plot(group['total_cores'], group['execution_time'], marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Cores (MPI Processes × OMP Threads)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("MCTS Scheduling: Execution Time vs. Cores")
    plt.grid(True, which="both", ls="--")
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/mcts_execution_time.png")
    plt.close()
    
    # Plot 2: Speedup vs number of cores
    plt.figure(figsize=(12, 8))
    
    # Group data by problem/compiler/parallelization 
    for (prob, comp, para), group in df.groupby(['problem', 'compiler', 'parallelization']):
        # Calculate speedup relative to smallest core count
        base_time = group.loc[group['total_cores'].idxmin(), 'execution_time']
        speedup = base_time / group['execution_time']
        
        label = f"{prob}, {comp}, {para}"
        plt.plot(group['total_cores'], speedup, marker='o', linestyle='-', label=label)
        
        # Add ideal speedup line (using same min core count as the data)
        min_cores = group['total_cores'].min()
        cores_range = sorted(group['total_cores'].unique())
        ideal_speedup = [c/min_cores for c in cores_range]
        plt.plot(cores_range, ideal_speedup, 'k--', alpha=0.3)
    
    plt.xlabel("Number of Cores (MPI Processes × OMP Threads)")
    plt.ylabel("Speedup")
    plt.title("MCTS Scheduling: Speedup vs. Cores")
    plt.grid(True, which="both", ls="--")
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/mcts_speedup.png")
    plt.close()
    
    # Plot 3: Parallel efficiency vs number of cores
    plt.figure(figsize=(12, 8))
    
    for (prob, comp, para), group in df.groupby(['problem', 'compiler', 'parallelization']):
        # Calculate efficiency
        base_time = group.loc[group['total_cores'].idxmin(), 'execution_time']
        min_cores = group['total_cores'].min()
        
        efficiency = (base_time / group['execution_time']) / (group['total_cores'] / min_cores)
        
        label = f"{prob}, {comp}, {para}"
        plt.plot(group['total_cores'], efficiency, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Cores (MPI Processes × OMP Threads)")
    plt.ylabel("Parallel Efficiency")
    plt.title("MCTS Scheduling: Parallel Efficiency vs. Cores")
    plt.grid(True, which="both", ls="--")
    plt.xscale('log', base=2)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.ylim(0, 1.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/mcts_efficiency.png")
    plt.close()
    
    # Plot 4: Makespan vs number of cores
    plt.figure(figsize=(12, 8))
    
    for (prob, comp, para), group in df.groupby(['problem', 'compiler', 'parallelization']):
        label = f"{prob}, {comp}, {para}"
        plt.plot(group['total_cores'], group['makespan'], marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Cores (MPI Processes × OMP Threads)")
    plt.ylabel("Makespan")
    plt.title("MCTS Scheduling: Solution Makespan vs. Cores")
    plt.grid(True, which="both", ls="--")
    plt.xscale('log', base=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/mcts_makespan.png")
    plt.close()
    
    # Plot 5: Execution time breakdown by parallelization strategy
    if len(df['parallelization'].unique()) > 1:
        plt.figure(figsize=(14, 8))
        
        for comp, comp_group in df.groupby('compiler'):
            plt.subplot(1, len(df['compiler'].unique()), df['compiler'].unique().tolist().index(comp) + 1)
            
            for para, para_group in comp_group.groupby('parallelization'):
                for prob, prob_group in para_group.groupby('problem'):
                    label = f"{prob}, {para}"
                    plt.plot(prob_group['total_cores'], prob_group['execution_time'], 
                             marker='o', linestyle='-', label=label)
            
            plt.xlabel("Number of Cores")
            plt.ylabel("Execution Time (seconds)")
            plt.title(f"Compiler: {comp}")
            plt.grid(True, which="both", ls="--")
            plt.xscale('log', base=2)
            plt.yscale('log', base=10)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("plots/mcts_parallelization_comparison.png")
        plt.close()
    
    print(f"Plots saved to 'plots/' directory.")

def main():
    """Main function to parse arguments and run workflow."""
    parser = argparse.ArgumentParser(description="Automate MCTS Scheduling Experiments")
    parser.add_argument('--build', action='store_true', help="Build targets for specified configurations.")
    parser.add_argument('--run', action='store_true', help="Run experiments (generates and submits jobs).")
    parser.add_argument('--wait', action='store_true', help="Wait for submitted jobs to complete.")
    parser.add_argument('--parse', action='store_true', help="Parse results from output logs.")
    parser.add_argument('--plot', action='store_true', help="Generate plots from parsed results.")
    
    # Build options
    parser.add_argument('--compilers', nargs='+', choices=COMPILERS.keys(), default=['gcc'], 
                        help="Compilers to use.")
    parser.add_argument('--problems', nargs='+', default=['problem_instances/sample_problem01.txt'], 
                        help="Problem instance files.")
    parser.add_argument('--processes', nargs='+', type=int, default=[1, 2, 4, 8], 
                        help="MPI processes to test.")
    parser.add_argument('--omp-threads', nargs='+', type=int, default=[1, 2, 4], 
                        help="OpenMP threads per process to test.")
    parser.add_argument('--simulations', nargs='+', type=int, default=[10000], 
                        help="MCTS simulation counts to test.")
    parser.add_argument('--parallelization', nargs='+', choices=['treeMPI', 'rootMPI'], default=['treeMPI'], 
                        help="Parallelization strategy to use.")
    parser.add_argument('--nodes', nargs='+', default=['auto'], 
                        help="Number of nodes for each process count. Specify 'auto' for automatic calculation, or provide specific counts for each process count.")
    parser.add_argument('--tracing', action='store_true', help="Enable execution tracing.")
    
    # Job control options
    parser.add_argument('--time-limit', type=str, default=MAX_JOB_TIME, 
                        help="Job time limit in HH:MM:SS format.")
    parser.add_argument('--partition', type=str, default=CLUSTER_PARTITION, 
                        help="Slurm partition to submit jobs to.")
    parser.add_argument('--max-wait', type=int, default=7200, 
                        help="Maximum time to wait for jobs (seconds).")
    parser.add_argument('--cores-per-node', type=int, default=40,
                        help="Number of CPU cores per compute node.")
    
    # Run all steps
    parser.add_argument('--all', action='store_true', 
                        help="Run build, run, wait, parse, and plot steps.")
    
    args = parser.parse_args()
    
    if args.all:
        args.build = True
        args.run = True
        args.wait = True
        args.parse = True
        args.plot = True
    
    if not (args.build or args.run or args.wait or args.parse or args.plot):
        parser.print_help()
        return
    
    # Track job IDs and run IDs across steps
    job_ids = []
    run_ids = []
    
    # --- Build Phase ---
    if args.build:
        print("\n=== Build Phase ===")
        for compiler, problem, simulations, parallelization, omp_threads in itertools.product(
            args.compilers, args.problems, args.simulations, args.parallelization, args.omp_threads):
            
            # Use basename of problem file as identifier
            problem_name = os.path.basename(problem)
            
            build_target(
                compiler_name=compiler,
                problem_size=problem_name,
                simulations=simulations,
                parallelization=parallelization,
                omp_threads=omp_threads,
                tracing=args.tracing
            )
      # --- Run Phase ---
    if args.run:
        print("\n=== Run Phase ===")
        
        # Process the nodes argument
        nodes_arg = args.nodes
        
        job_ids, run_ids = run_experiments(
            problem_instances=args.problems,
            processes_list=args.processes,
            omp_threads_list=args.omp_threads,
            simulations_list=args.simulations,
            parallelization_list=args.parallelization,
            compilers_to_run=args.compilers,
            nodes_list=nodes_arg,
            cores_per_node=args.cores_per_node,
            tracing=args.tracing,
            time_limit=args.time_limit,
            partition=args.partition
        )
    
    # --- Wait Phase ---
    if args.wait and job_ids:
        print("\n=== Wait Phase ===")
        wait_for_jobs(job_ids, max_wait_time=args.max_wait)
    
    # --- Parse Phase ---
    results_df = None
    if args.parse:
        print("\n=== Parse Phase ===")
        results_df = parse_results(run_ids if run_ids else None)
    
    # --- Plot Phase ---
    if args.plot:
        print("\n=== Plot Phase ===")
        if results_df is None:
            # Try to load existing CSV if parse wasn't run
            try:
                results_df = pd.read_csv("mcts_experiment_results.csv")
            except FileNotFoundError:
                print("Error: mcts_experiment_results.csv not found. Run with --parse first.")
        
        if results_df is not None:
            plot_results(results_df)
    
    print("\nMCTS experiments automation script finished.")

if __name__ == "__main__":
    main()
