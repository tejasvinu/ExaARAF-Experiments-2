#!/usr/bin/env python3
"""
MCTS Scheduling Optimization Experiment Manager

This script automates the setup, configuration, and submission of MCTS
scheduling optimization experiments with comprehensive logging.
"""

import os
import sys
import json
import time
import uuid
import subprocess
import datetime
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class ExperimentManager:
    """Manages MCTS scheduling optimization experiments."""

    def __init__(self, base_dir: str):
        """
        Initialize the experiment manager.
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.src_dir = self.base_dir / "src"
        self.scripts_dir = self.base_dir / "scripts"
        self.problem_instances_dir = self.base_dir / "problem_instances"
        self.experiment_results_dir = self.base_dir / "experiment_results"
        self.utils_dir = self.base_dir / "utils"
        
        # Ensure all directories exist
        for dir_path in [self.src_dir, self.scripts_dir, self.problem_instances_dir, 
                         self.experiment_results_dir, self.utils_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
            
    def generate_run_id(self, 
                        problem_instance: str, 
                        simulations: int, 
                        parallelization: str, 
                        compiler: str, 
                        optimization: str, 
                        processes: int, 
                        omp_threads: int, 
                        tracing: bool) -> str:
        """
        Generate a unique run ID based on experiment parameters.
        
        Args:
            problem_instance: Name of the problem instance
            simulations: Number of simulations
            parallelization: Parallelization strategy (e.g., 'treeMPI', 'rootMPI')
            compiler: Compiler used (e.g., 'gcc', 'icc', 'oneapi')
            optimization: Optimization flags (e.g., 'O3xHost')
            processes: Number of MPI processes
            omp_threads: Number of OpenMP threads per process
            tracing: Whether tracing is enabled
            
        Returns:
            Unique run ID string
        """
        # Extract problem name without extension
        problem_name = os.path.splitext(os.path.basename(problem_instance))[0]
        
        # Format components
        sim_str = f"sims{simulations // 1000}k" if simulations >= 1000 else f"sims{simulations}"
        trace_str = "traceOn" if tracing else "traceOff"
        
        # Combine components
        run_id = f"{problem_name}_{sim_str}_{parallelization}_{compiler}_{optimization}_p{processes}_omp{omp_threads}_{trace_str}"
        
        # Add timestamp and unique suffix to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        unique_suffix = uuid.uuid4().hex[:4]
        
        return f"{run_id}_{timestamp}_{unique_suffix}"
    def setup_experiment(self, 
                         problem_instance: str,
                         mcts_settings: Dict[str, Any],
                         build_settings: Dict[str, Any],
                         parallel_settings: Dict[str, Any],
                         tracing: bool = False,
                         nodes: Optional[int] = None,
                         cores_per_node: int = 40) -> str:
        """
        Set up a new experiment with the given parameters.
        
        Args:
            problem_instance: Path to problem instance file
            mcts_settings: MCTS algorithm settings
            build_settings: Compiler and build settings
            parallel_settings: MPI and OpenMP settings
            tracing: Whether to enable tracing
            nodes: Number of nodes to request in job submission
            cores_per_node: Number of CPU cores per compute node
            
        Returns:
            The generated run ID
        """
        # Extract settings
        simulations = mcts_settings.get("simulations", 10000)
        parallelization = build_settings.get("parallelization", "treeMPI")
        compiler = build_settings.get("compiler", "gcc")
        optimization = build_settings.get("optimization", "O3")
        processes = parallel_settings.get("processes", 1)
        omp_threads = parallel_settings.get("omp_threads", 1)
        
        # Generate run ID
        run_id = self.generate_run_id(
            problem_instance=problem_instance,
            simulations=simulations,
            parallelization=parallelization,
            compiler=compiler,
            optimization=optimization,
            processes=processes,
            omp_threads=omp_threads,
            tracing=tracing
        )
        
        # Create experiment directory
        run_dir = self.experiment_results_dir / run_id
        run_dir.mkdir(exist_ok=False)
        
        # Create subdirectories
        trace_dir = run_dir / "trace_files"
        trace_dir.mkdir(exist_ok=True)
        
        # Prepare configuration
        config = {
            "run_id": run_id,
            "submission_timestamp": datetime.datetime.now().isoformat(),
            "problem_instance": problem_instance,
            "mcts_settings": mcts_settings,
            "build_settings": build_settings,
            "parallel_settings": parallel_settings,
            "tracing": tracing,
            "executable_path": "",  # Will be filled after build
            "git_hash": self._get_git_hash(),
            "slurm_job_id": None  # Will be filled after submission
        }
        
        # Save configuration
        config_path = run_dir / "run_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Copy monitoring script to run directory
        monitor_script_src = self.scripts_dir / "monitor_script.sh"
        monitor_script_dst = run_dir / "monitor_script.sh"
        if monitor_script_src.exists():
            shutil.copy(str(monitor_script_src), str(monitor_script_dst))
        else:
            self._create_monitor_script(monitor_script_dst)
        
        return run_id
    
    def _get_git_hash(self) -> str:
        """Get current Git hash or placeholder if not in a Git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "no_git_hash_available"
    
    def _create_monitor_script(self, script_path: Path) -> None:
        """Create a system monitoring script."""
        script_content = """#!/bin/bash
# System resource monitoring script for MCTS experiments

# Get the job ID from Slurm or use PID if not in Slurm
if [ -n "${SLURM_JOB_ID}" ]; then
    MONITOR_ID="${SLURM_JOB_ID}"
    METRIC_FILE="system_metrics.job${SLURM_JOB_ID}.log"
else
    MONITOR_ID="$$"
    METRIC_FILE="system_metrics.pid$$.log"
fi

# Create header for metrics file
echo "timestamp,cpu_pct,mem_rss_kb,mem_vms_kb,io_read_kb,io_write_kb,cpu_user_time,cpu_system_time" > "$METRIC_FILE"

# Function to get process stats
get_process_stats() {
    local pids=("$@")
    local stats=""
    
    # Sum resources across all relevant processes
    local total_cpu=0
    local total_rss=0
    local total_vms=0
    local total_read=0
    local total_write=0
    local total_utime=0
    local total_stime=0
    
    for pid in "${pids[@]}"; do
        if [ -d "/proc/$pid" ]; then
            # Get CPU usage
            local cpu_stat=$(top -b -n 1 -p $pid | grep $pid)
            local cpu_pct=$(echo "$cpu_stat" | awk '{print $9}')
            
            # Get memory usage
            local mem_stat=$(cat /proc/$pid/status | grep -E 'VmRSS|VmSize')
            local rss=$(echo "$mem_stat" | grep 'VmRSS' | awk '{print $2}')
            local vms=$(echo "$mem_stat" | grep 'VmSize' | awk '{print $2}')
            
            # Get I/O stats
            local io_stat=$(cat /proc/$pid/io 2>/dev/null)
            local read_bytes=$(echo "$io_stat" | grep 'read_bytes' | awk '{print $2}')
            local write_bytes=$(echo "$io_stat" | grep 'write_bytes' | awk '{print $2}')
            
            # Get CPU time
            local cpu_time=$(cat /proc/$pid/stat | awk '{print $14, $15}')
            local utime=$(echo "$cpu_time" | awk '{print $1}')
            local stime=$(echo "$cpu_time" | awk '{print $2}')
            
            # Add to totals
            total_cpu=$(echo "$total_cpu + $cpu_pct" | bc)
            total_rss=$(echo "$total_rss + $rss" | bc)
            total_vms=$(echo "$total_vms + $vms" | bc)
            total_read=$(echo "$total_read + $read_bytes" | bc)
            total_write=$(echo "$total_write + $write_bytes" | bc)
            total_utime=$(echo "$total_utime + $utime" | bc)
            total_stime=$(echo "$total_stime + $stime" | bc)
        fi
    done
    
    # Return comma-separated stats
    total_read_kb=$(echo "$total_read / 1024" | bc)
    total_write_kb=$(echo "$total_write / 1024" | bc)
    echo "$total_cpu,$total_rss,$total_vms,$total_read_kb,$total_write_kb,$total_utime,$total_stime"
}

# Monitor interval in seconds
INTERVAL=5

# If we're in a Slurm job, get all PIDs associated with it
if [ -n "${SLURM_JOB_ID}" ]; then
    echo "Monitoring all processes in Slurm job ${SLURM_JOB_ID}"
    
    while true; do
        # Get all PIDs in this Slurm job
        pids=($(ps -u $USER -o pid= --sort=-pid | xargs -I{} sh -c "grep -l ${SLURM_JOB_ID} /proc/{}/environ 2>/dev/null | grep -o '[0-9]*'"))
        
        if [ ${#pids[@]} -eq 0 ]; then
            echo "No processes found for job ${SLURM_JOB_ID}, sleeping..."
            sleep $INTERVAL
            continue
        fi
        
        # Get timestamp
        timestamp=$(date +%s)
        
        # Get process stats
        stats=$(get_process_stats "${pids[@]}")
        
        # Write to log
        echo "$timestamp,$stats" >> "$METRIC_FILE"
        
        sleep $INTERVAL
    done
else
    # Non-Slurm mode: monitor a specific process and its children
    target_pid=$1
    if [ -z "$target_pid" ]; then
        echo "Error: When not in Slurm, you must provide a PID to monitor"
        exit 1
    fi
    
    echo "Monitoring process $target_pid and its children"
    
    while kill -0 $target_pid 2>/dev/null; do
        # Get all child PIDs
        pids=($(pstree -p $target_pid | grep -o '([0-9]\+)' | grep -o '[0-9]\+'))
        pids+=($target_pid)
        
        # Get timestamp
        timestamp=$(date +%s)
        
        # Get process stats
        stats=$(get_process_stats "${pids[@]}")
        
        # Write to log
        echo "$timestamp,$stats" >> "$METRIC_FILE"
        
        sleep $INTERVAL
    done
fi

echo "Monitoring finished"
"""
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def build_executable(self, run_id: str, 
                         build_settings: Dict[str, Any], 
                         parallel_settings: Dict[str, Any],
                         tracing: bool) -> str:
        """
        Build the MCTS executable with the specified settings.
        
        Args:
            run_id: The experiment run ID
            build_settings: Compiler and build settings
            parallel_settings: MPI and OpenMP settings
            tracing: Whether to enable tracing
            
        Returns:
            Path to the built executable
        """
        run_dir = self.experiment_results_dir / run_id
        
        # Define compiler flags
        compiler = build_settings.get("compiler", "gcc")
        optimization = build_settings.get("optimization", "O3")
        
        compiler_map = {
            "gcc": "mpicxx",
            "icc": "mpiicc",
            "oneapi": "mpicxx -cxx=icpx"
        }
        
        compiler_cmd = compiler_map.get(compiler, "mpicxx")
        
        opt_flags_map = {
            "O3": "-O3",
            "O2": "-O2",
            "O3xHost": "-O3 -xHost",
            "debug": "-g -O0"
        }
        
        opt_flags = opt_flags_map.get(optimization, "-O3")
        
        # Add OpenMP flag if needed
        omp_threads = parallel_settings.get("omp_threads", 1)
        omp_flag = "-fopenmp" if omp_threads > 1 else ""
        
        # Add tracing flags if needed
        trace_flags = ""
        if tracing:
            if compiler in ["icc", "oneapi"]:
                trace_flags = "-trace"
            else:
                # Assume Score-P for GCC
                trace_flags = "-DSCOREP_USER_ENABLE"
        
        # Create build directory for this specific build
        build_dir = run_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Generate a build script
        build_script_path = build_dir / "build.sh"
        build_script_content = f"""#!/bin/bash
set -e

# Build MCTS executable with {compiler} and {optimization}
{compiler_cmd} {opt_flags} {omp_flag} {trace_flags} \\
    -DMCTS_SIMULATIONS={parallel_settings.get('simulations', 10000)} \\
    -DMCTS_PARALLELIZATION="{build_settings.get('parallelization', 'treeMPI')}" \\
    -DMCTS_OMP_THREADS={omp_threads} \\
    -I{self.src_dir}/mcts \\
    {self.src_dir}/mcts/main.cpp \\
    -o {build_dir}/mcts_scheduler

echo "Build completed: {build_dir}/mcts_scheduler"
"""
        with open(build_script_path, 'w') as f:
            f.write(build_script_content)
        
        # Make script executable
        os.chmod(build_script_path, 0o755)
        
        # Execute build script (placeholder for actual build)
        # In a real implementation, you'd run this script and handle errors
        # For now, we'll just record the path where the executable would be
        executable_path = build_dir / "mcts_scheduler"
        
        # Update the config with executable path
        config_path = run_dir / "run_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["executable_path"] = str(executable_path)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(executable_path)
    def generate_slurm_script(self, 
                             run_id: str, 
                             parallel_settings: Dict[str, Any],
                             time_limit: str = "01:00:00",
                             cores_per_node: int = 40,
                             nodes: Optional[int] = None) -> str:
        """
        Generate a Slurm script for the experiment.
        
        Args:
            run_id: The experiment run ID
            parallel_settings: MPI and OpenMP settings
            time_limit: Job time limit in HH:MM:SS format
            cores_per_node: Number of CPU cores per compute node
            nodes: Explicit number of nodes to request (overrides automatic calculation)
            
        Returns:
            Path to the generated Slurm script
        """
        run_dir = self.experiment_results_dir / run_id
        
        # Load config
        config_path = run_dir / "run_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract settings
        processes = parallel_settings.get("processes", 1)
        
        # Calculate nodes required if not explicitly provided
        if nodes is None:
            nodes = (processes + cores_per_node - 1) // cores_per_node  # Ceiling division
        
        tasks_per_node = min(processes, cores_per_node)
        omp_threads = parallel_settings.get("omp_threads", 1)
        
        executable_path = config.get("executable_path", "")
        problem_instance = config.get("problem_instance", "")
        tracing = config.get("tracing", False)
        
        # Calculate total cores and memory
        total_cores = processes * omp_threads
        mem_per_core = 4  # GB per core
        total_mem = nodes * cores_per_node * mem_per_core  # Total memory in GB
        
        # Generate Slurm script
        slurm_script_path = run_dir / "run.slurm"
        slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=mcts_{run_id}
#SBATCH --output={run_dir}/job_output.log
#SBATCH --error={run_dir}/job_error.log
#SBATCH --nodes={nodes}
#SBATCH --ntasks={processes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task={omp_threads}
#SBATCH --mem={total_mem}G
#SBATCH --time={time_limit}

echo "Starting job for run_id: {run_id}"
echo "Timestamp: $(date)"
echo "Nodes: {nodes}, Processes: {processes}, OMP Threads: {omp_threads}"
echo "Cores per node: {cores_per_node}, Total cores: {total_cores}"

# Load necessary modules
module purge
module load intel/oneapi
module load mpi/intel

# Set environment variables
export OMP_NUM_THREADS={omp_threads}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Set tracing environment if enabled
"""
        
        if tracing:
            slurm_script_content += f"""
# Intel Trace Analyzer settings
export VT_LOGFILE_PREFIX="{run_dir}/trace_files/trace"
export VT_LOGFILE_FORMAT="SINGLESTF"
"""
        
        slurm_script_content += f"""
# Start monitoring script in background
bash {run_dir}/monitor_script.sh $$ &
MONITOR_PID=$!

# Ensure monitoring script is killed when this script exits
trap "kill $MONITOR_PID" EXIT

# Run the application
echo "Running MCTS scheduler with {processes} processes and {omp_threads} threads per process"
mpirun -n {processes} {executable_path} --problem={problem_instance} --output={run_dir}/solution.json

# Post-processing
echo "Job completed at $(date)"
echo "Results written to {run_dir}/solution.json"
"""
        
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script_content)
        
        return str(slurm_script_path)
    
    def submit_job(self, run_id: str) -> Optional[str]:
        """
        Submit a job to the Slurm scheduler.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Slurm job ID if submission successful, None otherwise
        """
        run_dir = self.experiment_results_dir / run_id
        slurm_script_path = run_dir / "run.slurm"
        
        # Check if script exists
        if not slurm_script_path.exists():
            print(f"Error: Slurm script not found at {slurm_script_path}")
            return None
        
        # Submit job
        try:
            result = subprocess.run(
                ["sbatch", str(slurm_script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Extract job ID
            output = result.stdout.strip()
            job_id = output.split()[-1]
            
            # Update config with job ID
            config_path = run_dir / "run_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config["slurm_job_id"] = job_id
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Job submitted with ID: {job_id}")
            return job_id
            
        except subprocess.SubprocessError as e:
            print(f"Error submitting job: {e}")
            return None


def main():
    """Main function to parse arguments and run the experiment manager."""
    parser = argparse.ArgumentParser(description="MCTS Scheduling Optimization Experiment Manager")
    
    # Base directory
    parser.add_argument("--base-dir", type=str, default=os.getcwd(),
                        help="Base directory for experiments")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up a new experiment")
    setup_parser.add_argument("--problem", type=str, required=True,
                             help="Path to problem instance file")
    setup_parser.add_argument("--simulations", type=int, default=10000,
                             help="Number of MCTS simulations")
    setup_parser.add_argument("--exploration", type=float, default=1.414,
                             help="MCTS exploration constant (e.g., for UCT).")
    setup_parser.add_argument("--parallelization", type=str, default="treeMPI",
                             choices=["treeMPI", "rootMPI", "hybrid"],
                             help="Parallelization strategy")
    setup_parser.add_argument("--compiler", type=str, default="gcc",
                             choices=["gcc", "icc", "oneapi"],
                             help="Compiler to use")
    setup_parser.add_argument("--optimization", type=str, default="O3",
                             choices=["O2", "O3", "O3xHost", "debug"],
                             help="Optimization level")
    setup_parser.add_argument("--processes", type=int, default=1,
                             help="Number of MPI processes")    
    setup_parser.add_argument("--omp-threads", type=int, required=True, help="Number of OpenMP threads per process.")
    setup_parser.add_argument("--tracing", action="store_true", help="Enable execution tracing.")
    setup_parser.add_argument("--nodes", type=int, default=1, help="Number of compute nodes to request.")
    setup_parser.add_argument("--cores-per-node", type=int, default=40, help="Number of CPU cores per compute node.") # Added cores-per-node

    # --- Generate Slurm Script Subcommand ---
    slurm_parser = subparsers.add_parser("generate-slurm", help="Generate Slurm script for an experiment")
    slurm_parser.add_argument("--run-id", type=str, required=True,
                             help="Run ID of the experiment")
    slurm_parser.add_argument("--time-limit", type=str, default="01:00:00",
                             help="Job time limit in HH:MM:SS format")
    slurm_parser.add_argument("--nodes", type=int, default=None,
                             help="Number of nodes to request (overrides automatic calculation)")
    slurm_parser.add_argument("--cores-per-node", type=int, default=40,
                             help="Number of CPU cores per compute node")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build executable for an experiment")
    build_parser.add_argument("--run-id", type=str, required=True,
                             help="Run ID of the experiment")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit job for an experiment")
    submit_parser.add_argument("--run-id", type=str, required=True,
                             help="Run ID of the experiment")
    submit_parser.add_argument("--time-limit", type=str, default="01:00:00",
                             help="Job time limit in HH:MM:SS format")
    submit_parser.add_argument("--nodes", type=int, default=None,
                             help="Number of nodes to request (overrides config)") # Added nodes
    submit_parser.add_argument("--cores-per-node", type=int, default=40,
                             help="Number of CPU cores per node (overrides config)") # Added cores-per-node
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize experiment manager
    manager = ExperimentManager(args.base_dir)
    
    # Execute command
    if args.command == "setup":
        # Prepare settings dictionaries
        mcts_settings = {
            "simulations": args.simulations,
            "exploration": args.exploration
        }
        
        build_settings = {
            "parallelization": args.parallelization,
            "compiler": args.compiler,
            "optimization": args.optimization
        }
        
        parallel_settings = {
            "processes": args.processes,
            "omp_threads": args.omp_threads
        }
          # Set up experiment
        run_id = manager.setup_experiment(
            problem_instance=args.problem,
            mcts_settings=mcts_settings,
            build_settings=build_settings,
            parallel_settings=parallel_settings,
            tracing=args.tracing,
            nodes=args.nodes,
            cores_per_node=args.cores_per_node
        )
        
        print(f"Experiment set up with run_id: {run_id}")
        print(f"Directory: {manager.experiment_results_dir / run_id}")
        
    elif args.command == "build":
        # Load config for the run
        run_dir = manager.experiment_results_dir / args.run_id
        config_path = run_dir / "run_config.json"
        
        if not config_path.exists():
            print(f"Error: Config file not found for run_id: {args.run_id}")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Build executable
        executable_path = manager.build_executable(
            run_id=args.run_id,
            build_settings=config.get("build_settings", {}),
            parallel_settings=config.get("parallel_settings", {}),
            tracing=config.get("tracing", False)
        )
        
        print(f"Executable built at: {executable_path}")
        
    elif args.command == "submit":
        # Generate Slurm script
        run_dir = manager.experiment_results_dir / args.run_id
        config_path = run_dir / "run_config.json"
        
        if not config_path.exists():
            print(f"Error: Config file not found for run_id: {args.run_id}")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use command line argument for nodes if provided, otherwise use config
        nodes = args.nodes if args.nodes is not None else config.get("nodes")
        cores_per_node = args.cores_per_node if args.cores_per_node != 40 else config.get("cores_per_node", 40)
        
        # Generate Slurm script
        slurm_script_path = manager.generate_slurm_script(
            run_id=args.run_id,
            parallel_settings=config.get("parallel_settings", {}),
            time_limit=args.time_limit,
            cores_per_node=cores_per_node,
            nodes=nodes
        )
        
        print(f"Slurm script generated at: {slurm_script_path}")
        
        # Submit job
        job_id = manager.submit_job(args.run_id)
        if job_id:
            print(f"Job submitted with ID: {job_id}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
