# MCTS for Scheduling Optimization with Comprehensive Logging

This repository contains a framework for experimenting with Monte Carlo Tree Search (MCTS) for scheduling optimization problems, with comprehensive logging capabilities and multi-node scaling support.

## Project Structure

```
ExaARAF-Experiments-2/
├── src/
│   └── mcts/                  # MCTS implementation
│       └── main.cpp           # Main MCTS scheduler implementation
│
├── scripts/
│   ├── experiment_manager.py  # Python script for managing experiments
│   ├── monitor_script.sh      # Shell script for monitoring system resources
│   └── results_parser.py      # Python script for parsing and aggregating results
│
├── problem_instances/         # Directory containing problem instance files
│   └── sample_problem01.txt   # Sample problem instance
│
├── experiment_results/        # Directory for storing experiment results
│   └── [run_id]/              # Directory for each experiment run
│       ├── run_config.json    # Configuration for the run
│       ├── run.slurm          # Slurm script for the run
│       ├── job_output.log     # Standard output from the job
│       ├── job_error.log      # Standard error from the job
│       ├── solution.json      # Solution output from MCTS
│       ├── system_metrics.*.log # System resource usage logs
│       ├── build/             # Build directory for the executable
│       └── trace_files/       # Directory for trace files (if tracing enabled)
│
├── automate_experiments.py    # Script for automating experiment workflows
├── run_experiments.bat        # Windows batch script for running experiments
├── run_experiments.sh         # Linux/Unix shell script for running experiments
├── AUTOMATION.md              # Documentation for automation tools
├── example_workflow.py        # Example workflow script
├── README.md                  # This file
│
└── utils/                     # Utility scripts and tools
```

## Problem Instance Format

Problem instances are specified in text files with the following format:

```
N M
T_1 D_1 [Dep_1,1 Dep_1,2 ...]
T_2 D_2 [Dep_2,1 Dep_2,2 ...]
...
T_N D_N [Dep_N,1 Dep_N,2 ...]
```

Where:
- `N` is the number of tasks
- `M` is the number of machines
- `T_i` is the duration of task i
- `D_i` is the number of dependencies for task i
- `Dep_i,j` is the j-th dependency for task i

## Usage

### Setting Up an Experiment

```bash
python scripts/experiment_manager.py setup \\
    --problem problem_instances/sample_problem01.txt \\
    --simulations 50000 \\
    --exploration 1.414 \\
    --parallelization treeMPI \\
    --compiler oneapi \\
    --optimization O3xHost \\
    --processes 32 \\
    --omp-threads 4 \\
    --nodes 1 \\
    --cores-per-node 40 \\ # Added cores-per-node
    --tracing
```

This will:
1. Generate a unique run ID
2. Create a directory under `experiment_results/`
3. Create a configuration file with all parameters
4. Prepare the monitoring script

### Building the Executable

```bash
python scripts/experiment_manager.py build \
    --run-id [run_id]
```

This will:
1. Read the configuration for the run
2. Generate a build script with appropriate compiler flags
3. Build the MCTS scheduler executable

### Submitting the Job

```bash
python scripts/experiment_manager.py submit \\
    --run-id [run_id] \\
    --time-limit 04:00:00 \\ # Optional: Override time limit
    # --nodes 2 \\             # Optional: Override node count from config
    # --cores-per-node 32 \\ # Optional: Override cores per node from config
```

This will:
1. Generate a Slurm script with appropriate settings
2. Submit the job to the Slurm scheduler
3. Update the configuration with the job ID

### Parsing Results

```bash
python scripts/results_parser.py \
    --output experiment_results.csv
```

This will:
1. Iterate through all experiment runs
2. Parse and aggregate results from each run
3. Generate a CSV file with all results

## Logging Strategy

For each experiment run, the following information is collected:

1. **Configuration**:
   - Problem instance details
   - MCTS settings (simulations, exploration parameter)
   - Build settings (parallelization, compiler, optimization)
   - Parallel settings (MPI processes, OpenMP threads)
   - Git hash of the code
   - Slurm job ID

2. **Results**:
   - Makespan of the final solution
   - Execution time
   - Number of MCTS nodes explored
   - Number of simulations performed

3. **System Metrics**:
   - CPU usage (mean and max)
   - Memory usage (max RSS and VMS)
   - I/O statistics (total read/write)

4. **Tracing Information** (if enabled):
   - Trace files for detailed parallel execution analysis

## Example Workflow

1. Create a problem instance in `problem_instances/`
2. Set up an experiment using the experiment manager
3. Build the executable
4. Submit the job
5. Parse and analyze the results

## Dependencies

- C++ compiler with MPI and OpenMP support
- Python 3.6+
- MPI implementation (Intel MPI, OpenMPI, etc.)
- Slurm workload manager
- Optional: Intel Trace Analyzer or Score-P for tracing

## Experiment Automation

For automated experiment workflows including scaling tests and performance analysis, we provide several automation tools:

- **automate_experiments.py**: A comprehensive script for building, submitting, and analyzing multiple MCTS experiments.
- **run_experiments.bat/sh**: Helper scripts with pre-configured experiment templates.

See [AUTOMATION.md](AUTOMATION.md) for detailed documentation on the automation tools.

Example of running a scaling test with the automation tool:

```bash
python automate_experiments.py --all \\
    --compilers gcc \\
    --problems problem_instances/sample_problem01.txt \\
    --processes 1 2 4 8 16 \\
    --omp-threads 1 2 4 \\
    --simulations 10000 \\
    --parallelization treeMPI \\
    --nodes auto # Example using automatic node calculation
```

This will automatically:
1. Build the MCTS executable with appropriate configuration
2. Submit jobs for each parameter combination
3. Wait for jobs to complete
4. Parse and aggregate results
5. Generate performance analysis plots

## Multi-Node Scaling

This project supports scaling experiments across multiple compute nodes to evaluate the performance of MCTS scheduling algorithms in distributed environments. The automation tools provide the following multi-node capabilities:

- Automatic or manual node allocation for different process counts
- Comprehensive performance analysis across node configurations
- Visualization of node scaling efficiency
- Support for hybrid MPI+OpenMP parallelization across nodes

To run multi-node scaling experiments, use the `--nodes` parameter with the automation scripts:

```bash
# Example of multi-node scaling experiment with explicit node allocation
# Ensure the number of node counts matches the number of process counts
python automate_experiments.py --build --run --parse --plot \\
    --compilers gcc \\
    --processes 16 32 64 128 256 \\
    --omp-threads 2 \\
    --simulations 50000 \\
    --parallelization treeMPI \\
    --nodes 1 1 2 4 8 # 5 process counts, 5 node counts

# Example using automatic node calculation based on cores_per_node
python automate_experiments.py --build --run --parse --plot \\
    --compilers gcc \\
    --processes 16 32 64 128 256 \\
    --omp-threads 2 \\
    --simulations 50000 \\
    --parallelization treeMPI \\
    --nodes auto \\
    --cores-per-node 40

# Using the helper script for multi-node tests
./run_experiments.sh  # Then select option 8 for multi-node scaling test
```

For more details on multi-node scaling options, see the [Automation Documentation](AUTOMATION.md)

## Advanced Scaling Tests

For more comprehensive multi-node scaling experiments, the `scaling_test.py` script provides detailed analysis and visualization:

```bash
# Run a multi-node scaling test with explicit process and node counts
python scaling_test.py --problem problem_instances/sample_problem01.txt \
    --processes 16 32 64 128 256 \
    --nodes 1 1 2 4 8 \
    --omp-threads 1 2 4 \
    --simulations 50000 \
    --compiler gcc \
    --time-limit 04:00:00

# Run a test with automatic node allocation
python scaling_test.py --problem problem_instances/sample_problem01.txt \
    --processes 16 32 64 128 256 \
    --omp-threads 1 2 \
    --cores-per-node 40 \
    --output-dir scaling_results
```

The scaling test script will:
1. Automatically set up and run experiments across multiple node configurations
2. Wait for all jobs to complete (or use `--no-wait` to submit and exit)
3. Generate comprehensive scaling analysis plots in the specified output directory
4. Save detailed performance metrics to CSV for further analysis
