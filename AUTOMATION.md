# Experiment Automation Tools

This directory contains tools for automating MCTS scheduling optimization experiments, including building, job submission, and result analysis.

## Automation Scripts

- **automate_experiments.py**: Main Python script for automating experiments
- **run_experiments.bat**: Windows batch script with pre-configured experiment templates
- **run_experiments.sh**: Linux/Unix shell script with pre-configured experiment templates

## Automation Features

The automation tools provide the following functionality:

1. **Building MCTS code** with different compilers, optimization levels, and parallelization strategies
2. **Submitting batch jobs** to Slurm scheduler with different parameter combinations
3. **Monitoring job progress** and waiting for completion
4. **Parsing results** from experiment logs and output files
5. **Analyzing and visualizing results** through various plots and metrics

## Usage Examples

### Basic Commands

```bash
# Build MCTS with GCC compiler
python automate_experiments.py --build --compilers gcc --problems problem_instances/sample_problem01.txt

# Run experiments with 1, 2, 4, and 8 processes
python automate_experiments.py --run --compilers gcc --problems problem_instances/sample_problem01.txt --processes 1 2 4 8

# Parse results and generate plots
python automate_experiments.py --parse --plot

# Run all steps (build, run, wait, parse, plot)
python automate_experiments.py --all --compilers gcc --problems problem_instances/sample_problem01.txt --processes 1 2 4
```

### Advanced Examples

```bash
# Compare two parallelization strategies
python automate_experiments.py --build --run --parse --plot \
    --compilers gcc \
    --problems problem_instances/sample_problem01.txt \
    --processes 1 2 4 8 16 \
    --omp-threads 1 4 \
    --simulations 10000 \
    --parallelization treeMPI rootMPI

# Full scaling experiment with custom job settings
python automate_experiments.py --build --run --parse --plot \
    --compilers gcc oneapi \
    --problems problem_instances/sample_problem01.txt \
    --processes 1 2 4 8 16 32 64 \
    --omp-threads 1 2 4 8 \
    --simulations 50000 \
    --time-limit 04:00:00 \
    --partition compute
```

### Multi-Node Scaling Examples

```bash
# Multi-node scaling test with automatic node allocation
# The script calculates nodes needed based on processes and cores-per-node
python automate_experiments.py --build --run --parse --plot \
    --compilers gcc \
    --problems problem_instances/sample_problem01.txt \
    --processes 16 32 64 128 256 \
    --omp-threads 1 2 \
    --simulations 50000 \
    --parallelization treeMPI \
    --nodes auto \
    --cores-per-node 48  # Using 48-core node configuration

# Multi-node scaling test with explicit node allocation
# The number of items in --nodes must match the number of items in --processes
python automate_experiments.py --build --run --parse --plot \
    --compilers gcc \
    --problems problem_instances/sample_problem01.txt \
    --processes 48 96 192 384 768 \ # Process counts aligned with 48-core nodes
    --omp-threads 1 \
    --simulations 50000 \
    --parallelization treeMPI \
    --nodes 1 2 4 8 16 \ # Node counts matched to process counts
    --cores-per-node 48
```

The `--nodes` parameter can be used in two ways:
1. Setting it to `auto` (or omitting it, as 'auto' is the default) will automatically calculate the appropriate number of nodes based on each process count in `--processes` and the value of `--cores-per-node`. The calculation ensures optimal resource utilization based on the available 48 cores per node.

2. Providing a list of specific node counts. **The number of node counts provided must exactly match the number of process counts specified with `--processes`**. 

Examples of automatic node calculation with 48 cores per node:
- 48 processes × 1 thread = 48 cores (1 node)
- 96 processes × 1 thread = 96 cores (2 nodes)
- 48 processes × 2 threads = 96 cores (2 nodes)
- 192 processes × 1 thread = 192 cores (4 nodes)

Note: When using OpenMP threads (--omp-threads > 1), the total core requirement is calculated as: processes × threads. Make sure this total doesn't exceed the available cores (nodes × cores-per-node).

## Command-Line Options

The `automate_experiments.py` script supports the following options:

### General Flow Options
- `--build`: Build executable(s) for specified configurations
- `--run`: Generate and submit Slurm jobs
- `--wait`: Wait for submitted jobs to complete
- `--parse`: Parse results from output logs
- `--plot`: Generate plots from parsed results
- `--all`: Run all steps (build, run, wait, parse, plot)

### Configuration Options
- `--compilers`: Compilers to use (gcc, oneapi)
- `--problems`: Problem instance files to test
- `--processes`: MPI process counts to test
- `--omp-threads`: OpenMP threads per process to test
- `--simulations`: MCTS simulation counts to test
- `--parallelization`: Parallelization strategy (treeMPI, rootMPI)
- `--nodes`: Number of nodes to use for each process count. Default is ['auto']. Provide 'auto' for automatic calculation based on `--cores-per-node`, or provide a list of specific integer counts matching the number of entries in `--processes`.
- `--cores-per-node`: Number of CPU cores per compute node (default: 40). Used for 'auto' node calculation.
- `--tracing`: Enable execution tracing

### Job Control Options
- `--time-limit`: Job time limit in HH:MM:SS format
- `--partition`: Slurm partition to submit jobs to
- `--max-wait`: Maximum time to wait for jobs (seconds)

## Generated Plots

The automation tools generate the following plots:

1. **Execution Time vs. Cores**: Shows how execution time scales with number of cores
2. **Speedup vs. Cores**: Shows parallel speedup relative to the smallest core count
3. **Parallel Efficiency vs. Cores**: Shows how efficiently additional cores are utilized
4. **Execution Time vs. Nodes**: Shows how execution time scales with number of compute nodes
5. **Node Scaling Efficiency vs. Nodes**: Shows how efficiently additional nodes are utilized
6. **Makespan vs. Cores**: Shows solution quality (makespan) as a function of core count
7. **Parallelization Strategy Comparison**: Compares different parallelization strategies

## Results CSV Format

The parsed results are saved in `mcts_experiment_results.csv` with the following columns:

- `run_id`: Unique identifier for the experiment run
- `problem`: Problem instance name
- `processes`: Number of MPI processes
- `omp_threads`: Number of OpenMP threads per process
- `total_cores`: Total number of cores (processes × threads)
- `simulations`: Number of MCTS simulations
- `parallelization`: Parallelization strategy
- `compiler`: Compiler used
- `nodes`: Number of nodes requested for the run
- `makespan`: Resulting schedule makespan
- `execution_time`: Total execution time in seconds
- `mcts_nodes`: Number of MCTS nodes explored

## Environment Requirements

- Python 3.6+
- Pandas and Matplotlib for analysis and visualization
- Access to a Slurm-based compute cluster
- MPI and OpenMP development libraries
- GCC and/or Intel OneAPI compilers

## Advanced Scaling Test Tools

For more detailed and customizable multi-node scaling experiments, the project includes specialized scaling test tools:

### Scaling Test Script

The `scaling_test.py` script provides comprehensive multi-node scaling analysis:

```bash
python scaling_test.py --problem problem_instances/sample_problem01.txt \
    --processes 16 32 64 128 256 \
    --nodes 1 1 2 4 8 \
    --omp-threads 1 2 4 \
    --simulations 50000 \
    --compiler gcc \
    --output-dir scaling_results
```

### Scaling Test Wrappers

For convenience, wrapper scripts are provided for both Linux and Windows:

**Linux/Unix:**
```bash
./run_scaling_test.sh --problem problem_instances/sample_problem01.txt \
    --processes 16 32 64 128 256 \
    --nodes 1 1 2 4 8 \
    --threads 1 2
```

**Windows:**
```batch
run_scaling_test.bat --problem problem_instances/sample_problem01.txt ^
    --processes 16 32 64 128 256 ^
    --nodes 1 1 2 4 8 ^
    --threads 1 2
```

### Scaling Analysis

The scaling test tools generate advanced performance analysis plots:

1. **Process Scaling per Node Configuration**: Shows how execution time varies with process count for different node and thread configurations
2. **Node Scaling Efficiency**: Shows how efficiently the application scales across multiple nodes
3. **Combined MPI+OpenMP Performance**: Analyzes the interaction between MPI processes and OpenMP threads across nodes

All results are saved to CSV for further custom analysis.
