#!/bin/bash
# Scaling test script wrapper

show_help() {
    echo "MCTS Scaling Test Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --problem FILE      Problem instance file"
    echo "  -n, --nodes N [N...]    Node counts for each process"
    echo "  --processes N [N...]    Process counts to test"
    echo "  -t, --threads N [N...]  OpenMP thread counts (default: 1 2 4)"
    echo "  -s, --sims N            Simulations count (default: 50000)"
    echo "  -c, --compiler NAME     Compiler: gcc or oneapi (default: gcc)"
    echo "  --para NAME             Parallelization: treeMPI or rootMPI (default: treeMPI)"
    echo "  --time HH:MM:SS         Time limit (default: 04:00:00)"
    echo "  --cpn N                 Cores per node (default: 40)"
    echo "  --no-wait               Don't wait for jobs to complete"
    echo "  -o, --output DIR        Output directory (default: scaling_analysis)"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --problem problem_instances/sample_problem01.txt --processes 16 32 64 128 --nodes 1 1 2 4"
    echo ""
}

# Default values
PROBLEM=""
PROCESSES=()
NODES=()
THREADS=(1 2 4)
SIMS=50000
COMPILER="gcc"
PARA="treeMPI"
TIME="04:00:00"
CPN=40
NO_WAIT=""
OUTPUT="scaling_analysis"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--problem)
            PROBLEM="$2"
            shift 2
            ;;
        --processes)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                PROCESSES+=("$1")
                shift
            done
            ;;
        -n|--nodes)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                NODES+=("$1")
                shift
            done
            ;;
        -t|--threads)
            THREADS=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                THREADS+=("$1")
                shift
            done
            ;;
        -s|--sims)
            SIMS="$2"
            shift 2
            ;;
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        --para)
            PARA="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --cpn)
            CPN="$2"
            shift 2
            ;;
        --no-wait)
            NO_WAIT="--no-wait"
            shift
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$PROBLEM" ]; then
    echo "Error: Problem instance file is required"
    show_help
    exit 1
fi

if [ ${#PROCESSES[@]} -eq 0 ]; then
    echo "Error: Process counts are required"
    show_help
    exit 1
fi

# Build process and node arguments
PROC_ARG=""
for p in "${PROCESSES[@]}"; do
    PROC_ARG+="$p "
done

NODE_ARG=""
if [ ${#NODES[@]} -gt 0 ]; then
    for n in "${NODES[@]}"; do
        NODE_ARG+="$n "
    done
    NODE_ARG="--nodes $NODE_ARG"
fi

# Build thread argument
THREAD_ARG=""
for t in "${THREADS[@]}"; do
    THREAD_ARG+="$t "
done

# Run the scaling test
echo "Running MCTS Scaling Test:"
echo "  Problem: $PROBLEM"
echo "  Processes: $PROC_ARG"
if [ -n "$NODE_ARG" ]; then
    echo "  Nodes: ${NODES[*]}"
fi
echo "  Threads: ${THREADS[*]}"
echo "  Simulations: $SIMS"
echo "  Compiler: $COMPILER"
echo "  Parallelization: $PARA"
echo "  Output: $OUTPUT"
echo ""

python scaling_test.py --problem "$PROBLEM" \
    --processes $PROC_ARG \
    $NODE_ARG \
    --omp-threads $THREAD_ARG \
    --simulations $SIMS \
    --compiler $COMPILER \
    --parallelization $PARA \
    --time-limit $TIME \
    --cores-per-node $CPN \
    --output-dir $OUTPUT \
    $NO_WAIT
