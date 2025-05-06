#!/bin/bash
# MCTS Scheduling Experiments Bash Script
# This script provides example commands for running different experiment configurations

echo "MCTS Scheduling Experiments"
echo "============================"

# Create the plots directory if it doesn't exist
mkdir -p plots

show_menu() {
    echo ""
    echo "Select an experiment to run:"
    echo "1. Build all configurations"
    echo "2. Quick test (single problem, few cores)"
    echo "3. Full scaling test (all problems, all core counts)"
    echo "4. Compiler comparison (GCC vs OneAPI)"
    echo "5. Parallelization strategy comparison"
    echo "6. Parse results and generate plots"
    echo "7. Run all steps (build, run, wait, parse, plot)"
    echo "8. Multi-node scaling test"
    echo "9. Exit"
    echo ""
}

build_all() {
    echo "Building all configurations..."
    python3 automate_experiments.py --build --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --simulations 10000 50000 --parallelization treeMPI rootMPI --omp-threads 1 2 4
}

quick_test() {
    echo "Running quick test..."
    python3 automate_experiments.py --build --run --wait --parse --plot --compilers gcc --problems problem_instances/sample_problem01.txt --processes 1 2 4 --omp-threads 1 2 --simulations 10000 --parallelization treeMPI --time-limit 00:10:00
}

full_scaling() {
    echo "Running full scaling test..."
    python3 automate_experiments.py --build --run --parse --plot --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 32 --omp-threads 1 2 4 --simulations 50000 --parallelization treeMPI --nodes auto
}

compiler_test() {
    echo "Running compiler comparison test..."
    python3 automate_experiments.py --build --run --parse --plot --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 --omp-threads 1 --simulations 10000 --parallelization treeMPI
}

parallelization_test() {
    echo "Running parallelization strategy comparison..."
    python3 automate_experiments.py --build --run --parse --plot --compilers gcc --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 --omp-threads 1 4 --simulations 10000 --parallelization treeMPI rootMPI
}

parse_plot() {
    echo "Parsing results and generating plots..."
    python3 automate_experiments.py --parse --plot
}

run_all() {
    echo "Running all experiment steps..."
    python3 automate_experiments.py --all --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 32 64 --omp-threads 1 2 4 --simulations 10000 --parallelization treeMPI rootMPI --nodes auto
}

multi_node_scaling() {
    echo "Running multi-node scaling test..."
    # Explicit node assignments for different process counts
    python3 automate_experiments.py --build --run --parse --plot \
        --compilers gcc \
        --problems problem_instances/sample_problem01.txt \
        --processes 16 32 64 128 256 \
        --omp-threads 1 2 \
        --simulations 50000 \
        --parallelization treeMPI \
        --nodes 1 1 2 4 8 \
        --time-limit 04:00:00
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter your choice (1-9): " choice
    
    case $choice in
        1) build_all ;;
        2) quick_test ;;
        3) full_scaling ;;
        4) compiler_test ;;
        5) parallelization_test ;;
        6) parse_plot ;;
        7) run_all ;;
        8) multi_node_scaling ;;
        9) echo "Exiting..."; exit 0 ;;
        *) echo "Invalid choice. Please try again." ;;
    esac
    
    echo ""
    echo "Operation completed."
    read -p "Press Enter to continue..."
done
