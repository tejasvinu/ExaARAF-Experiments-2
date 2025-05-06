@echo off
REM MCTS Scheduling Experiments Batch Script
REM This script provides example commands for running different experiment configurations

echo MCTS Scheduling Experiments
echo ============================

REM Create the plots directory if it doesn't exist
if not exist plots mkdir plots

:menu
echo.
echo Select an experiment to run:
echo 1. Build all configurations
echo 2. Quick test (single problem, few cores)
echo 3. Full scaling test (all problems, all core counts)
echo 4. Compiler comparison (GCC vs OneAPI)
echo 5. Parallelization strategy comparison
echo 6. Parse results and generate plots
echo 7. Run all steps (build, run, wait, parse, plot)
echo 8. Multi-node scaling test
echo 9. Exit

set /p choice=Enter your choice (1-9): 

if "%choice%"=="1" goto build_all
if "%choice%"=="2" goto quick_test
if "%choice%"=="3" goto full_scaling
if "%choice%"=="4" goto compiler_test
if "%choice%"=="5" goto parallelization_test
if "%choice%"=="6" goto parse_plot
if "%choice%"=="7" goto run_all
if "%choice%"=="8" goto multi_node_scaling
if "%choice%"=="9" goto end

echo Invalid choice. Please try again.
goto menu

:build_all
echo.
echo Building all configurations...
python automate_experiments.py --build --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --simulations 10000 50000 --parallelization treeMPI rootMPI --omp-threads 1 2 4
goto end

:quick_test
echo.
echo Running quick test...
python automate_experiments.py --build --run --wait --parse --plot --compilers gcc --problems problem_instances/sample_problem01.txt --processes 1 2 4 --omp-threads 1 2 --simulations 10000 --parallelization treeMPI --time-limit 00:10:00
goto end

:full_scaling
echo.
echo Running full scaling test...
python automate_experiments.py --build --run --parse --plot --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 32 --omp-threads 1 2 4 --simulations 50000 --parallelization treeMPI --nodes auto
goto end

:compiler_test
echo.
echo Running compiler comparison test...
python automate_experiments.py --build --run --parse --plot --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 --omp-threads 1 --simulations 10000 --parallelization treeMPI
goto end

:parallelization_test
echo.
echo Running parallelization strategy comparison...
python automate_experiments.py --build --run --parse --plot --compilers gcc --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 --omp-threads 1 4 --simulations 10000 --parallelization treeMPI rootMPI
goto end

:parse_plot
echo.
echo Parsing results and generating plots...
python automate_experiments.py --parse --plot
goto end

:run_all
echo.
echo Running all experiment steps...
python automate_experiments.py --all --compilers gcc oneapi --problems problem_instances/sample_problem01.txt --processes 1 2 4 8 16 32 64 --omp-threads 1 2 4 --simulations 10000 --parallelization treeMPI rootMPI --nodes auto
goto end

:multi_node_scaling
echo.
echo Running multi-node scaling test...
python automate_experiments.py --build --run --parse --plot --compilers gcc --problems problem_instances/sample_problem01.txt --processes 16 32 64 128 256 --omp-threads 1 2 --simulations 50000 --parallelization treeMPI --nodes 1 1 2 4 8 --time-limit 04:00:00
goto end

:end
echo.
echo Operation completed.
pause
