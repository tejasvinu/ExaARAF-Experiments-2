@echo off
REM Scaling test script wrapper for Windows

setlocal enabledelayedexpansion

REM Default values
set "PROBLEM="
set "PROCESSES="
set "NODES="
set "THREADS=1 2 4"
set "SIMS=50000"
set "COMPILER=gcc"
set "PARA=treeMPI"
set "TIME=04:00:00"
set "CPN=40"
set "NO_WAIT="
set "OUTPUT=scaling_analysis"

if "%~1"=="" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help

:parse_args
if "%~1"=="" goto :check_args

if "%~1"=="--problem" (
    set "PROBLEM=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="-p" (
    set "PROBLEM=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="--processes" (
    set "PROCESSES="
    shift
    :parse_processes
    if "%~1"=="" goto :check_args
    if "%~1:~0,1%"=="-" goto :parse_args
    set "PROCESSES=!PROCESSES! %~1"
    shift
    goto :parse_processes
)

if "%~1"=="--nodes" (
    set "NODES="
    shift
    :parse_nodes
    if "%~1"=="" goto :check_args
    if "%~1:~0,1%"=="-" goto :parse_args
    set "NODES=!NODES! %~1"
    shift
    goto :parse_nodes
)

if "%~1"=="--threads" (
    set "THREADS="
    shift
    :parse_threads
    if "%~1"=="" goto :check_args
    if "%~1:~0,1%"=="-" goto :parse_args
    set "THREADS=!THREADS! %~1"
    shift
    goto :parse_threads
)

if "%~1"=="-t" (
    set "THREADS="
    shift
    :parse_threads2
    if "%~1"=="" goto :check_args
    if "%~1:~0,1%"=="-" goto :parse_args
    set "THREADS=!THREADS! %~1"
    shift
    goto :parse_threads2
)

if "%~1"=="--sims" (
    set "SIMS=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="-s" (
    set "SIMS=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="--compiler" (
    set "COMPILER=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="-c" (
    set "COMPILER=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="--para" (
    set "PARA=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="--time" (
    set "TIME=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="--cpn" (
    set "CPN=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="--no-wait" (
    set "NO_WAIT=--no-wait"
    shift
    goto :parse_args
)

if "%~1"=="--output" (
    set "OUTPUT=%~2"
    shift
    shift
    goto :parse_args
)

if "%~1"=="-o" (
    set "OUTPUT=%~2"
    shift
    shift
    goto :parse_args
)

echo Unknown option: %~1
goto :show_help

:check_args
if "%PROBLEM%"=="" (
    echo Error: Problem instance file is required
    goto :show_help
)

if "%PROCESSES%"=="" (
    echo Error: Process counts are required
    goto :show_help
)

REM Build node argument
set "NODE_ARG="
if "%NODES%" NEQ "" (
    set "NODE_ARG=--nodes%NODES%"
)

REM Run the scaling test
echo Running MCTS Scaling Test:
echo   Problem: %PROBLEM%
echo   Processes:%PROCESSES%
if "%NODE_ARG%" NEQ "" (
    echo   Nodes:%NODES%
)
echo   Threads:%THREADS%
echo   Simulations: %SIMS%
echo   Compiler: %COMPILER%
echo   Parallelization: %PARA%
echo   Output: %OUTPUT%
echo.

python scaling_test.py --problem "%PROBLEM%" ^
    --processes%PROCESSES% ^
    %NODE_ARG% ^
    --omp-threads%THREADS% ^
    --simulations %SIMS% ^
    --compiler %COMPILER% ^
    --parallelization %PARA% ^
    --time-limit %TIME% ^
    --cores-per-node %CPN% ^
    --output-dir %OUTPUT% ^
    %NO_WAIT%

goto :eof

:show_help
echo MCTS Scaling Test Script
echo.
echo Usage: %0 [options]
echo.
echo Options:
echo   -p, --problem FILE      Problem instance file
echo   -n, --nodes N [N...]    Node counts for each process
echo   --processes N [N...]    Process counts to test
echo   -t, --threads N [N...]  OpenMP thread counts (default: 1 2 4)
echo   -s, --sims N            Simulations count (default: 50000)
echo   -c, --compiler NAME     Compiler: gcc or oneapi (default: gcc)
echo   --para NAME             Parallelization: treeMPI or rootMPI (default: treeMPI)
echo   --time HH:MM:SS         Time limit (default: 04:00:00)
echo   --cpn N                 Cores per node (default: 40)
echo   --no-wait               Don't wait for jobs to complete
echo   -o, --output DIR        Output directory (default: scaling_analysis)
echo   -h, --help              Show this help
echo.
echo Example:
echo   %0 --problem problem_instances/sample_problem01.txt --processes 16 32 64 128 --nodes 1 1 2 4
echo.

:eof
endlocal
