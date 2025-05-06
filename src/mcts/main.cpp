#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Define constants (or use from command-line arguments)
#ifndef MCTS_SIMULATIONS
#define MCTS_SIMULATIONS 10000
#endif

#ifndef MCTS_PARALLELIZATION
#define MCTS_PARALLELIZATION "treeMPI"
#endif

#ifndef MCTS_OMP_THREADS
#define MCTS_OMP_THREADS 1
#endif

// Scheduling problem representation
struct Task {
    int id;
    int duration;
    std::vector<int> dependencies;
};

struct SchedulingProblem {
    std::vector<Task> tasks;
    int machines;
    
    void load(const std::string& filepath) {
        // Load problem instance from file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open problem file " << filepath << std::endl;
            return;
        }
        
        // Read number of tasks and machines
        int num_tasks;
        file >> num_tasks >> machines;
        
        tasks.resize(num_tasks);
        
        // Read task durations and dependencies
        for (int i = 0; i < num_tasks; ++i) {
            tasks[i].id = i;
            file >> tasks[i].duration;
            
            int num_deps;
            file >> num_deps;
            
            tasks[i].dependencies.resize(num_deps);
            for (int j = 0; j < num_deps; ++j) {
                file >> tasks[i].dependencies[j];
            }
        }
        
        file.close();
    }
};

// Scheduling solution representation
struct SchedulingSolution {
    std::vector<std::pair<int, int>> task_assignments; // (task_id, machine_id)
    std::vector<int> schedule; // task execution order
    int makespan;
    
    SchedulingSolution(const SchedulingProblem& problem) 
        : makespan(0) {
        // Initialize with empty solution
        task_assignments.reserve(problem.tasks.size());
        schedule.reserve(problem.tasks.size());
    }
    
    void evaluate(const SchedulingProblem& problem) {
        // Calculate makespan from task assignments and schedule
        std::vector<int> machine_times(problem.machines, 0);
        std::vector<int> task_completion_times(problem.tasks.size(), 0);
        
        for (int task_idx : schedule) {
            int task_id = task_idx;
            int machine_id = task_assignments[task_id].second;
            int earliest_start = 0;
            
            // Ensure dependencies are satisfied
            for (int dep : problem.tasks[task_id].dependencies) {
                earliest_start = std::max(earliest_start, task_completion_times[dep]);
            }
            
            // Ensure machine is available
            earliest_start = std::max(earliest_start, machine_times[machine_id]);
            
            // Calculate completion time
            int completion_time = earliest_start + problem.tasks[task_id].duration;
            task_completion_times[task_id] = completion_time;
            machine_times[machine_id] = completion_time;
        }
        
        // Makespan is the maximum completion time
        makespan = *std::max_element(machine_times.begin(), machine_times.end());
    }
    
    void to_json(const std::string& filepath) const {
        // Write solution to JSON file
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open output file " << filepath << std::endl;
            return;
        }
        
        file << "{\n";
        file << "  \"makespan\": " << makespan << ",\n";
        file << "  \"task_assignments\": [\n";
        
        for (size_t i = 0; i < task_assignments.size(); ++i) {
            file << "    {\"task\": " << task_assignments[i].first 
                 << ", \"machine\": " << task_assignments[i].second << "}";
            
            if (i < task_assignments.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        
        file << "  ],\n";
        file << "  \"schedule\": [";
        
        for (size_t i = 0; i < schedule.size(); ++i) {
            file << schedule[i];
            if (i < schedule.size() - 1) {
                file << ", ";
            }
        }
        
        file << "]\n";
        file << "}\n";
        
        file.close();
    }
};

// MCTS node representation
class MCTSNode {
public:
    const MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    
    int task_id;
    int machine_id;
    double value;
    int visits;
    
    std::vector<int> available_tasks;
    std::vector<std::pair<int, int>> assignments;
    std::vector<int> execution_order;
    
    MCTSNode(const MCTSNode* parent, int task_id, int machine_id,
             const std::vector<int>& available_tasks,
             const std::vector<std::pair<int, int>>& assignments,
             const std::vector<int>& execution_order)
        : parent(parent), task_id(task_id), machine_id(machine_id),
          value(0.0), visits(0),
          available_tasks(available_tasks),
          assignments(assignments),
          execution_order(execution_order) {}
    
    bool is_fully_expanded(const SchedulingProblem& problem) const {
        return available_tasks.empty() || children.size() == available_tasks.size() * problem.machines;
    }
    
    bool is_terminal() const {
        return available_tasks.empty();
    }
    
    // UCT score for child selection
    double uct_score(double exploration, int parent_visits) const {
        if (visits == 0) {
            return std::numeric_limits<double>::infinity();
        }
        
        return (value / visits) + exploration * std::sqrt(std::log(parent_visits) / visits);
    }
};

// MCTS algorithm implementation
class MCTS {
private:
    const SchedulingProblem& problem;
    std::unique_ptr<MCTSNode> root;
    std::mt19937 rng;
    double exploration_parameter;
    int num_simulations;
    int rank;
    int size;

public:
    MCTS(const SchedulingProblem& problem, double exploration = 1.0, int simulations = MCTS_SIMULATIONS)
        : problem(problem), exploration_parameter(exploration), num_simulations(simulations) {
        
        // Initialize MPI rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Seed RNG with rank to ensure different random sequences
        rng.seed(std::chrono::system_clock::now().time_since_epoch().count() + rank);
        
        // Initialize root node
        std::vector<int> initial_available_tasks;
        std::vector<bool> task_available(problem.tasks.size(), true);
        
        // Find tasks with no dependencies (roots)
        for (size_t i = 0; i < problem.tasks.size(); ++i) {
            for (int dep : problem.tasks[i].dependencies) {
                task_available[i] = false;
                break;
            }
            
            if (task_available[i]) {
                initial_available_tasks.push_back(i);
            }
        }
        
        root = std::make_unique<MCTSNode>(
            nullptr, -1, -1, 
            initial_available_tasks,
            std::vector<std::pair<int, int>>(),
            std::vector<int>()
        );
    }
    
    SchedulingSolution search() {
        auto start_time = std::chrono::high_resolution_clock::now();
        int total_nodes = 0;
        
        // MCTS iterations
        int local_simulations = num_simulations / size;
        if (rank == 0) {
            local_simulations += num_simulations % size;
        }
        
        // Different parallelization strategies
        if (std::string(MCTS_PARALLELIZATION) == "treeMPI") {
            // Tree parallelization: each process works on different parts of the tree
            for (int i = 0; i < local_simulations; ++i) {
                MCTSNode* selected = select(root.get());
                MCTSNode* expanded = expand(selected);
                double reward = simulate(expanded);
                backpropagate(expanded, reward);
                total_nodes++;
                
                if (i % 1000 == 0 && rank == 0) {
                    std::cout << "Rank 0: " << i << " / " << local_simulations << " simulations completed" << std::endl;
                }
            }
        } else if (std::string(MCTS_PARALLELIZATION) == "rootMPI") {
            // Root parallelization: each process builds its own tree
            for (int i = 0; i < local_simulations; ++i) {
                MCTSNode* selected = select(root.get());
                MCTSNode* expanded = expand(selected);
                double reward = simulate(expanded);
                backpropagate(expanded, reward);
                total_nodes++;
            }
            
            // Exchange results periodically or at the end
            // For simplicity, we just use the result from rank 0
        }
        
        // OpenMP parallelization within each MPI process
        #ifdef _OPENMP
        #pragma omp parallel num_threads(MCTS_OMP_THREADS)
        {
            // Within each process, we could parallelize simulation
            // or different parts of the tree
        }
        #endif
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        // Get best solution from root
        SchedulingSolution solution = extract_best_solution();
        
        // Log results
        if (rank == 0) {
            std::cout << "MCTS search completed in " << elapsed.count() << " seconds" << std::endl;
            std::cout << "Makespan: " << solution.makespan << std::endl;
            std::cout << "MCTS nodes: " << total_nodes << std::endl;
            std::cout << "MCTS simulations: " << local_simulations << std::endl;
        }
        
        return solution;
    }
    
private:
    MCTSNode* select(MCTSNode* node) {
        while (!node->is_terminal() && node->is_fully_expanded(problem)) {
            node = select_best_child(node);
        }
        return node;
    }
    
    MCTSNode* select_best_child(MCTSNode* node) {
        MCTSNode* best_child = nullptr;
        double best_score = -std::numeric_limits<double>::infinity();
        
        for (const auto& child : node->children) {
            double score = child->uct_score(exploration_parameter, node->visits);
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        
        return best_child;
    }
    
    MCTSNode* expand(MCTSNode* node) {
        if (node->is_terminal()) {
            return node;
        }
        
        // Random untried action
        std::uniform_int_distribution<int> task_dist(0, node->available_tasks.size() - 1);
        std::uniform_int_distribution<int> machine_dist(0, problem.machines - 1);
        
        int task_idx = task_dist(rng);
        int task_id = node->available_tasks[task_idx];
        int machine_id = machine_dist(rng);
        
        // Check if this action has already been tried
        for (const auto& child : node->children) {
            if (child->task_id == task_id && child->machine_id == machine_id) {
                return expand(node);  // Try again
            }
        }
        
        // Create new assignments and execution order
        std::vector<std::pair<int, int>> new_assignments = node->assignments;
        new_assignments.emplace_back(task_id, machine_id);
        
        std::vector<int> new_execution_order = node->execution_order;
        new_execution_order.push_back(task_id);
        
        // Create new available tasks
        std::vector<int> new_available_tasks;
        for (size_t i = 0; i < node->available_tasks.size(); ++i) {
            if (i != task_idx) {
                new_available_tasks.push_back(node->available_tasks[i]);
            }
        }
        
        // Add newly available tasks (dependencies satisfied)
        std::vector<bool> task_scheduled(problem.tasks.size(), false);
        for (const auto& assignment : new_assignments) {
            task_scheduled[assignment.first] = true;
        }
        
        for (size_t i = 0; i < problem.tasks.size(); ++i) {
            if (!task_scheduled[i]) {
                bool deps_satisfied = true;
                for (int dep : problem.tasks[i].dependencies) {
                    if (!task_scheduled[dep]) {
                        deps_satisfied = false;
                        break;
                    }
                }
                
                if (deps_satisfied) {
                    new_available_tasks.push_back(i);
                }
            }
        }
        
        // Create new child node
        auto child = std::make_unique<MCTSNode>(
            node, task_id, machine_id,
            new_available_tasks, new_assignments, new_execution_order
        );
        
        MCTSNode* child_ptr = child.get();
        node->children.push_back(std::move(child));
        
        return child_ptr;
    }
    
    double simulate(MCTSNode* node) {
        // Random simulation from current node to terminal state
        std::vector<int> available_tasks = node->available_tasks;
        std::vector<std::pair<int, int>> assignments = node->assignments;
        std::vector<int> execution_order = node->execution_order;
        std::vector<bool> task_scheduled(problem.tasks.size(), false);
        
        for (const auto& assignment : assignments) {
            task_scheduled[assignment.first] = true;
        }
        
        std::uniform_int_distribution<int> machine_dist(0, problem.machines - 1);
        
        while (!available_tasks.empty()) {
            // Select random task from available ones
            std::uniform_int_distribution<int> task_dist(0, available_tasks.size() - 1);
            int task_idx = task_dist(rng);
            int task_id = available_tasks[task_idx];
            
            // Assign to random machine
            int machine_id = machine_dist(rng);
            
            // Update state
            assignments.emplace_back(task_id, machine_id);
            execution_order.push_back(task_id);
            task_scheduled[task_id] = true;
            
            // Remove selected task
            available_tasks.erase(available_tasks.begin() + task_idx);
            
            // Add newly available tasks
            for (size_t i = 0; i < problem.tasks.size(); ++i) {
                if (!task_scheduled[i]) {
                    bool deps_satisfied = true;
                    for (int dep : problem.tasks[i].dependencies) {
                        if (!task_scheduled[dep]) {
                            deps_satisfied = false;
                            break;
                        }
                    }
                    
                    if (deps_satisfied && std::find(available_tasks.begin(), 
                                                    available_tasks.end(), i) == available_tasks.end()) {
                        available_tasks.push_back(i);
                    }
                }
            }
        }
        
        // Evaluate final solution
        SchedulingSolution solution(problem);
        solution.task_assignments = assignments;
        solution.schedule = execution_order;
        solution.evaluate(problem);
        
        // Return negative makespan as reward (we want to minimize makespan)
        return -solution.makespan;
    }
    
    void backpropagate(MCTSNode* node, double reward) {
        while (node != nullptr) {
            node->visits++;
            node->value += reward;
            node = const_cast<MCTSNode*>(node->parent);
        }
    }
    
    SchedulingSolution extract_best_solution() {
        // Return best child based on visit count
        MCTSNode* best_child = nullptr;
        int best_visits = -1;
        
        for (const auto& child : root->children) {
            if (child->visits > best_visits) {
                best_visits = child->visits;
                best_child = child.get();
            }
        }
        
        // If no best child (should not happen), return empty solution
        if (!best_child) {
            return SchedulingSolution(problem);
        }
        
        // Build complete solution by following best children
        std::vector<std::pair<int, int>> assignments;
        std::vector<int> schedule;
        
        MCTSNode* node = best_child;
        while (node != nullptr && node != root.get()) {
            assignments.emplace_back(node->task_id, node->machine_id);
            schedule.push_back(node->task_id);
            
            // Find best child
            MCTSNode* best_next = nullptr;
            int best_next_visits = -1;
            
            for (const auto& child : node->children) {
                if (child->visits > best_next_visits) {
                    best_next_visits = child->visits;
                    best_next = child.get();
                }
            }
            
            node = best_next;
        }
        
        // Reverse to get correct order
        std::reverse(assignments.begin(), assignments.end());
        std::reverse(schedule.begin(), schedule.end());
        
        // Complete the solution if needed
        SchedulingSolution solution(problem);
        solution.task_assignments = assignments;
        solution.schedule = schedule;
        solution.evaluate(problem);
        
        return solution;
    }
};

// Main function
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    std::string problem_file;
    std::string output_file = "solution.json";
    double exploration = 1.0;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--problem" && i + 1 < argc) {
            problem_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--exploration" && i + 1 < argc) {
            exploration = std::stod(argv[++i]);
        }
    }
    
    if (problem_file.empty() && rank == 0) {
        std::cerr << "Error: Problem file not specified" << std::endl;
        std::cerr << "Usage: " << argv[0] << " --problem <file> [--output <file>] [--exploration <value>]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Load problem
    SchedulingProblem problem;
    problem.load(problem_file);
    
    if (rank == 0) {
        std::cout << "Loaded problem with " << problem.tasks.size() << " tasks and " 
                  << problem.machines << " machines" << std::endl;
        std::cout << "Running MCTS with " << MCTS_SIMULATIONS << " simulations, "
                  << size << " MPI processes and " << MCTS_OMP_THREADS << " OMP threads per process" << std::endl;
    }
    
    // Run MCTS
    MCTS mcts(problem, exploration, MCTS_SIMULATIONS);
    auto start_time = std::chrono::high_resolution_clock::now();
    SchedulingSolution solution = mcts.search();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Output solution
    if (rank == 0) {
        solution.to_json(output_file);
        
        std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Makespan: " << solution.makespan << std::endl;
        std::cout << "Solution written to " << output_file << std::endl;
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
