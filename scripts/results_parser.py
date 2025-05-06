#!/usr/bin/env python3
"""
MCTS Experiment Results Parser

This script parses and aggregates results from MCTS scheduling optimization experiments.
"""

import os
import sys
import json
import csv
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class ExperimentResultsParser:
    """Parses and aggregates results from MCTS scheduling optimization experiments."""

    def __init__(self, base_dir: str):
        """
        Initialize the results parser.
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.experiment_results_dir = self.base_dir / "experiment_results"
    
    def get_all_run_ids(self) -> List[str]:
        """
        Get all run IDs from the experiment results directory.
        
        Returns:
            List of run IDs
        """
        run_dirs = [d.name for d in self.experiment_results_dir.iterdir() 
                   if d.is_dir() and (d / "run_config.json").exists()]
        return run_dirs
    
    def parse_run_config(self, run_id: str) -> Dict[str, Any]:
        """
        Parse the configuration for a run.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Dictionary containing run configuration
        """
        config_path = self.experiment_results_dir / run_id / "run_config.json"
        
        if not config_path.exists():
            print(f"Warning: Config file not found for run_id: {run_id}")
            return {}
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def parse_solution_file(self, run_id: str) -> Dict[str, Any]:
        """
        Parse the solution file for a run.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Dictionary containing solution data
        """
        solution_path = self.experiment_results_dir / run_id / "solution.json"
        
        if not solution_path.exists():
            print(f"Warning: Solution file not found for run_id: {run_id}")
            return {}
        
        with open(solution_path, 'r') as f:
            return json.load(f)
    
    def parse_job_output(self, run_id: str) -> Dict[str, Any]:
        """
        Parse the job output log for a run.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Dictionary containing extracted results
        """
        output_path = self.experiment_results_dir / run_id / "job_output.log"
        
        if not output_path.exists():
            print(f"Warning: Job output log not found for run_id: {run_id}")
            return {}
        
        results = {
            "makespan": None,
            "execution_time": None,
            "mcts_nodes": None,
            "mcts_simulations": None
        }
        
        try:
            with open(output_path, 'r') as f:
                content = f.read()
                
                # Extract makespan
                makespan_match = re.search(r"Makespan:\s+(\d+\.?\d*)", content)
                if makespan_match:
                    results["makespan"] = float(makespan_match.group(1))
                
                # Extract execution time
                time_match = re.search(r"Execution time:\s+(\d+\.?\d*)\s+seconds", content)
                if time_match:
                    results["execution_time"] = float(time_match.group(1))
                
                # Extract MCTS nodes
                nodes_match = re.search(r"MCTS nodes:\s+(\d+)", content)
                if nodes_match:
                    results["mcts_nodes"] = int(nodes_match.group(1))
                
                # Extract MCTS simulations
                sims_match = re.search(r"MCTS simulations:\s+(\d+)", content)
                if sims_match:
                    results["mcts_simulations"] = int(sims_match.group(1))
                
            return results
            
        except Exception as e:
            print(f"Error parsing job output for run_id {run_id}: {e}")
            return results
    
    def parse_system_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Parse system metrics logs for a run.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Dictionary containing aggregated metrics
        """
        run_dir = self.experiment_results_dir / run_id
        metric_files = list(run_dir.glob("system_metrics.*.log"))
        
        if not metric_files:
            print(f"Warning: No system metrics files found for run_id: {run_id}")
            return {}
        
        all_metrics = []
        
        for metric_file in metric_files:
            try:
                df = pd.read_csv(metric_file)
                all_metrics.append(df)
            except Exception as e:
                print(f"Error reading metrics file {metric_file}: {e}")
        
        if not all_metrics:
            return {}
        
        # Combine metrics from all files
        if len(all_metrics) > 1:
            combined_df = pd.concat(all_metrics, ignore_index=True)
        else:
            combined_df = all_metrics[0]
        
        # Calculate aggregated metrics
        metrics = {
            "cpu_pct_mean": combined_df["cpu_pct"].mean(),
            "cpu_pct_max": combined_df["cpu_pct"].max(),
            "mem_rss_kb_max": combined_df["mem_rss_kb"].max(),
            "mem_vms_kb_max": combined_df["mem_vms_kb"].max(),
            "io_read_kb_total": combined_df["io_read_kb"].max(),
            "io_write_kb_total": combined_df["io_write_kb"].max()
        }
        
        return metrics
    
    def check_trace_files(self, run_id: str) -> Dict[str, Any]:
        """
        Check for trace files and get their information.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Dictionary containing trace file information
        """
        trace_dir = self.experiment_results_dir / run_id / "trace_files"
        
        if not trace_dir.exists() or not trace_dir.is_dir():
            return {"trace_files_exist": False, "trace_files_count": 0, "trace_files_total_size_mb": 0}
        
        trace_files = list(trace_dir.glob("*"))
        total_size = sum(f.stat().st_size for f in trace_files if f.is_file())
        
        return {
            "trace_files_exist": len(trace_files) > 0,
            "trace_files_count": len(trace_files),
            "trace_files_total_size_mb": total_size / (1024 * 1024)
        }
    
    def aggregate_run_results(self, run_id: str) -> Dict[str, Any]:
        """
        Aggregate all results for a single run.
        
        Args:
            run_id: The experiment run ID
            
        Returns:
            Dictionary containing all aggregated results
        """
        # Collect results from different sources
        config = self.parse_run_config(run_id)
        solution = self.parse_solution_file(run_id)
        job_output = self.parse_job_output(run_id)
        metrics = self.parse_system_metrics(run_id)
        trace_info = self.check_trace_files(run_id)
        
        # Extract key configuration parameters
        mcts_settings = config.get("mcts_settings", {})
        build_settings = config.get("build_settings", {})
        parallel_settings = config.get("parallel_settings", {})
        
        # Combine all results
        results = {
            "run_id": run_id,
            "problem_instance": config.get("problem_instance", ""),
            "git_hash": config.get("git_hash", ""),
            "slurm_job_id": config.get("slurm_job_id", ""),
            "submission_timestamp": config.get("submission_timestamp", ""),
            
            # MCTS settings
            "mcts_simulations": mcts_settings.get("simulations", 0),
            "mcts_exploration": mcts_settings.get("exploration", 0),
            
            # Build settings
            "parallelization": build_settings.get("parallelization", ""),
            "compiler": build_settings.get("compiler", ""),
            "optimization": build_settings.get("optimization", ""),
            
            # Parallel settings
            "mpi_processes": parallel_settings.get("processes", 0),
            "omp_threads": parallel_settings.get("omp_threads", 0),
            "total_cores": parallel_settings.get("processes", 0) * parallel_settings.get("omp_threads", 0),
            
            # Tracing
            "tracing_enabled": config.get("tracing", False),
            
            # Solution results
            "makespan": solution.get("makespan", job_output.get("makespan", None)),
            "execution_time": solution.get("execution_time", job_output.get("execution_time", None)),
            "mcts_nodes_explored": solution.get("nodes_explored", job_output.get("mcts_nodes", None)),
            "mcts_actual_simulations": solution.get("simulations_performed", job_output.get("mcts_simulations", None)),
            
            # System metrics
            "cpu_pct_mean": metrics.get("cpu_pct_mean", None),
            "cpu_pct_max": metrics.get("cpu_pct_max", None),
            "mem_rss_kb_max": metrics.get("mem_rss_kb_max", None),
            "mem_vms_kb_max": metrics.get("mem_vms_kb_max", None),
            "mem_rss_gb_max": metrics.get("mem_rss_kb_max", 0) / (1024 * 1024) if metrics.get("mem_rss_kb_max") else None,
            "io_read_kb_total": metrics.get("io_read_kb_total", None),
            "io_write_kb_total": metrics.get("io_write_kb_total", None),
            
            # Trace information
            "trace_files_exist": trace_info.get("trace_files_exist", False),
            "trace_files_count": trace_info.get("trace_files_count", 0),
            "trace_files_total_size_mb": trace_info.get("trace_files_total_size_mb", 0),
            
            # Paths
            "results_dir": str(self.experiment_results_dir / run_id)
        }
        
        return results
    
    def generate_results_table(self, run_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a results table for the specified runs or all runs.
        
        Args:
            run_ids: List of run IDs to include, or None for all runs
            
        Returns:
            Pandas DataFrame containing the results table
        """
        if run_ids is None:
            run_ids = self.get_all_run_ids()
        
        all_results = []
        for run_id in run_ids:
            run_results = self.aggregate_run_results(run_id)
            all_results.append(run_results)
        
        return pd.DataFrame(all_results)
    
    def export_results_to_csv(self, output_path: str, run_ids: Optional[List[str]] = None) -> str:
        """
        Export results table to CSV.
        
        Args:
            output_path: Path to the output CSV file
            run_ids: List of run IDs to include, or None for all runs
            
        Returns:
            Path to the CSV file
        """
        df = self.generate_results_table(run_ids)
        df.to_csv(output_path, index=False)
        return output_path


def main():
    """Main function to parse arguments and run the results parser."""
    parser = argparse.ArgumentParser(description="MCTS Experiment Results Parser")
    
    # Base directory
    parser.add_argument("--base-dir", type=str, default=os.getcwd(),
                        help="Base directory for experiments")
    
    # Output file
    parser.add_argument("--output", type=str, default="experiment_results.csv",
                        help="Path to output CSV file")
    
    # Specific run IDs
    parser.add_argument("--run-ids", type=str, nargs="+",
                        help="Specific run IDs to include")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize results parser
    parser = ExperimentResultsParser(args.base_dir)
    
    # Export results
    output_path = parser.export_results_to_csv(args.output, args.run_ids)
    print(f"Results exported to: {output_path}")


if __name__ == "__main__":
    import re  # Import at the top in real code
    main()
