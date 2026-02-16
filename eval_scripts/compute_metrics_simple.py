#!/usr/bin/env python
"""
Compute performance metrics for AAM corridor evaluations.
Reads from eval_summary.csv files created by eval_metrics_logger.py

This is a simplified version that just aggregates pre-computed metrics
from graph_mpe_runner.py (via eval_metrics_logger.py).
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def load_evaluation_summary(eval_dir):
    """
    Load pre-computed metrics from eval_summary.csv.

    Args:
        eval_dir: Directory containing eval_summary.csv

    Returns:
        dict with metrics, or None if file not found
    """
    csv_file = os.path.join(eval_dir, 'eval_summary.csv')

    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)

    if len(df) == 0:
        return None

    # Return first (and only) row as dict
    return df.iloc[0].to_dict()


def create_results_table(results_df, output_dir):
    """Create formatted tables for paper."""

    # Group by scenario and num_agents
    grouped = results_df.groupby(['scenario', 'num_agents']).mean()

    # Create CSV
    csv_path = os.path.join(output_dir, 'metrics_table.csv')
    grouped.to_csv(csv_path)
    print(f"Saved CSV table to: {csv_path}")

    # Create LaTeX table
    latex_path = os.path.join(output_dir, 'metrics_table.tex')

    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance metrics for AAM corridor ")
        f.write("scenarios.}\n")
        f.write("\\label{table:results}\n")
        f.write("\\begin{tabular}{llcccccc}\n")
        f.write("\\hline\n")
        f.write("Scenario & Agents & $C\\%$ & $S\\%$ & $T$ (s) & ")
        f.write("$\\Delta d$ (m) & $I\\%$ & $\\Theta$ (ag/min) \\\\\n")
        f.write("\\hline\n")

        for (scenario, num_agents), row in grouped.iterrows():
            f.write(f"{scenario} & {num_agents} & ")
            f.write(f"{row['conformance_pct']:.1f} & ")
            f.write(f"{row['success_pct']:.1f} & ")
            f.write(f"{row['completion_time_mean']:.1f}")
            f.write(f"$\\pm${row['completion_time_std']:.1f} & ")
            f.write(f"{row['delta_d_mean']:.1f}")
            f.write(f"$\\pm${row['delta_d_std']:.1f} & ")
            f.write(f"{row['intervention_pct']:.1f} & ")
            f.write(f"{row['throughput']:.2f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table to: {latex_path}")

    return grouped


def main():
    parser = argparse.ArgumentParser(
        description='Compute AAM corridor performance metrics'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save computed metrics')

    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir if args.output_dir else results_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing evaluation results from: {results_dir}\n")

    # Find all evaluation subdirectories
    eval_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir()]

    if len(eval_dirs) == 0:
        print("No evaluation directories found!")
        return

    print(f"Found {len(eval_dirs)} evaluation runs\n")

    # Process each evaluation
    all_results = []

    for eval_dir in eval_dirs:
        dir_name = eval_dir.name
        print(f"Processing: {dir_name}...", end=' ')

        # Parse directory name: {scenario}_agents{num}_seed{seed}
        parts = dir_name.split('_')
        try:
            # Find 'agents' keyword
            agents_idx = None
            for i, p in enumerate(parts):
                if 'agents' in p:
                    agents_idx = i
                    break

            if agents_idx is None:
                print(f"Warning: Could not parse: {dir_name}")
                continue

            scenario = '_'.join(parts[:agents_idx])
            num_agents = int(''.join(filter(str.isdigit, parts[agents_idx])))

            # Find seed
            seed = 1  # default
            for p in parts:
                if 'seed' in p:
                    seed = int(''.join(filter(str.isdigit, p)))

        except Exception as e:
            print(f"Warning: Parse error: {e}")
            continue

        # Load metrics
        metrics = load_evaluation_summary(str(eval_dir))

        if metrics is None:
            print("FAILED (no eval_summary.csv)")
            continue

        # Add metadata
        metrics['scenario'] = scenario
        metrics['num_agents'] = num_agents
        metrics['seed'] = seed

        # Map metric names for consistency
        result = {
            'scenario': scenario,
            'num_agents': num_agents,
            'seed': seed,
            'conformance_pct': metrics.get('conformance_pct_mean', 0.0),
            'success_pct': metrics.get('success_rate_mean', 0.0),
            'completion_time_mean': metrics.get('completion_time_mean', 0.0),
            'completion_time_std': metrics.get('completion_time_std', 0.0),
            'delta_d_mean': metrics.get('delta_d_mean', 0.0),
            'delta_d_std': metrics.get('delta_d_std', 0.0),
            'intervention_pct': metrics.get('spacing_violations_mean', 0.0),
            'throughput': metrics.get('throughput_mean', 0.0),
        }

        all_results.append(result)
        print("✓")

    if len(all_results) == 0:
        print("\nNo valid results found!")
        print("Did you run evaluations with --eval_mode flag?")
        return

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Save raw results
    raw_path = os.path.join(output_dir, 'raw_metrics.csv')
    results_df.to_csv(raw_path, index=False)
    print(f"\nSaved raw metrics to: {raw_path}")

    # Create summary tables
    print("\nCreating summary tables...")
    summary = create_results_table(results_df, output_dir)

    print("\n" + "="*80)
    print("METRICS COMPUTATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nSummary statistics:")
    print(summary)

    print("\nMetric descriptions:")
    print("  C%  - Conformance to corridor boundaries (higher is better)")
    print("  S%  - Success rate (higher is better)")
    print("  T   - Completion time in seconds (lower is better)")
    print("  Δd  - Separation violation magnitude in meters (lower better)")
    print("  I%  - Need for tactical intervention (lower is better)")
    print("  Θ   - Exit throughput in agents/minute (higher is better)")


if __name__ == '__main__':
    main()
