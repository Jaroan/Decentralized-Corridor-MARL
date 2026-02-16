#!/usr/bin/env python
"""
Compute performance metrics for AAM corridor evaluations.

Metrics computed:
1. Conformance to corridor boundaries (C%) - Higher is better
2. Success rate (S%) - Higher is better
3. Completion time per episode (T, s) - Lower is better
4. Violation of separation minimum (Δd, m) - Lower is better (mean ± std)
5. Need for tactical intervention (I%) - Lower is better

Output: CSV tables and LaTeX tables for paper submission
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_episode_data(eval_dir):
    """
    Load evaluation data from a single evaluation run.

    Reads from eval_summary.csv created by eval_metrics_logger.py
    (only created when --eval_mode flag is used).
    """
    # Look for CSV summary file
    csv_file = os.path.join(eval_dir, 'eval_summary.csv')

    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping...")
        print(f"  (Did you run with --eval_mode flag?)")
        return None

    # Read CSV
    df = pd.read_csv(csv_file)

    # Return the first (and only) row as a dict
    if len(df) == 0:
        print(f"Warning: {csv_file} is empty, skipping...")
        return None

    return df.iloc[0].to_dict()


def process_evaluation_run(eval_dir):
    """Process a single evaluation run and return metrics dict."""
    data = load_episode_data(eval_dir)

    if data is None:
        return None

    # Data is already aggregated from eval_summary.csv
    # Just map to our standard metric names
    return {
        'conformance_pct': data.get('conformance_pct_mean', 0.0),
        'success_pct': data.get('success_rate_mean', 0.0),
        'completion_time_mean': data.get('completion_time_mean', 0.0),
        'completion_time_std': data.get('completion_time_std', 0.0),
        'delta_d_mean': data.get('delta_d_mean', 0.0),
        'delta_d_std': data.get('delta_d_std', 0.0),
        'num_violations': 0,  # Not tracked separately
        'intervention_pct': data.get('spacing_violations_mean', 0.0),
    }


# Remove all the individual metric computation functions since
# they're already computed in graph_mpe_runner.py
def compute_conformance(episode_data, corridor_width=0.6):
    """
    Compute conformance to corridor boundaries (C%).

    Returns the percentage of time each aircraft stays within corridor boundaries
    while traversing through it, averaged over all aircraft.

    Args:
        episode_data: dict with keys:
            - 'agent_positions': list of np.array of shape (num_steps, num_agents, 2)
            - 'agent_tubes': list of np.array of shape (num_steps, num_agents)
            - 'tube_params': tube parameters
            - 'in_tube': list of np.array of shape (num_steps, num_agents) - bool
        corridor_width: width of corridor in world units

    Returns:
        conformance_pct: float, percentage (0-100)
    """
    conformance_times = []

    for ep_idx, (positions, tubes, in_tube) in enumerate(zip(
        episode_data['agent_positions'],
        episode_data['agent_tubes'],
        episode_data['in_tube']
    )):
        num_steps, num_agents = positions.shape[0], positions.shape[1]

        for agent_id in range(num_agents):
            in_corridor_steps = 0
            within_bounds_steps = 0

            for step in range(num_steps):
                if in_tube[step, agent_id]:
                    in_corridor_steps += 1

                    # Check if within bounds (lateral deviation from centerline)
                    # This requires tube_coords calculation
                    # Assuming episode_data has 'lateral_deviation'
                    lateral_dev = episode_data.get('lateral_deviation', None)
                    if lateral_dev is not None:
                        if abs(lateral_dev[ep_idx][step, agent_id]) <= corridor_width / 2:
                            within_bounds_steps += 1

            if in_corridor_steps > 0:
                conformance_times.append(100.0 * within_bounds_steps / in_corridor_steps)

    if len(conformance_times) == 0:
        return 0.0

    return np.mean(conformance_times)


def compute_success_rate(episode_data):
    """
    Compute success rate (S%).

    Success = aircraft reached goal within episode length.

    Args:
        episode_data: dict with 'goal_reached' - list of np.array (num_agents,) bool

    Returns:
        success_pct: float, percentage (0-100)
    """
    total_agents = 0
    successful_agents = 0

    for ep_goals in episode_data['goal_reached']:
        total_agents += len(ep_goals)
        successful_agents += np.sum(ep_goals)

    if total_agents == 0:
        return 0.0

    return 100.0 * successful_agents / total_agents


def compute_completion_time(episode_data):
    """
    Compute average completion time per episode (T, s).

    Time when all agents reach their goals (or episode length if not all reach).

    Args:
        episode_data: dict with 'completion_times' - list of floats (seconds)

    Returns:
        mean_time: float, seconds
        std_time: float, seconds
    """
    times = episode_data['completion_times']
    return np.mean(times), np.std(times)


def compute_separation_violations(episode_data, separation_min=SEPARATION_MINIMUM):
    """
    Compute separation violation statistics (Δd).

    Δd = separation_min - actual_separation (only when violation occurs)

    Args:
        episode_data: dict with 'pairwise_distances' - list of np.array
                      of shape (num_steps, num_agents, num_agents)
        separation_min: minimum separation in meters

    Returns:
        mean_violation: float, mean Δd in meters (only when violation occurs)
        std_violation: float, std Δd in meters
        num_violations: int, total number of violation instances
    """
    violations = []

    for distances in episode_data['pairwise_distances']:
        num_steps, num_agents, _ = distances.shape

        for step in range(num_steps):
            for i in range(num_agents):
                for j in range(i+1, num_agents):
                    dist = distances[step, i, j]
                    if dist < separation_min:
                        delta_d = separation_min - dist
                        violations.append(delta_d)

    if len(violations) == 0:
        return 0.0, 0.0, 0

    return np.mean(violations), np.std(violations), len(violations)


def compute_intervention_need(episode_data, separation_min=SEPARATION_MINIMUM):
    """
    Compute need for tactical intervention (I%).

    I% = (time with separation violations) / (total time in corridors) * 100

    Args:
        episode_data: dict with:
            - 'pairwise_distances': list of np.array (num_steps, num_agents, num_agents)
            - 'in_tube': list of np.array (num_steps, num_agents) bool
        separation_min: minimum separation in meters

    Returns:
        intervention_pct: float, percentage (0-100)
    """
    total_corridor_time = 0
    violation_time = 0

    for distances, in_tube in zip(episode_data['pairwise_distances'],
                                   episode_data['in_tube']):
        num_steps, num_agents, _ = distances.shape

        for step in range(num_steps):
            # Count agents in corridor at this step
            agents_in_corridor = np.sum(in_tube[step])

            if agents_in_corridor >= 2:
                total_corridor_time += 1

                # Check for any violations at this step
                has_violation = False
                for i in range(num_agents):
                    if not in_tube[step, i]:
                        continue
                    for j in range(i+1, num_agents):
                        if not in_tube[step, j]:
                            continue
                        if distances[step, i, j] < separation_min:
                            has_violation = True
                            break
                    if has_violation:
                        break

                if has_violation:
                    violation_time += 1

    if total_corridor_time == 0:
        return 0.0

    return 100.0 * violation_time / total_corridor_time


def process_evaluation_run(eval_dir):
    """Process a single evaluation run and compute all metrics."""
    data = load_episode_data(eval_dir)

    if data is None:
        return None

    metrics = {}

    # Conformance to corridor boundaries
    metrics['conformance_pct'] = compute_conformance(data)

    # Success rate
    metrics['success_pct'] = compute_success_rate(data)

    # Completion time
    mean_time, std_time = compute_completion_time(data)
    metrics['completion_time_mean'] = mean_time
    metrics['completion_time_std'] = std_time

    # Separation violations
    mean_viol, std_viol, num_viol = compute_separation_violations(data)
    metrics['delta_d_mean'] = mean_viol
    metrics['delta_d_std'] = std_viol
    metrics['num_violations'] = num_viol

    # Intervention need
    metrics['intervention_pct'] = compute_intervention_need(data)

    return metrics


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
        f.write("\\caption{Performance metrics for AAM corridor scenarios.}\n")
        f.write("\\label{table:results}\n")
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\hline\n")
        f.write("Scenario & Agents & $C\\%$ & $S\\%$ & $T$ (s) & $\\Delta d$ (m) & $I\\%$ \\\\\n")
        f.write("\\hline\n")

        for (scenario, num_agents), row in grouped.iterrows():
            f.write(f"{scenario} & {num_agents} & ")
            f.write(f"{row['conformance_pct']:.1f} & ")
            f.write(f"{row['success_pct']:.1f} & ")
            f.write(f"{row['completion_time_mean']:.1f}$\\pm${row['completion_time_std']:.1f} & ")
            f.write(f"{row['delta_d_mean']:.1f}$\\pm${row['delta_d_std']:.1f} & ")
            f.write(f"{row['intervention_pct']:.1f} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table to: {latex_path}")

    return grouped


def main():
    parser = argparse.ArgumentParser(description='Compute AAM corridor performance metrics')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save computed metrics (default: same as results_dir)')

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

        # Parse directory name to extract scenario and num_agents
        # Expected format: {scenario}_agents{num}_seed{seed}
        parts = dir_name.split('_')
        try:
            scenario_idx = parts.index('agents') - 1
            scenario = '_'.join(parts[:scenario_idx+1])

            agents_idx = parts.index('agents')
            num_agents = int(''.join(filter(str.isdigit, parts[agents_idx])))

            seed_idx = [i for i, p in enumerate(parts) if 'seed' in p][0]
            seed = int(''.join(filter(str.isdigit, parts[seed_idx])))
        except:
            print(f"Warning: Could not parse directory name: {dir_name}")
            continue

        # Compute metrics
        metrics = process_evaluation_run(str(eval_dir))

        if metrics is None:
            print("FAILED")
            continue

        metrics['scenario'] = scenario
        metrics['num_agents'] = num_agents
        metrics['seed'] = seed

        all_results.append(metrics)
        print("✓")

    if len(all_results) == 0:
        print("\nNo valid results found!")
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
    print("  Δd  - Separation violation magnitude in meters (lower is better)")
    print("  I%  - Need for tactical intervention (lower is better)")


if __name__ == '__main__':
    main()
