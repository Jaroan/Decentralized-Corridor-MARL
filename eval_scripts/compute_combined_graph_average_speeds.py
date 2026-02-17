#!/usr/bin/env python
"""
Compute average speeds for simple combined graph experiments.

Analyzes trajectory data from different agent counts and computes average speeds.

Usage:
    python eval_scripts/compute_combined_graph_average_speeds.py
"""

import numpy as np
import pandas as pd
import glob
import os
from typing import Tuple


def compute_average_speeds_from_trajectories(trajectory_dir: str,
                                             num_episodes: int) -> Tuple[float, float, float]:
    """
    Compute average speed across all agents and episodes.

    Returns:
        (mean_speed, min_speed, max_speed) in knots
    """
    pattern = os.path.join(trajectory_dir, "trajectory_*_episode_*.npz")
    traj_files = sorted(glob.glob(pattern))[:num_episodes]

    if len(traj_files) == 0:
        print(f"  No trajectories found in {trajectory_dir}")
        return (0.0, 0.0, 0.0)

    all_speeds = []

    for traj_file in traj_files:
        data = np.load(traj_file, allow_pickle=True)

        # Get positions and compute velocities
        if 'positions' not in data:
            print(f"  Warning: No position data in {traj_file}")
            continue

        positions = data['positions']  # shape: (timesteps, num_agents, 2) in km
        dt = float(data['dt']) if 'dt' in data else 1.0  # seconds

        # Exclude first and last positions (spawn/exit may have unusual speeds)
        if len(positions) > 2:
            positions = positions[1:-1]  # Remove first and last timestep

        # Compute velocities from positions
        # positions are in km, so velocity will be in km/s
        position_diffs = np.diff(positions, axis=0)  # (timesteps-1, num_agents, 2)
        velocity_vectors = position_diffs / dt  # km/s

        # Compute speed magnitudes
        speeds_kms = np.linalg.norm(velocity_vectors, axis=2)  # km/s

        # Convert to knots (1 knot = 0.514444 * 0.001 km/s)
        speeds_knots = speeds_kms / (0.514444 * 0.001)

        # For each agent, find when they stop (speed < 60 knots) and exclude those timesteps
        num_agents = speeds_knots.shape[1]
        for agent_id in range(num_agents):
            agent_speeds = speeds_knots[:, agent_id]

            # Find first timestep where speed drops below 60 knots (agent reached goal)
            stopped_mask = agent_speeds < 60.0
            if np.any(stopped_mask):
                first_stop_idx = np.argmax(stopped_mask)
                # Only use speeds before agent stopped
                if first_stop_idx > 0:
                    active_speeds = agent_speeds[:first_stop_idx]
                else:
                    # Agent stopped immediately, skip this agent
                    continue
            else:
                # Agent never stopped, use all speeds
                active_speeds = agent_speeds

            if len(active_speeds) > 0:
                agent_mean_speed = np.mean(active_speeds)
                all_speeds.append(agent_mean_speed)

    if len(all_speeds) == 0:
        return (0.0, 0.0, 0.0)

    mean_speed = np.mean(all_speeds)
    min_speed = np.min(all_speeds)
    max_speed = np.max(all_speeds)

    return (mean_speed, min_speed, max_speed)


def main():
    print("\n" + "="*80)
    print("SIMPLE COMBINED GRAPH AVERAGE SPEED ANALYSIS")
    print("="*80 + "\n")

    # Experiment configurations
    experiments = [
        {"num_agents": 10, "num_episodes": 50, "dir": "eval_combined_graph_10agents"},
        {"num_agents": 20, "num_episodes": 50, "dir": "eval_combined_graph_20agents"},
        {"num_agents": 30, "num_episodes": 50, "dir": "eval_combined_graph_30agents"},
        {"num_agents": 40, "num_episodes": 10, "dir": "eval_combined_graph_40agents"},
    ]

    results = []

    for exp in experiments:
        print(f"Processing {exp['num_agents']} agents ({exp['num_episodes']} episodes)...")

        mean_speed, min_speed, max_speed = compute_average_speeds_from_trajectories(
            exp['dir'], exp['num_episodes']
        )

        results.append({
            'num_agents': exp['num_agents'],
            'num_episodes': exp['num_episodes'],
            'mean_speed_knots': mean_speed,
            'min_speed_knots': min_speed,
            'max_speed_knots': max_speed,
        })

        print(f"  Mean speed: {mean_speed:.2f} knots")
        print(f"  Speed range: {min_speed:.2f} - {max_speed:.2f} knots")
        print()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_file = "combined_graph_average_speeds.csv"
    df.to_csv(output_file, index=False)

    print("="*80)
    print("SUMMARY TABLE")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print()

    print(f"Results saved to: {output_file}")
    print()

    # Print insights
    print("="*80)
    print("INSIGHTS")
    print("="*80)
    print(f"Max theoretical speed: 175.0 knots")
    print(f"\nSpeed vs Agent Count:")
    for idx, row in df.iterrows():
        pct_of_max = (row['mean_speed_knots'] / 175.0) * 100
        print(f"  {int(row['num_agents']):2d} agents: {row['mean_speed_knots']:6.2f} knots "
              f"({pct_of_max:5.1f}% of max)")

    # Compute slowdown
    if len(df) > 1:
        baseline_speed = df.iloc[0]['mean_speed_knots']
        print(f"\nSlowdown vs 10 agents baseline:")
        for idx, row in df.iterrows():
            if row['num_agents'] == 10:
                continue
            slowdown_pct = ((baseline_speed - row['mean_speed_knots']) / baseline_speed) * 100
            print(f"  {int(row['num_agents']):2d} agents: {slowdown_pct:+5.1f}% "
                  f"({row['mean_speed_knots']:.2f} vs {baseline_speed:.2f} knots)")

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
