#!/usr/bin/env python
"""
Plot averaged outflow rate over multiple episodes.

Usage:
    python eval_scripts/plot_outflow_rate_averaged.py \
        --trajectory_dir eval_40agents_combined_graph \
        --num_episodes 5 \
        --output_dir eval_40agents_combined_graph/plots
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import List, Tuple
import glob

# Set font to Times New Roman for publication
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# Import functions from the original script
import sys
sys.path.append(os.path.dirname(__file__))

# We'll duplicate the necessary functions here for simplicity
def compute_exit_times(trajectory_data: dict) -> List[float]:
    """Extract exit times from trajectory data."""
    if 'agent_exit_times' in trajectory_data:
        agent_exit_times = trajectory_data['agent_exit_times']
        num_timesteps = trajectory_data['positions'].shape[0]

        valid_exits = []
        for agent_id, exit_time in enumerate(agent_exit_times):
            if 0 <= exit_time < num_timesteps:
                valid_exits.append(float(exit_time))

        return valid_exits
    return []


def compute_outflow_rate(exit_times: List[float],
                         num_timesteps: int,
                         window_size: int = 60,
                         dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous outflow rate over time."""
    timesteps = np.arange(num_timesteps)
    outflow_rates = np.zeros(num_timesteps)

    exit_times_array = np.array(exit_times) if len(exit_times) > 0 else np.array([])

    for t in range(num_timesteps):
        window_start = max(0, t - window_size)
        window_end = t

        if len(exit_times_array) > 0:
            exits_in_window = np.sum((exit_times_array >= window_start) &
                                    (exit_times_array < window_end))
        else:
            exits_in_window = 0

        window_duration_sec = (window_end - window_start) * dt
        window_duration_min = window_duration_sec / 60.0

        if window_duration_min > 0:
            outflow_rates[t] = exits_in_window / window_duration_min
        else:
            outflow_rates[t] = 0.0

    return timesteps, outflow_rates


def infer_agent_max_speeds(data: dict, dt: float) -> np.ndarray:
    """
    Infer per-agent max speed from trajectory by taking the 95th percentile
    of observed speeds, then snapping to the nearest known speed tier
    (140 knots slow or 175 knots fast).

    Returns:
        agent_max_speeds_kms: array of shape (num_agents,) in km/s
    """
    positions = data['positions']  # (timesteps, num_agents, 2) in km
    num_agents = positions.shape[1]

    slow_speed_kms = 140 * 0.514444 * 0.001
    fast_speed_kms = 175 * 0.514444 * 0.001
    threshold_kms = (slow_speed_kms + fast_speed_kms) / 2  # midpoint ≈ 157.5 kts

    agent_max_speeds = np.zeros(num_agents)

    position_diffs = np.diff(positions, axis=0)         # (timesteps-1, num_agents, 2)
    velocities = position_diffs / dt                    # km/s
    speeds = np.linalg.norm(velocities, axis=2)         # (timesteps-1, num_agents)

    for agent_id in range(num_agents):
        agent_speeds = speeds[:, agent_id]
        # Use 95th percentile to avoid noise; agents at goal have speed 0
        peak_speed = np.percentile(agent_speeds[agent_speeds > 0.001], 95) \
            if np.any(agent_speeds > 0.001) else fast_speed_kms

        # Snap to nearest speed tier
        agent_max_speeds[agent_id] = (
            slow_speed_kms if peak_speed < threshold_kms else fast_speed_kms
        )

    num_slow = int(np.sum(agent_max_speeds < threshold_kms))
    num_fast = num_agents - num_slow
    print(f"  Inferred agent speeds: {num_slow} slow (140 kts), {num_fast} fast (175 kts)")

    return agent_max_speeds


def compute_optimal_throughput(data: dict, window_size: int, dt: float,
                               agent_max_speeds_kms: np.ndarray = None) -> np.ndarray:
    """
    Compute theoretical optimal throughput.

    Args:
        data: trajectory data dict
        window_size: sliding window in timesteps
        dt: timestep duration in seconds
        agent_max_speeds_kms: per-agent max speed in km/s, or None to use 175 kts for all
    """
    all_routes = [
        [0, 1, 2, 3, 5, 12, 13, 14, 16],
        [0, 1, 2, 3, 5, 12, 13, 15, 17],
        [4, 5, 12, 13, 14, 16],
        [4, 5, 12, 13, 15, 17],
        [6, 7, 9, 11, 12, 13, 14, 16],
        [6, 7, 9, 11, 12, 13, 15, 17],
        [6, 8, 10, 11, 12, 13, 14, 16],
        [6, 8, 10, 11, 12, 13, 15, 17],
    ]

    corridor_entrances = data['corridor_entrances']
    corridor_exits = data['corridor_exits']
    num_agents = int(data['num_agents'])

    route_lengths = []
    for route in all_routes:
        length = 0.0
        for corridor_id in route:
            entrance = corridor_entrances[corridor_id]
            exit_pos = corridor_exits[corridor_id]
            corridor_length = float(np.linalg.norm(exit_pos - entrance))
            length += corridor_length
        route_lengths.append(length)

    positions = data['positions']
    start_positions = positions[0, :, :]

    # Default: all agents at 175 knots
    if agent_max_speeds_kms is None:
        agent_max_speeds_kms = np.full(num_agents, 175.0 * 0.514444 * 0.001)

    optimal_exit_times = []
    for agent_id in range(num_agents):
        route_idx = agent_id % len(all_routes)
        route = all_routes[route_idx]
        route_length = route_lengths[route_idx]

        start_pos = start_positions[agent_id]
        first_corridor_entrance = corridor_entrances[route[0]]
        approach_distance = float(np.linalg.norm(start_pos - first_corridor_entrance))

        total_distance = approach_distance + route_length
        optimal_time_seconds = total_distance / agent_max_speeds_kms[agent_id]
        optimal_time_timesteps = optimal_time_seconds / dt

        optimal_exit_times.append(optimal_time_timesteps)

    num_timesteps = positions.shape[0]
    _, optimal_rate = compute_outflow_rate(optimal_exit_times, num_timesteps, window_size, dt)

    return optimal_rate


def plot_averaged_outflow_rate(trajectory_dir: str, num_episodes: int,
                               output_dir: str, window_size: int = 60, dt: float = 1.0):
    """
    Create averaged outflow rate plot from multiple episodes.

    Args:
        trajectory_dir: Directory containing trajectory files
        num_episodes: Number of episodes to average
        output_dir: Directory to save plots
        window_size: Sliding window size in timesteps
        dt: Timestep duration in seconds
    """
    print(f"\nLoading {num_episodes} episodes from: {trajectory_dir}")

    # Find trajectory files
    pattern = os.path.join(trajectory_dir, "trajectory_*_episode_*.npz")
    all_files = sorted(glob.glob(pattern))

    if len(all_files) == 0:
        print(f"Error: No trajectory files found matching: {pattern}")
        return

    # Select first num_episodes files
    trajectory_files = all_files[:num_episodes]
    print(f"Found {len(trajectory_files)} trajectory files")

    # Collect outflow rates from all episodes
    all_actual_rates = []
    all_optimal_rates = []

    for i, traj_file in enumerate(trajectory_files):
        print(f"\nProcessing episode {i}...")
        data = np.load(traj_file, allow_pickle=True)

        num_timesteps = data['positions'].shape[0]
        num_agents = data['positions'].shape[1]

        # Compute exit times and rates
        exit_times = compute_exit_times(data)
        _, actual_rate = compute_outflow_rate(exit_times, num_timesteps, window_size, dt)
        all_actual_rates.append(actual_rate)

        # Compute optimal rate (only once, should be similar across episodes)
        if i == 0:
            optimal_rate = compute_optimal_throughput(data, window_size, dt)
            all_optimal_rates.append(optimal_rate)

        print(f"  Episode {i}: {len(exit_times)}/{num_agents} agents exited")

    # Pad all episodes to the same length (max timesteps across episodes)
    max_timesteps = max(len(r) for r in all_actual_rates)
    padded_rates = np.zeros((len(all_actual_rates), max_timesteps))
    for i, rate in enumerate(all_actual_rates):
        padded_rates[i, :len(rate)] = rate
    all_actual_rates = padded_rates

    # Pad optimal rate to same length if needed
    if len(optimal_rate) < max_timesteps:
        optimal_rate = np.pad(optimal_rate, (0, max_timesteps - len(optimal_rate)))
    else:
        optimal_rate = optimal_rate[:max_timesteps]

    # Compute mean and std (only over timesteps that have data)
    mean_actual_rate = np.mean(all_actual_rates, axis=0)
    std_actual_rate = np.std(all_actual_rates, axis=0)

    # Time axis
    num_timesteps = max_timesteps
    timesteps = np.arange(num_timesteps)
    time_seconds = timesteps * dt

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot optimal throughput
    ax.plot(time_seconds, optimal_rate, 'b--', linewidth=2, alpha=0.7,
            label='Optimal Throughput (max speed)')

    # Plot mean actual rate with std as shaded region
    ax.plot(time_seconds, mean_actual_rate, 'k-', linewidth=3,
            label=f'Actual Outflow (mean, n={num_episodes})')
    ax.fill_between(time_seconds,
                     mean_actual_rate - std_actual_rate,
                     mean_actual_rate + std_actual_rate,
                     color='gray', alpha=0.3,)

    # Formatting
    ax.set_xlabel('Episode Time (seconds)', fontsize=20)
    ax.set_ylabel('Outflow Rate (agents/minute)', fontsize=20)
    ax.set_title(f'Exit Throughput Over Time,  Homogeneous (N={num_agents} agents)',
                 fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc='upper right')
    ax.tick_params(labelsize=16)

    # Add statistics box
    actual_peak = np.max(mean_actual_rate)
    optimal_peak = np.max(optimal_rate)

    stats_text = (
        f'Episodes: {num_episodes}\n'
        f'Actual peak: {actual_peak:.1f} ag/min\n'
        f'Optimal peak: {optimal_peak:.1f} ag/min\n'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tight layout
    fig.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)

    png_file = os.path.join(output_dir, f'outflow_rate_averaged_{num_episodes}eps.png')
    pdf_file = os.path.join(output_dir, f'outflow_rate_averaged_{num_episodes}eps.pdf')

    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    print(f"\nAveraged plots saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.close(fig)

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean peak rate: {actual_peak:.2f} ± {np.std([np.max(r) for r in all_actual_rates]):.2f} ag/min")
    print(f"  Optimal peak rate: {optimal_peak:.2f} ag/min")
    print(f"  Efficiency: {(actual_peak / optimal_peak * 100):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Plot averaged outflow rate over multiple episodes'
    )
    parser.add_argument('--trajectory_dir', type=str, required=True,
                       help='Directory containing trajectory NPZ files')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to average (default: 5)')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Sliding window size in timesteps (default: 60)')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Timestep duration in seconds (default: 1.0)')

    args = parser.parse_args()

    plot_averaged_outflow_rate(args.trajectory_dir, args.num_episodes,
                               args.output_dir, args.window_size, args.dt)

    print("\n✓ Averaged outflow rate analysis complete!")


if __name__ == '__main__':
    main()
