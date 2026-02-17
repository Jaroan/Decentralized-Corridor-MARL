#!/usr/bin/env python
"""
Plot outflow rate (throughput) over time for simple_combined_graph scenario.

This script tracks when agents exit through the final corridors (T16, T17)
and plots the cumulative outflow rate over episode timesteps.

Usage:
    python eval_scripts/plot_outflow_rate.py \
        --trajectory_file eval_output/trajectory_40agents_episode_0.npz \
        --output_dir eval_output/plots
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import List, Tuple

# Set font to Times New Roman for publication
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'


def compute_exit_times(trajectory_data: dict) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute exit times for each agent from trajectory data.

    Args:
        trajectory_data: Dictionary loaded from trajectory NPZ file

    Returns:
        (t16_exits, t17_exits, all_exits): Lists of timesteps when agents exited
            through T16 (left), T17 (right), and combined
    """
    # Extract data
    positions = trajectory_data['positions']  # (timesteps, num_agents, 2)
    num_timesteps, num_agents, _ = positions.shape

    # Check if we have pre-computed exit times
    if 'agent_exit_times' in trajectory_data:
        agent_exit_times = trajectory_data['agent_exit_times']

        # Filter out agents that didn't exit (marked with -1 or episode_length)
        valid_exits = []
        for agent_id, exit_time in enumerate(agent_exit_times):
            if 0 <= exit_time < num_timesteps:
                valid_exits.append(float(exit_time))

        print(f"\nProcessing {num_agents} agents over {num_timesteps} timesteps...")
        print(f"Using pre-computed exit times from trajectory data")
        print(f"\nTotal exits: {len(valid_exits)}")
        print(f"  Agents that didn't exit: {num_agents - len(valid_exits)}")

        # For simple_combined_graph, we don't have per-corridor exit data
        # So we'll treat all exits as combined
        # You can split them 50/50 as a rough approximation, or leave separate plots empty
        all_exit_times = valid_exits

        # Rough approximation: assume exits alternate between T16 and T17
        # (This is a simplification - actual corridor choice depends on routes)
        t16_exit_times = [t for i, t in enumerate(sorted(valid_exits)) if i % 2 == 0]
        t17_exit_times = [t for i, t in enumerate(sorted(valid_exits)) if i % 2 == 1]

        print(f"  T16 (left exit, approx): {len(t16_exit_times)}")
        print(f"  T17 (right exit, approx): {len(t17_exit_times)}")
        print(f"\nNote: Per-corridor breakdown is approximate.")
        print(f"      Use total rate for accurate throughput measurement.")

        return t16_exit_times, t17_exit_times, all_exit_times

    # Fallback: Try to extract from phases/tubes if available
    phases = trajectory_data.get('phases', None)
    current_tubes = trajectory_data.get('tubes', None)

    # Track exit times
    t16_exit_times = []  # Left exit
    t17_exit_times = []  # Right exit
    all_exit_times = []

    # Track which agents have already exited
    has_exited = np.zeros(num_agents, dtype=bool)

    # Exit corridor IDs for simple_combined_graph
    LEFT_EXIT_CORRIDOR = 16  # T16
    RIGHT_EXIT_CORRIDOR = 17  # T17

    print(f"\nProcessing {num_agents} agents over {num_timesteps} timesteps...")
    print("Extracting exit times from phase/tube data...")

    # Scan through time
    for t in range(num_timesteps):
        for agent_id in range(num_agents):
            if has_exited[agent_id]:
                continue

            # Check if this agent has reached the exit
            # An agent has exited when it completes the final corridor (phase 2 of T16 or T17)
            if current_tubes is not None and phases is not None:
                current_tube = int(current_tubes[t, agent_id])
                current_phase = int(phases[t, agent_id])

                # Check if agent is in phase 2 (completed) of an exit corridor
                if current_phase == 2:
                    if current_tube == LEFT_EXIT_CORRIDOR:
                        t16_exit_times.append(t)
                        all_exit_times.append(t)
                        has_exited[agent_id] = True
                    elif current_tube == RIGHT_EXIT_CORRIDOR:
                        t17_exit_times.append(t)
                        all_exit_times.append(t)
                        has_exited[agent_id] = True

    print(f"\nTotal exits: {len(all_exit_times)}")
    print(f"  T16 (left exit): {len(t16_exit_times)}")
    print(f"  T17 (right exit): {len(t17_exit_times)}")
    print(f"  Agents that didn't exit: {num_agents - len(all_exit_times)}")

    return t16_exit_times, t17_exit_times, all_exit_times


def compute_outflow_rate(exit_times: List[float],
                         num_timesteps: int,
                         window_size: int = 60,
                         dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous outflow rate over time.

    Args:
        exit_times: List of timesteps when agents exited
        num_timesteps: Total number of timesteps in episode
        window_size: Sliding window size in timesteps for rate computation
        dt: Timestep duration in seconds

    Returns:
        (timesteps, outflow_rates): Arrays of timesteps and corresponding
            outflow rates in agents/minute
    """
    timesteps = np.arange(num_timesteps)
    outflow_rates = np.zeros(num_timesteps)

    # Convert exit times to array
    exit_times_array = np.array(exit_times) if len(exit_times) > 0 else np.array([])

    # Compute rate at each timestep using sliding window
    for t in range(num_timesteps):
        window_start = max(0, t - window_size)
        window_end = t

        # Count exits in this window
        if len(exit_times_array) > 0:
            exits_in_window = np.sum((exit_times_array >= window_start) &
                                    (exit_times_array < window_end))
        else:
            exits_in_window = 0

        # Compute rate: (exits in window) / (window duration in minutes)
        window_duration_sec = (window_end - window_start) * dt
        window_duration_min = window_duration_sec / 60.0

        if window_duration_min > 0:
            outflow_rates[t] = exits_in_window / window_duration_min
        else:
            outflow_rates[t] = 0.0

    return timesteps, outflow_rates


def compute_optimal_throughput(data: dict, window_size: int, dt: float) -> Tuple[np.ndarray, List[float]]:
    """
    Compute theoretical optimal throughput if agents traveled at max speed.

    Args:
        data: Trajectory data dictionary
        window_size: Sliding window size for rate computation
        dt: Timestep duration in seconds

    Returns:
        (timesteps, optimal_rate, optimal_exits): Optimal outflow rate over time and exit times
    """
    # Route definitions from simple_combined_graph
    all_routes = [
        [0, 1, 2, 3, 5, 12, 13, 14, 16],    # 0: A → left exit
        [0, 1, 2, 3, 5, 12, 13, 15, 17],    # 1: A → right exit
        [4, 5, 12, 13, 14, 16],              # 2: B → left exit
        [4, 5, 12, 13, 15, 17],              # 3: B → right exit
        [6, 7, 9, 11, 12, 13, 14, 16],       # 4: CL → left exit
        [6, 7, 9, 11, 12, 13, 15, 17],       # 5: CL → right exit
        [6, 8, 10, 11, 12, 13, 14, 16],      # 6: CR → left exit
        [6, 8, 10, 11, 12, 13, 15, 17],      # 7: CR → right exit
    ]

    # Extract corridor geometry
    corridor_entrances = data['corridor_entrances']  # (num_corridors, 2)
    corridor_exits = data['corridor_exits']  # (num_corridors, 2)
    num_agents = data['num_agents']

    # Calculate route lengths
    print("\nCalculating optimal throughput...")
    route_lengths = []
    for route_idx, route in enumerate(all_routes):
        length = 0.0
        for corridor_id in route:
            entrance = corridor_entrances[corridor_id]
            exit_pos = corridor_exits[corridor_id]
            corridor_length = float(np.linalg.norm(exit_pos - entrance))
            length += corridor_length
        route_lengths.append(length)
        print(f"  Route {route_idx}: {len(route)} corridors, {length:.2f} km")

    # Get agent starting positions
    positions = data['positions']  # (timesteps, num_agents, 2)
    start_positions = positions[0, :, :]  # (num_agents, 2)

    # Maximum speed: 175 knots = 0.09 km/s (approximately)
    max_speed_knots = 175.0
    max_speed_kms = max_speed_knots * 0.514444 * 0.001  # Convert to km/s

    print(f"\nMax speed: {max_speed_knots} knots = {max_speed_kms:.4f} km/s")

    # Calculate optimal exit time for each agent
    optimal_exit_times = []
    for agent_id in range(num_agents):
        # Agents cycle through routes
        route_idx = agent_id % len(all_routes)
        route = all_routes[route_idx]
        route_length = route_lengths[route_idx]

        # Starting position
        start_pos = start_positions[agent_id]

        # Distance from start to first corridor entrance
        first_corridor_entrance = corridor_entrances[route[0]]
        approach_distance = float(np.linalg.norm(start_pos - first_corridor_entrance))

        # Total distance = approach + route length
        total_distance = approach_distance + route_length

        # Optimal time = distance / max_speed
        optimal_time_seconds = total_distance / max_speed_kms
        optimal_time_timesteps = optimal_time_seconds / dt

        optimal_exit_times.append(optimal_time_timesteps)

    print(f"\nOptimal exit times: {np.min(optimal_exit_times):.1f} - {np.max(optimal_exit_times):.1f} timesteps")

    # Compute optimal outflow rate
    num_timesteps = positions.shape[0]
    _, optimal_rate = compute_outflow_rate(optimal_exit_times, num_timesteps, window_size, dt)

    return optimal_rate, optimal_exit_times


def plot_outflow_rate(trajectory_file: str, output_dir: str, window_size: int = 60, dt: float = 1.0):
    """
    Create outflow rate plot from trajectory data.

    Args:
        trajectory_file: Path to trajectory NPZ file
        output_dir: Directory to save plots
        window_size: Sliding window size in timesteps (default: 60 for smoother curves)
        dt: Timestep duration in seconds
    """
    # Load trajectory data
    print(f"Loading trajectory data from: {trajectory_file}")
    data = np.load(trajectory_file, allow_pickle=True)

    # Get episode info
    num_timesteps = data['positions'].shape[0]
    num_agents = data['positions'].shape[1]

    # Compute exit times
    t16_exits, t17_exits, all_exits = compute_exit_times(data)

    # Compute actual outflow rates
    print("\nComputing actual outflow rates...")
    print(f"Using sliding window of {window_size} timesteps ({window_size * dt:.1f} seconds)")
    timesteps, total_rate = compute_outflow_rate(all_exits, num_timesteps, window_size, dt)

    # Compute optimal throughput
    optimal_rate, optimal_exits = compute_optimal_throughput(data, window_size, dt)

    # Convert timesteps to time in seconds
    time_seconds = timesteps * dt

    # Create figure with tighter layout
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot optimal throughput (theoretical maximum)
    ax.plot(time_seconds, optimal_rate, 'b--', linewidth=2, alpha=0.7,
            label='Optimal Throughput (max speed)')

    # Plot actual outflow rate
    ax.plot(time_seconds, total_rate, 'k-', linewidth=3, label='Actual Outflow Rate')

    # Formatting with larger fonts
    ax.set_xlabel('Episode Time (seconds)', fontsize=20)
    ax.set_ylabel('Outflow Rate (agents/minute)', fontsize=20)
    ax.set_title(f'Exit Throughput Over Time (N={num_agents} agents)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc='upper right')
    ax.tick_params(labelsize=16)

    # Add statistics box
    active_rate = total_rate[total_rate > 0.1]  # Only consider non-zero periods
    active_optimal = optimal_rate[optimal_rate > 0.1]

    actual_peak = np.max(total_rate)
    optimal_peak = np.max(optimal_rate)
    efficiency = (actual_peak / optimal_peak * 100) if optimal_peak > 0 else 0

    stats_text = (
        f'Total exits: {len(all_exits)}/{num_agents} agents\n'
        f'Actual peak: {actual_peak:.1f} ag/min\n'
        f'Optimal peak: {optimal_peak:.1f} ag/min\n'
        # f'Efficiency: {efficiency:.1f}%\n'
        # f'Window: {window_size * dt:.1f} sec'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tight layout to minimize white space
    fig.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(trajectory_file))[0]
    png_file = os.path.join(output_dir, f'{base_name}_outflow_rate.png')
    pdf_file = os.path.join(output_dir, f'{base_name}_outflow_rate.pdf')

    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    print(f"\nPlots saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.close(fig)

    # Also create a cumulative exits plot
    plot_cumulative_exits(all_exits, t16_exits, t17_exits, num_timesteps,
                         num_agents, dt, output_dir, base_name)


def plot_cumulative_exits(all_exits: List[float], t16_exits: List[float],
                         t17_exits: List[float], num_timesteps: int,
                         num_agents: int, dt: float, output_dir: str, base_name: str):
    """Create cumulative exits plot."""
    timesteps = np.arange(num_timesteps)
    time_seconds = timesteps * dt

    # Compute cumulative exits
    cumulative_all = np.zeros(num_timesteps)
    cumulative_t16 = np.zeros(num_timesteps)
    cumulative_t17 = np.zeros(num_timesteps)

    for t in timesteps:
        cumulative_all[t] = np.sum(np.array(all_exits) <= t)
        cumulative_t16[t] = np.sum(np.array(t16_exits) <= t)
        cumulative_t17[t] = np.sum(np.array(t17_exits) <= t)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot cumulative exits
    ax.plot(time_seconds, cumulative_t16, 'b-', linewidth=2, alpha=0.6, label='T16 (Left Exit)')
    ax.plot(time_seconds, cumulative_t17, 'r-', linewidth=2, alpha=0.6, label='T17 (Right Exit)')
    ax.plot(time_seconds, cumulative_all, 'k-', linewidth=3, label='Total Exits')

    # Add horizontal line at total agents
    ax.axhline(y=num_agents, color='gray', linestyle='--', linewidth=1,
               label=f'Total Agents ({num_agents})')

    # Formatting with larger fonts
    ax.set_xlabel('Episode Time (seconds)', fontsize=20)
    ax.set_ylabel('Cumulative Number of Exits', fontsize=20)
    ax.set_title(f'Cumulative Exits Over Time (N={num_agents} agents)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc='lower right')
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, num_agents * 1.1)

    # Tight layout to minimize white space
    fig.tight_layout()

    # Save plot
    png_file = os.path.join(output_dir, f'{base_name}_cumulative_exits.png')
    pdf_file = os.path.join(output_dir, f'{base_name}_cumulative_exits.pdf')

    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot outflow rate over time for AAM corridor evaluation'
    )
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='Path to trajectory NPZ file')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Sliding window size in timesteps for rate calculation (default: 5)')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Timestep duration in seconds (default: 1.0)')

    args = parser.parse_args()

    # Verify trajectory file exists
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        return

    # Create plots
    plot_outflow_rate(args.trajectory_file, args.output_dir,
                     args.window_size, args.dt)

    print("\n✓ Outflow rate analysis complete!")


if __name__ == '__main__':
    main()
