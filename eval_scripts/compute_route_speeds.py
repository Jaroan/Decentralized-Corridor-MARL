#!/usr/bin/env python
"""
Compute speed statistics for each of the 8 routes in simple_combined_graph.

For each route, computes:
- Average speed
- Minimum speed
- Maximum speed
- Standard deviation

Usage:
    python eval_scripts/compute_route_speeds.py \
        --trajectory_file eval_40agents_combined_graph/trajectory_40agents_homogeneous_episode_0.npz \
        --output_dir eval_40agents_combined_graph/analysis
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import Dict, List, Tuple

# Set font to Times New Roman for publication
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'


# 8 routes in simple_combined_graph (corridor IDs)
ALL_ROUTES = [
    [0, 1, 2, 3, 5, 12, 13, 14, 16],     # Route 0: Entry A → Left exit
    [0, 1, 2, 3, 5, 12, 13, 15, 17],     # Route 1: Entry A → Right exit
    [4, 5, 12, 13, 14, 16],              # Route 2: Entry B → Left exit
    [4, 5, 12, 13, 15, 17],              # Route 3: Entry B → Right exit
    [6, 7, 9, 11, 12, 13, 14, 16],       # Route 4: Entry CL → Left exit
    [6, 7, 9, 11, 12, 13, 15, 17],       # Route 5: Entry CL → Right exit
    [6, 8, 10, 11, 12, 13, 14, 16],      # Route 6: Entry CR → Left exit
    [6, 8, 10, 11, 12, 13, 15, 17],      # Route 7: Entry CR → Right exit
]

ROUTE_NAMES = [
    "A→T16 (left)",
    "A→T17 (right)",
    "B→T16 (left)",
    "B→T17 (right)",
    "CL→T16 (left)",
    "CL→T17 (right)",
    "CR→T16 (left)",
    "CR→T17 (right)",
]


def compute_route_length(route: List[int], corridor_entrances: np.ndarray,
                         corridor_exits: np.ndarray) -> float:
    """
    Compute total length of a route including corridors and gaps between them.

    For each corridor: includes the corridor segment length
    Between corridors: includes the gap distance (exit of corridor N to entrance of corridor N+1)
    """
    length = 0.0

    for i, corridor_id in enumerate(route):
        # Add corridor segment length
        entrance = corridor_entrances[corridor_id]
        exit_pos = corridor_exits[corridor_id]
        corridor_length = float(np.linalg.norm(exit_pos - entrance))
        length += corridor_length

        # Add gap to next corridor (if not last corridor)
        if i < len(route) - 1:
            next_corridor_id = route[i + 1]
            next_entrance = corridor_entrances[next_corridor_id]
            gap_distance = float(np.linalg.norm(next_entrance - exit_pos))
            length += gap_distance

    return length


def compute_travel_time(agent_id: int, data: dict, dt: float = 1.0) -> float:
    """
    Compute time taken by an agent to complete their route.

    Time = exit_time - entry_time (when agent enters first corridor)
    If agent doesn't exit, returns -1.
    """
    if 'agent_exit_times' not in data:
        return -1.0

    exit_time = data['agent_exit_times'][agent_id]
    num_timesteps = data['positions'].shape[0]

    # Check if agent exited successfully
    if exit_time < 0 or exit_time >= num_timesteps:
        return -1.0

    # Find when agent entered first corridor (simplified: use timestep 0 as start)
    # Could be refined by detecting when agent actually enters first corridor
    start_time = 0.0

    travel_time_timesteps = exit_time - start_time
    travel_time_seconds = travel_time_timesteps * dt

    return travel_time_seconds


def compute_actual_trajectory_distance(agent_id: int, data: dict,
                                       exit_time: int) -> float:
    """
    Compute actual distance traveled by following agent's trajectory.

    Sums the distance between consecutive positions along the path.
    """
    positions = data['positions'][:int(exit_time) + 1, agent_id, :]

    total_distance = 0.0
    for i in range(1, len(positions)):
        segment_distance = np.linalg.norm(positions[i] - positions[i-1])
        total_distance += segment_distance

    return float(total_distance)


def find_corridor_entry_time(agent_id: int, data: dict,
                             first_corridor_entrance: np.ndarray,
                             entry_threshold: float = 0.5) -> int:
    """
    Find the timestep when agent enters the first corridor.

    Returns timestep when agent is within entry_threshold km of corridor entrance.
    """
    positions = data['positions'][:, agent_id, :]

    for t in range(len(positions)):
        dist_to_entrance = np.linalg.norm(positions[t] - first_corridor_entrance)
        if dist_to_entrance < entry_threshold:
            return t

    return 0  # If not found, return start


def compute_corridor_trajectory_distance(agent_id: int, data: dict,
                                         exit_time: int,
                                         entry_time: int) -> float:
    """
    Compute actual distance traveled from corridor entry to exit.

    Excludes the approach distance before entering first corridor.
    """
    positions = data['positions'][entry_time:int(exit_time) + 1, agent_id, :]

    total_distance = 0.0
    for i in range(1, len(positions)):
        segment_distance = np.linalg.norm(positions[i] - positions[i-1])
        total_distance += segment_distance

    return float(total_distance)


def compute_optimal_straight_line_distance(start_pos: np.ndarray,
                                           route: List[int],
                                           corridor_exits: np.ndarray) -> float:
    """
    Compute optimal (shortest) distance: straight line from start to final exit.
    """
    # Last corridor in route is the exit corridor
    final_corridor_id = route[-1]
    final_exit_pos = corridor_exits[final_corridor_id]

    # Straight-line distance from start to final exit
    optimal_distance = float(np.linalg.norm(final_exit_pos - start_pos))

    return optimal_distance


def compute_route_speeds(trajectory_file: str, dt: float = 1.0) -> pd.DataFrame:
    """
    Compute speed statistics for each route.

    Returns:
        DataFrame with columns: route_id, route_name, agent_id, distance_km,
                                time_s, speed_kms, speed_knots
    """
    print(f"Loading trajectory data from: {trajectory_file}")
    data = np.load(trajectory_file, allow_pickle=True)

    num_agents = data['positions'].shape[1]
    corridor_entrances = data['corridor_entrances']
    corridor_exits = data['corridor_exits']
    start_positions = data['positions'][0, :, :]

    print(f"Number of agents: {num_agents}")

    # Compute route lengths
    route_lengths = []
    for route in ALL_ROUTES:
        length = compute_route_length(route, corridor_entrances, corridor_exits)
        route_lengths.append(length)

    print("\nRoute lengths:")
    for i, (name, length) in enumerate(zip(ROUTE_NAMES, route_lengths)):
        print(f"  Route {i} ({name}): {length:.2f} km")

    # Analyze each agent
    results = []

    for agent_id in range(num_agents):
        route_idx = agent_id % len(ALL_ROUTES)
        route = ALL_ROUTES[route_idx]
        route_name = ROUTE_NAMES[route_idx]

        # Get route distance (sum of straight-line corridor segments)
        corridor_optimal_distance = route_lengths[route_idx]

        # Get start position
        start_pos = start_positions[agent_id]
        first_corridor_entrance = corridor_entrances[route[0]]
        approach_optimal_distance = float(np.linalg.norm(start_pos - first_corridor_entrance))

        # Optimal total distance (straight lines)
        optimal_total_distance = approach_optimal_distance + corridor_optimal_distance

        # Get travel time and exit time
        travel_time = compute_travel_time(agent_id, data, dt)
        exit_time = data['agent_exit_times'][agent_id]

        if travel_time <= 0:
            print(f"  Agent {agent_id}: Did not complete route")
            continue

        # Find when agent enters first corridor
        corridor_entry_time = find_corridor_entry_time(
            agent_id, data, first_corridor_entrance
        )

        # Compute ACTUAL trajectory distance (following curved path)
        # Full distance from start
        actual_trajectory_distance = compute_actual_trajectory_distance(
            agent_id, data, exit_time
        )

        # Corridor-only distance (from first corridor entry to exit)
        corridor_actual_distance = compute_corridor_trajectory_distance(
            agent_id, data, exit_time, corridor_entry_time
        )

        # Compute speed based on OPTIMAL corridor distance (more physically meaningful)
        # This avoids inflated speeds from detours
        corridor_speed_kms = corridor_optimal_distance / travel_time  # km/s
        corridor_speed_knots = corridor_speed_kms / (0.514444 * 0.001)  # convert to knots

        # Compute path efficiency (corridor-only)
        corridor_efficiency = (corridor_optimal_distance / corridor_actual_distance) * 100.0

        # Compute path efficiency (full path)
        path_efficiency = (optimal_total_distance / actual_trajectory_distance) * 100.0

        # Flag outliers (severe detours or physically impossible speeds)
        is_outlier = (corridor_efficiency < 50.0) or (corridor_speed_knots > 175.0)

        results.append({
            'route_id': route_idx,
            'route_name': route_name,
            'agent_id': agent_id,
            'approach_optimal_km': approach_optimal_distance,
            'corridor_optimal_km': corridor_optimal_distance,
            'optimal_total_km': optimal_total_distance,
            'actual_trajectory_km': actual_trajectory_distance,
            'corridor_actual_km': corridor_actual_distance,
            'corridor_efficiency_pct': corridor_efficiency,
            'path_efficiency_pct': path_efficiency,
            'time_s': travel_time,
            'speed_kms': corridor_speed_kms,
            'speed_knots': corridor_speed_knots,
            'is_outlier': is_outlier,
        })

    return pd.DataFrame(results)


def print_route_statistics(df: pd.DataFrame):
    """Print summary statistics for each route."""
    # Report outliers
    outliers = df[df['is_outlier']]
    if len(outliers) > 0:
        print("\n" + "="*100)
        print(f"OUTLIERS DETECTED: {len(outliers)} agents excluded from averages")
        print("="*100)
        for _, row in outliers.iterrows():
            print(f"  Agent {int(row['agent_id'])} (Route {int(row['route_id'])}): "
                  f"{row['corridor_actual_km']:.1f} km actual vs {row['corridor_optimal_km']:.1f} km optimal "
                  f"({row['corridor_efficiency_pct']:.1f}% efficient)")
        print("")

    # Filter out outliers for statistics
    df_clean = df[~df['is_outlier']].copy()

    print("\n" + "="*100)
    print("ROUTE SPEED AND DISTANCE STATISTICS (outliers excluded)")
    print("="*100)

    grouped = df_clean.groupby('route_id')

    # Speed statistics
    print(f"\n{'Route':<25} {'N':<5} {'Avg Speed':<12} {'Min Speed':<12} {'Max Speed':<12} {'Std':<10}")
    print(f"{'':25} {'':5} {'(knots)':<12} {'(knots)':<12} {'(knots)':<12} {'(knots)':<10}")
    print("-"*100)

    for route_id, group in grouped:
        route_name = ROUTE_NAMES[route_id]
        n = len(group)
        avg_speed = group['speed_knots'].mean()
        min_speed = group['speed_knots'].min()
        max_speed = group['speed_knots'].max()
        std_speed = group['speed_knots'].std()

        print(f"{route_name:<25} {n:<5} {avg_speed:>10.2f}  {min_speed:>10.2f}  {max_speed:>10.2f}  {std_speed:>10.2f}")

    # Overall speed statistics
    print("-"*100)
    print(f"{'Overall':<25} {len(df_clean):<5} {df_clean['speed_knots'].mean():>10.2f}  "
          f"{df_clean['speed_knots'].min():>10.2f}  {df_clean['speed_knots'].max():>10.2f}  "
          f"{df_clean['speed_knots'].std():>10.2f}")

    # Distance statistics
    print("\n" + "="*100)
    print("DISTANCE COMPARISON: Actual vs Optimal")
    print("="*100)
    print(f"\n{'Route':<25} {'Optimal Dist':<15} {'Actual Dist':<15} {'Extra Dist':<15} {'Efficiency':<12}")
    print(f"{'':25} {'(km)':<15} {'(km)':<15} {'(km)':<15} {'(%)':<12}")
    print("-"*100)

    for route_id, group in grouped:
        route_name = ROUTE_NAMES[route_id]
        optimal_avg = group['optimal_total_km'].mean()
        actual_avg = group['actual_trajectory_km'].mean()
        extra_dist = actual_avg - optimal_avg
        efficiency = group['path_efficiency_pct'].mean()

        print(f"{route_name:<25} {optimal_avg:>13.2f}  {actual_avg:>13.2f}  {extra_dist:>13.2f}  {efficiency:>10.1f}")

    # Overall distance statistics
    print("-"*100)
    optimal_overall = df_clean['optimal_total_km'].mean()
    actual_overall = df_clean['actual_trajectory_km'].mean()
    extra_overall = actual_overall - optimal_overall
    efficiency_overall = df_clean['path_efficiency_pct'].mean()
    print(f"{'Overall':<25} {optimal_overall:>13.2f}  {actual_overall:>13.2f}  {extra_overall:>13.2f}  {efficiency_overall:>10.1f}")

    print("\n" + "="*100)
    print(f"Max theoretical speed: 175.0 knots")
    print(f"Actual mean speed: {df_clean['speed_knots'].mean():.2f} knots ({df_clean['speed_knots'].mean()/175.0*100:.1f}% of max)")
    print(f"Path efficiency: {efficiency_overall:.1f}% (actual distance / optimal distance)")
    print(f"Outliers excluded: {len(outliers)} / {len(df)} agents")
    print("="*100)


def plot_route_speeds(df: pd.DataFrame, output_dir: str):
    """Create box plot of speeds by route."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plot
    route_data = []
    route_labels = []

    for route_id in range(len(ALL_ROUTES)):
        route_df = df[df['route_id'] == route_id]
        if len(route_df) > 0:
            route_data.append(route_df['speed_knots'].values)
            route_labels.append(f"R{route_id}")

    # Create box plot
    bp = ax.boxplot(route_data, labels=route_labels, patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Add horizontal line at max speed
    ax.axhline(y=175.0, color='r', linestyle='--', linewidth=2,
               label='Max Speed (175 knots)', alpha=0.7)

    # Formatting
    ax.set_xlabel('Route ID', fontsize=18)
    ax.set_ylabel('Average Speed (knots)', fontsize=18)
    ax.set_title('Agent Speed Distribution by Route', fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)

    # Add route names as second x-axis labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(1, len(route_labels) + 1))
    ax2.set_xticklabels([ROUTE_NAMES[i].split('→')[0] for i in range(len(route_labels))],
                         fontsize=12, rotation=0)
    ax2.set_xlabel('Entry Point', fontsize=16)

    fig.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_file = os.path.join(output_dir, 'route_speeds_boxplot.png')
    pdf_file = os.path.join(output_dir, 'route_speeds_boxplot.pdf')

    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    print(f"\nBox plot saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.close(fig)


def plot_speed_vs_distance(df: pd.DataFrame, output_dir: str):
    """Create scatter plot of speed vs total distance traveled."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by route
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_ROUTES)))

    for route_id in range(len(ALL_ROUTES)):
        route_df = df[df['route_id'] == route_id]
        if len(route_df) > 0:
            ax.scatter(route_df['actual_trajectory_km'], route_df['speed_knots'],
                      c=[colors[route_id]], label=f"R{route_id}", s=60, alpha=0.7)

    # Formatting
    ax.set_xlabel('Total Distance Traveled (km)', fontsize=18)
    ax.set_ylabel('Average Speed (knots)', fontsize=18)
    ax.set_title('Agent Speed vs Distance by Route', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, ncol=2, loc='best')
    ax.tick_params(labelsize=14)

    # Add max speed line
    ax.axhline(y=175.0, color='r', linestyle='--', linewidth=2, alpha=0.5)

    fig.tight_layout()

    # Save
    png_file = os.path.join(output_dir, 'speed_vs_distance.png')
    pdf_file = os.path.join(output_dir, 'speed_vs_distance.pdf')

    fig.savefig(png_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(pdf_file, bbox_inches='tight', pad_inches=0.05)

    print(f"\nScatter plot saved:")
    print(f"  PNG: {png_file}")
    print(f"  PDF: {pdf_file}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compute speed statistics for each route'
    )
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='Path to trajectory NPZ file')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Timestep duration in seconds (default: 1.0)')

    args = parser.parse_args()

    # Compute route speeds
    df = compute_route_speeds(args.trajectory_file, args.dt)

    if len(df) == 0:
        print("\nError: No completed routes found!")
        return

    # Print statistics
    print_route_statistics(df)

    # Save detailed CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(args.output_dir, 'route_speeds_detailed.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")

    # Save summary statistics CSV (excluding outliers)
    df_clean = df[~df['is_outlier']].copy()

    summary = df_clean.groupby(['route_id', 'route_name']).agg({
        'speed_knots': ['count', 'mean', 'min', 'max', 'std'],
        'corridor_optimal_km': 'mean',
        'corridor_actual_km': 'mean',
        'corridor_efficiency_pct': 'mean',
        'optimal_total_km': 'mean',
        'actual_trajectory_km': 'mean',
        'path_efficiency_pct': 'mean',
        'time_s': 'mean',
    }).round(2)

    summary_file = os.path.join(args.output_dir, 'route_speeds_summary.csv')
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")

    # Create plots
    print("\nGenerating plots...")
    plot_route_speeds(df, args.output_dir)
    plot_speed_vs_distance(df, args.output_dir)

    print("\n✓ Route speed analysis complete!")


if __name__ == '__main__':
    main()
