#!/usr/bin/env python
"""
Visualize agent trajectories from saved trajectory data.

Usage:
    python eval_scripts/plot_trajectories.py --trajectory_file trajectory_episode_0.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
import argparse
import os

# Set matplotlib to use Times New Roman fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts for math (similar to Times)
plt.rcParams['font.size'] = 14


def load_trajectory_data(filepath):
    """Load trajectory data from npz file."""
    data = np.load(filepath, allow_pickle=True)
    result = {
        'positions': data['positions'],  # (num_steps, num_agents, 2)
        'velocities': data['velocities'],  # (num_steps, num_agents)
        'headings': data['headings'],  # (num_steps, num_agents)
        'phases': data['phases'],  # (num_steps, num_agents)
        'tubes': data['tubes'],  # (num_steps, num_agents)
        'dt': float(data['dt']),
        'num_agents': int(data['num_agents']),
        'episode_length': int(data['episode_length']),
        'agent_exit_times': data['agent_exit_times']
    }

    # Load corridor geometries if available
    if 'num_corridors' in data:
        result['corridors'] = {
            'num_corridors': int(data['num_corridors']),
            'entrances': data['corridor_entrances'],
            'exits': data['corridor_exits'],
            'widths': data['corridor_widths'],
            'n_vecs': data['corridor_n_vecs']
        }
    else:
        result['corridors'] = None

    return result


def draw_corridors(ax, corridor_data):
    """
    Draw corridor rectangles on the plot.

    Args:
        ax: matplotlib axes object
        corridor_data: corridor geometry dict from load_trajectory_data()
    """
    if corridor_data is None:
        return

    for i in range(corridor_data['num_corridors']):
        entrance = corridor_data['entrances'][i]
        exit_pt = corridor_data['exits'][i]
        width = corridor_data['widths'][i]
        n_vec = corridor_data['n_vecs'][i]

        # Compute the four corners of the rectangle
        # entrance and exit are the centerline points
        # n_vec is the perpendicular direction
        # Increase width by 0.05 for visualization
        half_width = (width + 0.05) / 2.0

        # Four corners: entrance +/- half_width * n_vec, exit +/- half_width * n_vec
        corners = np.array([
            entrance + half_width * n_vec,
            entrance - half_width * n_vec,
            exit_pt - half_width * n_vec,
            exit_pt + half_width * n_vec
        ])

        # Create polygon patch
        corridor_patch = patches.Polygon(
            corners,
            closed=True,
            edgecolor='gray',
            facecolor='none',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6,
            zorder=1
        )
        ax.add_patch(corridor_patch)


def plot_all_trajectories(data, output_dir, highlight_agents=None):
    """
    Plot all agent trajectories with color-coded velocity.
    Creates 2 subplots: left=trajectories, right=corridors colored by avg velocity.

    Args:
        data: dict from load_trajectory_data()
        output_dir: directory to save plots
        highlight_agents: list of agent IDs to highlight (e.g., [0, 1, 2])
    """
    positions = data['positions']  # (num_steps, num_agents, 2)
    velocities = data['velocities']  # (num_steps, num_agents)
    num_agents = data['num_agents']

    # Convert velocity from km/s to knots for display
    velocities_knots = velocities / 0.514444 / 0.001  # km/s to knots

    # Determine actual spatial extent from data
    x_min, x_max = positions[:, :, 0].min(), positions[:, :, 0].max()
    y_min, y_max = positions[:, :, 1].min(), positions[:, :, 1].max()

    # Add 10% margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    # Velocity range (110-175 knots for AAM aircraft)
    v_min, v_max = 110, 175

    # ===== FIGURE 1: Agent Trajectories =====
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    for agent_id in range(num_agents):
        # Extract this agent's trajectory (exclude last point to avoid artifact)
        traj_x = positions[:-1, agent_id, 0]
        traj_y = positions[:-1, agent_id, 1]
        traj_v = velocities_knots[:-1, agent_id]

        # Create line segments for color mapping
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Determine if this agent should be highlighted
        is_highlighted = highlight_agents and agent_id in highlight_agents

        if is_highlighted:
            lc = LineCollection(segments, cmap='plasma_r', linewidth=1.7, alpha=0.9, zorder=10)
            lc.set_array(traj_v[:-1])
            lc.set_clim(v_min, v_max)
            line1 = ax1.add_collection(lc)
            # Only add label for the first highlighted agent
            if agent_id == highlight_agents[0]:
                ax1.plot(traj_x[0], traj_y[0], 'ks', markersize=4, label='Starting positions of slow agents', zorder=11)
            else:
                ax1.plot(traj_x[0], traj_y[0], 'ks', markersize=4, zorder=11)
        else:
            lc = LineCollection(segments, cmap='plasma_r', linewidth=1.5, alpha=0.6)
            lc.set_array(traj_v[:-1])
            lc.set_clim(v_min, v_max)
            line1 = ax1.add_collection(lc)

    # Add colorbar to left plot
    cbar1 = fig1.colorbar(line1, ax=ax1)
    cbar1.set_label('Velocity (knots)\nPurple: Fast (175) | Yellow: Slow (110)', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    # Draw corridor boundaries on left plot
    if 'corridors' in data and data['corridors'] is not None:
        draw_corridors(ax1, data['corridors'])

    # Set axis limits for left plot
    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
    ax1.set_ylim(max(y_min - y_margin, -5.0), max(y_max + y_margin, 11.0))
    ax1.set_xlabel('X Position (km)', fontsize=16)
    ax1.set_ylabel('Y Position (km)', fontsize=16)
    ax1.set_title(f'Agent Trajectories (N={num_agents})', fontsize=18)
    ax1.tick_params(labelsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    if highlight_agents:
        ax1.legend(fontsize=12, loc='upper center')

    # Save first figure
    os.makedirs(output_dir, exist_ok=True)
    filename_base_left = f'trajectories_left_{num_agents}agents'

    # Save PNG
    filename_png_left = os.path.join(output_dir, f'{filename_base_left}.png')
    fig1.savefig(filename_png_left, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved left trajectory plot to {filename_png_left}")

    # Save PDF
    filename_pdf_left = os.path.join(output_dir, f'{filename_base_left}.pdf')
    fig1.savefig(filename_pdf_left, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved left trajectory plot to {filename_pdf_left}")

    plt.close(fig1)

    # ===== FIGURE 2: Corridor Rectangles Colored by Average Velocity =====
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    # Flatten all trajectory points and velocities (exclude last point)
    all_positions = positions[:-1].reshape(-1, 2)  # (num_steps-1 * num_agents, 2)
    all_velocities = velocities_knots[:-1].flatten()  # (num_steps-1 * num_agents,)

    # Collect patches and velocities (separate lists for corridors and gaps)
    corridor_patches = []
    corridor_velocities = []
    gap_patches = []
    gap_velocities = []
    corridor_labels = []

    if 'corridors' in data and data['corridors'] is not None:
        corridor_data = data['corridors']

        # For each corridor, compute average velocity of points within it
        for i in range(corridor_data['num_corridors']):
            entrance = corridor_data['entrances'][i]
            exit_pt = corridor_data['exits'][i]
            width = corridor_data['widths'][i]
            n_vec = corridor_data['n_vecs'][i]

            half_width = (width + 0.05) / 2.0

            # Four corners of the corridor rectangle
            corners = np.array([
                entrance + half_width * n_vec,
                entrance - half_width * n_vec,
                exit_pt - half_width * n_vec,
                exit_pt + half_width * n_vec
            ])

            # Find points within this corridor rectangle
            # Compute corridor direction and perpendicular
            corridor_dir = exit_pt - entrance
            corridor_length = np.linalg.norm(corridor_dir)
            corridor_dir_norm = corridor_dir / corridor_length

            # Transform points to corridor local coordinate system
            rel_positions = all_positions - entrance[np.newaxis, :]
            along_corridor = np.dot(rel_positions, corridor_dir_norm)
            across_corridor = np.dot(rel_positions, n_vec)

            # Filter points within corridor bounds
            in_corridor = (
                (along_corridor >= 0) &
                (along_corridor <= corridor_length) &
                (np.abs(across_corridor) <= half_width)
            )

            # Compute average velocity for this corridor
            corridor_vels = all_velocities[in_corridor]
            if len(corridor_vels) > 0:
                avg_vel = corridor_vels.mean()
            else:
                avg_vel = (v_min + v_max) / 2  # Default if no points

            # Store patch and velocity for collection
            corridor_patches.append(patches.Polygon(corners, closed=True))
            corridor_velocities.append(avg_vel)

            # Add corridor label
            center = (entrance + exit_pt) / 2
            corridor_labels.append((center, f'C{i}'))

        # Draw gaps between ALL spatially connected corridors (not just consecutive)
        # Track which gaps we've already drawn to avoid duplicates
        drawn_gaps = set()

        # Excluded connections (pairs that should not have gaps drawn)
        excluded_pairs = {(5, 10), (4,11)}

        for i in range(corridor_data['num_corridors']):
            curr_exit = corridor_data['exits'][i]
            curr_width = corridor_data['widths'][i]
            curr_n = corridor_data['n_vecs'][i]

            # Check all other corridors to find connections
            for j in range(corridor_data['num_corridors']):
                if i == j:
                    continue

                next_entrance = corridor_data['entrances'][j]

                # Check if this corridor's exit connects to another corridor's entrance
                dist = np.linalg.norm(curr_exit - next_entrance)

                # Only draw if corridors are spatially close and we haven't drawn this gap yet
                gap_key = tuple(sorted([i, j]))  # Use sorted tuple to avoid duplicates
                if dist < 2.0 and dist > 0.01 and gap_key not in drawn_gaps and gap_key not in excluded_pairs:
                    drawn_gaps.add(gap_key)

                    next_width = corridor_data['widths'][j]
                    next_n = corridor_data['n_vecs'][j]

                    half_width1 = (curr_width + 0.05) / 2.0
                    half_width2 = (next_width + 0.05) / 2.0

                    gap_corners = np.array([
                        curr_exit + half_width1 * curr_n,
                        curr_exit - half_width1 * curr_n,
                        next_entrance - half_width2 * next_n,
                        next_entrance + half_width2 * next_n
                    ])

                    # Find points in this gap region
                    gap_dir = next_entrance - curr_exit
                    gap_length = np.linalg.norm(gap_dir)
                    gap_dir_norm = gap_dir / gap_length if gap_length > 0 else gap_dir

                    # Average perpendicular vector for gap
                    avg_n = (curr_n + next_n) / 2
                    avg_n = avg_n / np.linalg.norm(avg_n)

                    rel_positions = all_positions - curr_exit[np.newaxis, :]
                    along_gap = np.dot(rel_positions, gap_dir_norm)
                    across_gap = np.dot(rel_positions, avg_n)

                    avg_width = (half_width1 + half_width2) / 2
                    in_gap = (
                        (along_gap >= 0) &
                        (along_gap <= gap_length) &
                        (np.abs(across_gap) <= avg_width * 1.5)  # Slightly wider for gaps
                    )

                    gap_vels = all_velocities[in_gap]
                    if len(gap_vels) > 0:
                        gap_avg_vel = gap_vels.mean()
                    else:
                        gap_avg_vel = (v_min + v_max) / 2

                    # Store gap patch and velocity
                    gap_patches.append(patches.Polygon(gap_corners, closed=True))
                    gap_velocities.append(gap_avg_vel)

        # Add approach zones before entry corridors (C0, C4, C6) showing average starting speeds
        # C0 and C6 get 5 zones, C4 gets 2 zones
        entry_corridor_ids = [0, 4, 6]
        zone_length = 1.0  # Each zone is 1 km long

        for corridor_id in entry_corridor_ids:
            if corridor_id < corridor_data['num_corridors']:
                entrance = corridor_data['entrances'][corridor_id]
                width = corridor_data['widths'][corridor_id]
                n_vec = corridor_data['n_vecs'][corridor_id]

                # Compute direction into the corridor
                exit_pt = corridor_data['exits'][corridor_id]
                corridor_dir = exit_pt - entrance
                corridor_dir_norm = corridor_dir / np.linalg.norm(corridor_dir)

                half_width = (width + 0.05) / 2.0

                # Number of approach zones depends on corridor
                num_approach_zones = 5 if corridor_id in [0, 6] else 2

                # Create multiple 1km approach zones
                for zone_idx in range(num_approach_zones):
                    # Each zone starts further back
                    zone_start_dist = (zone_idx + 1) * zone_length
                    zone_end_dist = zone_idx * zone_length

                    zone_start = entrance - zone_start_dist * corridor_dir_norm
                    zone_end = entrance - zone_end_dist * corridor_dir_norm

                    # Four corners of this approach zone rectangle
                    approach_corners = np.array([
                        zone_end + half_width * n_vec,
                        zone_end - half_width * n_vec,
                        zone_start - half_width * n_vec,
                        zone_start + half_width * n_vec
                    ])

                    # Find all trajectory points in this specific zone
                    rel_positions = all_positions - zone_start[np.newaxis, :]
                    along_zone = np.dot(rel_positions, corridor_dir_norm)
                    across_zone = np.dot(rel_positions, n_vec)

                    in_zone = (
                        (along_zone >= 0) &
                        (along_zone <= zone_length) &
                        (np.abs(across_zone) <= half_width)
                    )

                    # Compute average velocity in this approach zone
                    zone_vels = all_velocities[in_zone]
                    if len(zone_vels) > 0:
                        avg_zone_vel = zone_vels.mean()
                    else:
                        avg_zone_vel = (v_min + v_max) / 2

                    # Add approach zone as a gap patch (transparent outline)
                    gap_patches.append(patches.Polygon(approach_corners, closed=True))
                    gap_velocities.append(avg_zone_vel)

        # Create PatchCollection for corridors (solid black edges)
        if len(corridor_patches) > 0:
            pc_corridors = PatchCollection(corridor_patches, cmap='plasma_r', alpha=0.65,
                                          edgecolor='black', linewidth=1.2, zorder=2)
            pc_corridors.set_array(np.array(corridor_velocities))
            pc_corridors.set_clim(v_min, v_max)
            ax2.add_collection(pc_corridors)

        # Create PatchCollection for gaps (dotted, nearly invisible edges)
        if len(gap_patches) > 0:
            pc_gaps = PatchCollection(gap_patches, cmap='plasma_r', alpha=0.65,
                                     edgecolor='lightgray', linewidth=0.3, linestyle=':', zorder=1)
            pc_gaps.set_array(np.array(gap_velocities))
            pc_gaps.set_clim(v_min, v_max)
            ax2.add_collection(pc_gaps)

        # Add corridor labels on top
        for center, label in corridor_labels:
            ax2.text(center[0], center[1], label, fontsize=5, ha='center', va='center',
                    color='white', weight='bold', zorder=3)

    # Set axis limits for right plot (same as left)
    ax2.set_xlim(x_min - x_margin, x_max + x_margin)
    ax2.set_ylim(max(y_min - y_margin, -5.0), max(y_max + y_margin, 11.0))
    ax2.set_xlabel('X Position (km)', fontsize=16)
    ax2.set_ylabel('Y Position (km)', fontsize=16)
    ax2.set_title(f'Heatmap of Average Speed', fontsize=18)
    ax2.tick_params(labelsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Save second figure
    filename_base_right = f'trajectories_right_{num_agents}agents'

    # Save PNG
    filename_png_right = os.path.join(output_dir, f'{filename_base_right}.png')
    fig2.savefig(filename_png_right, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved right heatmap plot to {filename_png_right}")

    # Save PDF
    filename_pdf_right = os.path.join(output_dir, f'{filename_base_right}.pdf')
    fig2.savefig(filename_pdf_right, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved right heatmap plot to {filename_pdf_right}")

    plt.close(fig2)


def plot_velocity_heatmap(data, output_dir, grid_resolution=0.1):
    """
    Create a 2D heatmap showing average velocities in different regions.
    Uses colorblind-friendly colormap: Red=slow (110 knots), Blue=fast (175 knots)

    Args:
        data: dict from load_trajectory_data()
        output_dir: directory to save plots
        grid_resolution: size of grid cells in km
    """
    positions = data['positions']  # (num_steps, num_agents, 2)
    velocities = data['velocities']  # (num_steps, num_agents)

    # Convert velocity to knots
    velocities_knots = velocities / 0.514444 / 0.001

    # Flatten data
    all_x = positions[:, :, 0].flatten()
    all_y = positions[:, :, 1].flatten()
    all_v = velocities_knots.flatten()

    # Determine bounds from actual data
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # Add small margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    # Create grid
    x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)

    # Compute 2D histogram with velocity as values
    velocity_sum, _, _ = np.histogram2d(all_x, all_y, bins=[x_bins, y_bins], weights=all_v)
    count, _, _ = np.histogram2d(all_x, all_y, bins=[x_bins, y_bins])

    # Average velocity in each cell (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_avg = velocity_sum / count
        velocity_avg[count == 0] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use colorblind-friendly colormap: RdYlBu (Red=slow, Blue=fast)
    # Velocity range: 110-175 knots (typical AAM speeds)
    im = ax.imshow(velocity_avg.T, origin='lower',
                   extent=[x_min, x_max, y_min, y_max],
                   cmap='plasma_r', aspect='auto', interpolation='bilinear',
                   vmin=110, vmax=175)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Average Velocity (knots)\nRed: Slow (110) | Blue: Fast (175)', fontsize=12)

    # Draw corridor boundaries (if available)
    if 'corridors' in data and data['corridors'] is not None:
        draw_corridors(ax, data['corridors'])

    ax.set_xlabel('X Position (km)', fontsize=14)
    ax.set_ylabel('Y Position (km)', fontsize=14)
    ax.set_title('Velocity Heatmap (Spatial Distribution)', fontsize=16)
    ax.grid(True, alpha=0.3, color='white')

    plt.tight_layout()

    # Save plot (both PNG and PDF)
    num_agents = data['num_agents']
    filename_base = f'velocity_heatmap_{num_agents}agents'

    # Save PNG
    filename_png = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved velocity heatmap to {filename_png}")

    # Save PDF
    filename_pdf = os.path.join(output_dir, f'{filename_base}.pdf')
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"Saved velocity heatmap to {filename_pdf}")

    plt.close()


def plot_corridor_velocity_heatmap(data, output_dir):
    """
    Plot average velocity per corridor segment.
    Shows which corridor segments have higher/lower average velocities.

    Args:
        data: dict from load_trajectory_data()
        output_dir: directory to save plots
    """
    velocities = data['velocities']  # (num_steps, num_agents)
    tubes = data['tubes']  # (num_steps, num_agents)

    # Convert to knots
    velocities_knots = velocities / 0.514444 / 0.001

    # Get unique corridor indices (excluding -1 which means not in corridor)
    unique_corridors = np.unique(tubes)
    unique_corridors = unique_corridors[unique_corridors >= 0]

    if len(unique_corridors) == 0:
        print("Warning: No corridor data available for corridor velocity heatmap")
        return

    # Calculate average velocity for each corridor
    corridor_velocities = []
    corridor_counts = []

    for corridor_id in unique_corridors:
        # Find all timesteps where any agent is in this corridor
        mask = (tubes == corridor_id)
        corridor_vels = velocities_knots[mask]

        if len(corridor_vels) > 0:
            corridor_velocities.append(corridor_vels.mean())
            corridor_counts.append(len(corridor_vels))
        else:
            corridor_velocities.append(0)
            corridor_counts.append(0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Average velocity per corridor (bar chart)
    colors = plt.cm.RdYlBu([(v - 140) / (175 - 140) for v in corridor_velocities])
    bars = ax1.bar(unique_corridors, corridor_velocities, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Corridor Segment (Tube ID)', fontsize=14)
    ax1.set_ylabel('Average Velocity (knots)', fontsize=14)
    ax1.set_title('Average Velocity per Corridor Segment', fontsize=16)
    ax1.axhline(140, color='red', linestyle='--', alpha=0.5, label='140 knots (slow)')
    ax1.axhline(175, color='blue', linestyle='--', alpha=0.5, label='175 knots (fast)')
    ax1.set_ylim(130, 180)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=10)

    # Plot 2: Sample count per corridor (to show corridor usage)
    ax2.bar(unique_corridors, corridor_counts, color='gray', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Corridor Segment (Tube ID)', fontsize=14)
    ax2.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_title('Corridor Usage (Agent-Timesteps)', fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot (both PNG and PDF)
    num_agents = data['num_agents']
    filename_base = f'corridor_velocity_heatmap_{num_agents}agents'

    # Save PNG
    filename_png = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved corridor velocity heatmap to {filename_png}")

    # Save PDF
    filename_pdf = os.path.join(output_dir, f'{filename_base}.pdf')
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"Saved corridor velocity heatmap to {filename_pdf}")

    plt.close()


def plot_velocity_time_series(data, output_dir, agent_ids=None):
    """
    Plot velocity over time for specific agents.

    Args:
        data: dict from load_trajectory_data()
        output_dir: directory to save plots
        agent_ids: list of agent IDs to plot (default: first 5)
    """
    velocities = data['velocities']  # (num_steps, num_agents)
    dt = data['dt']
    num_steps = data['episode_length']

    # Convert to knots
    velocities_knots = velocities / 0.514444 / 0.001

    # Time array
    time = np.arange(num_steps) * dt

    if agent_ids is None:
        agent_ids = list(range(min(5, data['num_agents'])))

    fig, ax = plt.subplots(figsize=(14, 6))

    for agent_id in agent_ids:
        ax.plot(time, velocities_knots[:, agent_id], label=f'Agent {agent_id}', linewidth=1.5)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Velocity (knots)', fontsize=14)
    ax.set_title('Velocity Time Series', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save plot (both PNG and PDF)
    num_agents = data['num_agents']
    filename_base = f'velocity_timeseries_{num_agents}agents'

    # Save PNG
    filename_png = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved velocity time series to {filename_png}")

    # Save PDF
    filename_pdf = os.path.join(output_dir, f'{filename_base}.pdf')
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"Saved velocity time series to {filename_pdf}")

    plt.close()


def plot_phase_transitions(data, output_dir, agent_ids=None):
    """
    Plot phase transitions over time for specific agents.

    Args:
        data: dict from load_trajectory_data()
        output_dir: directory to save plots
        agent_ids: list of agent IDs to plot
    """
    phases = data['phases']  # (num_steps, num_agents)
    dt = data['dt']
    num_steps = data['episode_length']

    time = np.arange(num_steps) * dt

    if agent_ids is None:
        agent_ids = list(range(min(5, data['num_agents'])))

    fig, ax = plt.subplots(figsize=(14, 6))

    for agent_id in agent_ids:
        ax.plot(time, phases[:, agent_id], label=f'Agent {agent_id}', linewidth=2, marker='o', markersize=2)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Phase', fontsize=14)
    ax.set_title('Corridor Phase Transitions', fontsize=16)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Pre-corridor', 'In-corridor', 'Post-corridor'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save plot (both PNG and PDF)
    num_agents = data['num_agents']
    filename_base = f'phase_transitions_{num_agents}agents'

    # Save PNG
    filename_png = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved phase transitions plot to {filename_png}")

    # Save PDF
    filename_pdf = os.path.join(output_dir, f'{filename_base}.pdf')
    plt.savefig(filename_pdf, bbox_inches='tight')
    print(f"Saved phase transitions plot to {filename_pdf}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize agent trajectories')
    parser.add_argument('--trajectory_file', type=str, required=True,
                        help='Path to trajectory_episode_X.npz file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (default: same as trajectory file)')
    parser.add_argument('--highlight_agents', type=int, nargs='+', default=None,
                        help='Agent IDs to highlight in trajectory plot (e.g., --highlight_agents 0 1 2)')
    parser.add_argument('--plot_agents', type=int, nargs='+', default=None,
                        help='Agent IDs for time series plots (default: first 5)')

    args = parser.parse_args()

    # Load data
    print(f"Loading trajectory data from {args.trajectory_file}...")
    data = load_trajectory_data(args.trajectory_file)

    print(f"Loaded data:")
    print(f"  Number of agents: {data['num_agents']}")
    print(f"  Episode length: {data['episode_length']} steps ({data['episode_length'] * data['dt']:.1f} seconds)")
    print(f"  Timestep: {data['dt']} seconds")

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.trajectory_file)

    print(f"\nGenerating plots...")

    # Generate plots
    plot_all_trajectories(data, args.output_dir, highlight_agents=args.highlight_agents)
    plot_velocity_heatmap(data, args.output_dir)
    plot_corridor_velocity_heatmap(data, args.output_dir)
    plot_velocity_time_series(data, args.output_dir, agent_ids=args.plot_agents)
    plot_phase_transitions(data, args.output_dir, agent_ids=args.plot_agents)

    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
