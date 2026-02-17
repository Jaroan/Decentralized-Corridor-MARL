#!/usr/bin/env python
"""
Plot labeled corridor network for simple_combined_graph.

Shows all 18 corridors with:
- Corridor IDs
- Entry/exit points marked
- Corridor widths visualized
- Route labels

Usage:
    python eval_scripts/plot_corridors_labeled.py \
        --trajectory_file eval_heterogeneous_combined_graph/trajectory_40agents_heterogeneous_episode_0.npz \
        --output_file corridor_network_labeled.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.patches import FancyArrowPatch

# Set font to Times New Roman for publication
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 12


def plot_corridor(ax, entrance, exit_pos, width, n_vec, corridor_id, color='blue', alpha=0.6):
    """Plot a single corridor as a rectangle."""
    # Direction vector
    direction = exit_pos - entrance
    length = np.linalg.norm(direction)

    # Corridor corners
    half_width = width / 2
    corner1 = entrance + half_width * n_vec
    corner2 = entrance - half_width * n_vec
    corner3 = exit_pos - half_width * n_vec
    corner4 = exit_pos + half_width * n_vec

    # Draw corridor as polygon
    corridor_poly = patches.Polygon(
        [corner1, corner2, corner3, corner4],
        closed=True,
        facecolor=color,
        edgecolor='black',
        alpha=alpha,
        linewidth=1.5
    )
    ax.add_patch(corridor_poly)

    # Add corridor ID label at midpoint
    midpoint = (entrance + exit_pos) / 2
    ax.text(midpoint[0], midpoint[1], f'T{corridor_id}',
            fontsize=10, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))

    return corridor_poly


def plot_entry_exit_markers(ax, entrance, exit_pos, corridor_id):
    """Mark entry and exit points of a corridor."""
    # Entry marker (circle)
    ax.plot(entrance[0], entrance[1], 'go', markersize=8, markeredgecolor='darkgreen', markeredgewidth=1.5)

    # Exit marker (square)
    ax.plot(exit_pos[0], exit_pos[1], 'rs', markersize=8, markeredgecolor='darkred', markeredgewidth=1.5)


def main():
    parser = argparse.ArgumentParser(description='Plot labeled corridor network')
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='Path to trajectory NPZ file')
    parser.add_argument('--output_file', type=str, default='corridor_network_labeled.pdf',
                       help='Output file path')

    args = parser.parse_args()

    # Load trajectory data
    print(f"Loading data from: {args.trajectory_file}")
    data = np.load(args.trajectory_file, allow_pickle=True)

    corridor_entrances = data['corridor_entrances']
    corridor_exits = data['corridor_exits']
    corridor_widths = data['corridor_widths']
    corridor_n_vecs = data['corridor_n_vecs']
    num_corridors = len(corridor_entrances)

    print(f"Found {num_corridors} corridors")

    # Define corridor groups and colors
    # Corridors are grouped by function: entry, merge, split, exit
    corridor_colors = {
        'entry_A': ([0, 1], '#FF6B6B'),      # Red - Entry from A
        'entry_B': ([4], '#4ECDC4'),          # Teal - Entry from B
        'entry_CL': ([6, 7], '#95E1D3'),     # Light teal - Entry from CL
        'entry_CR': ([6, 8], '#F38181'),     # Pink - Entry from CR (shares 6)
        'merge': ([2, 3, 9, 10, 11], '#FFA07A'),  # Light coral - Merge zones
        'core': ([5, 12, 13], '#FFD93D'),    # Yellow - Core corridor
        'split': ([14, 15], '#A8E6CF'),      # Light green - Split zones
        'exit': ([16, 17], '#6BCB77'),       # Green - Exit corridors
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot all corridors
    for i in range(num_corridors):
        # Determine color based on corridor group
        color = '#CCCCCC'  # Default gray
        for group_name, (corridor_ids, group_color) in corridor_colors.items():
            if i in corridor_ids:
                color = group_color
                break

        plot_corridor(ax, corridor_entrances[i], corridor_exits[i],
                     corridor_widths[i], corridor_n_vecs[i], i, color=color)

    # Add entry/exit markers for selected corridors
    for i in range(num_corridors):
        plot_entry_exit_markers(ax, corridor_entrances[i], corridor_exits[i], i)

    # Add legend for entry points
    entry_points = {
        'A': corridor_entrances[0],
        'B': corridor_entrances[4],
        'CL': corridor_entrances[6],
        'CR': corridor_entrances[8]
    }

    for label, pos in entry_points.items():
        ax.text(pos[0], pos[1] + 0.5, f'Entry {label}',
               fontsize=14, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=2))

    # Add exit labels
    exit_labels = {
        'T16 (Left)': corridor_exits[16],
        'T17 (Right)': corridor_exits[17]
    }

    for label, pos in exit_labels.items():
        ax.text(pos[0], pos[1] - 0.5, label,
               fontsize=14, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))

    # Add route annotations
    ax.text(-5, 7, '8 Routes:\nA→T16(L), A→T17(R)\nB→T16(L), B→T17(R)\nCL→T16(L), CL→T17(R)\nCR→T16(L), CR→T17(R)',
           fontsize=11, bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.95, edgecolor='black', linewidth=1.5),
           verticalalignment='top')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Entry Point', markeredgecolor='darkgreen', markeredgewidth=1.5),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=10, label='Exit Point', markeredgecolor='darkred', markeredgewidth=1.5),
        patches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Entry Corridors'),
        patches.Patch(facecolor='#FFA07A', edgecolor='black', label='Merge Zones'),
        patches.Patch(facecolor='#FFD93D', edgecolor='black', label='Core Corridor'),
        patches.Patch(facecolor='#A8E6CF', edgecolor='black', label='Split Zones'),
        patches.Patch(facecolor='#6BCB77', edgecolor='black', label='Exit Corridors'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (km)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (km)', fontsize=14, fontweight='bold')
    ax.set_title('Simple Combined Graph Corridor Network\n(18 Corridors, 4 Entry Points, 2 Exit Corridors, 8 Routes)',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#F5F5F5')

    # Save figure
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved labeled corridor network to: {args.output_file}")

    # Also save as PNG
    png_file = args.output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Also saved as PNG: {png_file}")

    plt.close()


if __name__ == '__main__':
    main()
