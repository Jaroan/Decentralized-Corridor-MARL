#!/usr/bin/env python
"""
Plot corridor network with realistic map background.

Uses Boston Harbor or San Francisco Bay Area as background to give
realistic AAM/UTM context with coastlines, metro regions, waterways.

Requirements:
    pip install contextily pillow

Usage:
    python eval_scripts/plot_corridors_with_map.py \
        --trajectory_file eval_heterogeneous_combined_graph/trajectory_40agents_heterogeneous_episode_0.npz \
        --output_file corridor_network_with_map.pdf \
        --location boston  # or 'sanfrancisco'
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.patches import Polygon

# Set font to Times New Roman for publication
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 11

# Try to import contextily for map backgrounds
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Warning: contextily not installed. Install with: pip install contextily")
    print("Will create visualization without map background.")


def meters_to_km(coords_meters):
    """Convert coordinates from meters to km."""
    return coords_meters / 1000.0


def km_to_web_mercator(coords_km, center_lat, center_lon):
    """
    Convert local km coordinates to Web Mercator (EPSG:3857) coordinates.

    Args:
        coords_km: (N, 2) array of coordinates in km
        center_lat: latitude of coordinate system origin
        center_lon: longitude of coordinate system origin

    Returns:
        coords_mercator: (N, 2) array in Web Mercator coordinates
    """
    # Earth radius in km
    R = 6371.0

    # Convert km to lat/lon offsets (approximate, good for small areas)
    lat_offset = (coords_km[:, 1] / R) * (180 / np.pi)
    lon_offset = (coords_km[:, 0] / R) * (180 / np.pi) / np.cos(center_lat * np.pi / 180)

    # Absolute coordinates
    lats = center_lat + lat_offset
    lons = center_lon + lon_offset

    # Convert to Web Mercator (EPSG:3857)
    # x = lon * 20037508.34 / 180
    # y = log(tan((90 + lat) * pi / 360)) * 20037508.34 / pi
    x_mercator = lons * 20037508.34 / 180
    y_mercator = np.log(np.tan((90 + lats) * np.pi / 360)) * 20037508.34 / np.pi

    return np.column_stack([x_mercator, y_mercator])


def plot_corridor_on_map(ax, entrance_merc, exit_merc, width_merc, n_vec, color='blue', alpha=0.8, linewidth=3):
    """Plot a single corridor on map coordinates."""
    # Corridor corners in Web Mercator
    half_width = width_merc / 2
    corner1 = entrance_merc + half_width * n_vec
    corner2 = entrance_merc - half_width * n_vec
    corner3 = exit_merc - half_width * n_vec
    corner4 = exit_merc + half_width * n_vec

    # Draw corridor as thick line (more visible on map)
    ax.plot([entrance_merc[0], exit_merc[0]], [entrance_merc[1], exit_merc[1]],
            color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round', zorder=10)

    # Draw corridor edges
    ax.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]],
            color=color, linewidth=linewidth*0.3, alpha=alpha*0.7, zorder=9)
    ax.plot([corner3[0], corner4[0]], [corner3[1], corner4[1]],
            color=color, linewidth=linewidth*0.3, alpha=alpha*0.7, zorder=9)


def main():
    parser = argparse.ArgumentParser(description='Plot corridor network with map background')
    parser.add_argument('--trajectory_file', type=str, required=True,
                       help='Path to trajectory NPZ file')
    parser.add_argument('--output_file', type=str, default='corridor_network_with_map.pdf',
                       help='Output file path')
    parser.add_argument('--location', type=str, default='boston',
                       choices=['boston', 'sanfrancisco'],
                       help='Map location to use as background')
    parser.add_argument('--map_opacity', type=float, default=0.25,
                       help='Map background opacity (0.2-0.3 recommended)')

    args = parser.parse_args()

    # Load trajectory data
    print(f"Loading data from: {args.trajectory_file}")
    data = np.load(args.trajectory_file, allow_pickle=True)

    corridor_entrances = data['corridor_entrances']  # in km
    corridor_exits = data['corridor_exits']  # in km
    corridor_widths = data['corridor_widths']  # in km
    corridor_n_vecs = data['corridor_n_vecs']
    num_corridors = len(corridor_entrances)

    print(f"Found {num_corridors} corridors")

    # Choose location for map background
    if args.location == 'boston':
        # Boston Harbor - realistic for UTM operations
        center_lat = 42.3601  # Boston Harbor
        center_lon = -71.0589
        zoom_level = 13
        location_name = "Boston Harbor"
    else:  # sanfrancisco
        # San Francisco Bay Area
        center_lat = 37.7749
        center_lon = -122.4194
        zoom_level = 12
        location_name = "San Francisco Bay Area"

    print(f"Using {location_name} as background")

    # Convert corridors to Web Mercator coordinates
    print("Converting corridor coordinates...")
    entrances_merc = km_to_web_mercator(corridor_entrances, center_lat, center_lon)
    exits_merc = km_to_web_mercator(corridor_exits, center_lat, center_lon)

    # Scale corridor widths to mercator (approximate)
    # At this latitude, 1 km ≈ 111.32 km per degree ≈ 20037508.34/360 meters per degree in mercator
    width_scale = 20037508.34 / 360 / 111.32  # meters per km in mercator
    widths_merc = corridor_widths * width_scale * 1000  # convert km to meters in mercator

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Add map background if contextily is available
    if HAS_CONTEXTILY:
        try:
            print(f"Fetching map tiles (zoom level {zoom_level})...")

            # Get bounds for the map
            x_min = np.min(entrances_merc[:, 0]) - 2000
            x_max = np.max(exits_merc[:, 0]) + 2000
            y_min = np.min(entrances_merc[:, 1]) - 2000
            y_max = np.max(exits_merc[:, 1]) + 2000

            # Fetch map tiles
            # Use CartoDB Positron for a clean grayscale look
            ctx.add_basemap(
                ax,
                crs='EPSG:3857',
                source=ctx.providers.CartoDB.PositronNoLabels,
                zoom=zoom_level,
                alpha=args.map_opacity,
                attribution=False
            )

            print("Map background added successfully")

        except Exception as e:
            print(f"Warning: Could not fetch map background: {e}")
            print("Continuing without map...")
            ax.set_facecolor('#E8E8E8')
    else:
        # Fallback: simple gray background
        ax.set_facecolor('#E8E8E8')

    # Define corridor colors - strong, vibrant colors for visibility on map
    corridor_colors_strong = [
        '#FF0000',  # 0: Bright red
        '#FF4500',  # 1: Orange red
        '#FF69B4',  # 2: Hot pink
        '#FF1493',  # 3: Deep pink
        '#00CED1',  # 4: Dark turquoise
        '#FFD700',  # 5: Gold (core corridor)
        '#1E90FF',  # 6: Dodger blue
        '#32CD32',  # 7: Lime green
        '#FF6347',  # 8: Tomato
        '#8A2BE2',  # 9: Blue violet
        '#DC143C',  # 10: Crimson
        '#00FA9A',  # 11: Medium spring green
        '#FFA500',  # 12: Orange (core)
        '#FF8C00',  # 13: Dark orange (core)
        '#00FF00',  # 14: Lime (split)
        '#7FFF00',  # 15: Chartreuse (split)
        '#00FF7F',  # 16: Spring green (exit)
        '#00FFFF',  # 17: Cyan (exit)
    ]

    # Plot all corridors with strong colors
    print("Plotting corridors...")
    for i in range(num_corridors):
        color = corridor_colors_strong[i]
        plot_corridor_on_map(ax, entrances_merc[i], exits_merc[i],
                            widths_merc[i], corridor_n_vecs[i],
                            color=color, alpha=0.85, linewidth=6)

    # Add scale bar (approximate)
    scale_bar_km = 2  # 2 km scale bar
    scale_bar_merc = scale_bar_km * width_scale * 1000
    x_pos = np.min(entrances_merc[:, 0]) + 1000
    y_pos = np.min(entrances_merc[:, 1]) + 500
    ax.plot([x_pos, x_pos + scale_bar_merc], [y_pos, y_pos],
            'k-', linewidth=4, solid_capstyle='butt', zorder=20)
    ax.text(x_pos + scale_bar_merc/2, y_pos - 300, f'{scale_bar_km} km',
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))

    # Title and labels
    ax.set_title(f'AAM Corridor Network over {location_name}\nSimple Combined Graph Scenario',
                fontsize=18, fontweight='bold', pad=20)

    # Remove axis labels (map coordinates are not meaningful)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add north arrow
    x_arrow = np.max(exits_merc[:, 0]) - 1000
    y_arrow = np.max(exits_merc[:, 1]) - 1000
    ax.annotate('N', xy=(x_arrow, y_arrow + 800), xytext=(x_arrow, y_arrow),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                fontsize=16, fontweight='bold', ha='center', zorder=20,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

    # Add legend
    ax.text(0.02, 0.98, f'{num_corridors} Corridors\n4 Entry Points\n2 Exit Terminals\n8 Routes',
            transform=ax.transAxes, fontsize=13, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))

    # Add attribution
    ax.text(0.98, 0.02, f'Map: {location_name}\nBase map © OpenStreetMap contributors',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save figure
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved corridor network with map to: {args.output_file}")

    # Also save as PNG
    png_file = args.output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Also saved as PNG: {png_file}")

    plt.close()
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
