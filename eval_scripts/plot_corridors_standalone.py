#!/usr/bin/env python
"""
Plot simple_combined_graph corridor network (no trajectory file needed).

Usage:
    # plain white background (labeled)
    python eval_scripts/plot_corridors_standalone.py

    # plain white background (no labels)
    python eval_scripts/plot_corridors_standalone.py --no_labels

    # city map background (requires: pip install contextily)
    python eval_scripts/plot_corridors_standalone.py --no_labels --map
    python eval_scripts/plot_corridors_standalone.py --no_labels --map --location sanfrancisco
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from PIL import Image

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 16

try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False


# ============================================================
# Corridor geometry (from simple_combined_graph.py make_world)
# ============================================================

def build_corridor_geometry():
    tw = 0.3
    gap = 0.2

    def _dir(a, b):
        d = b - a
        return d / np.linalg.norm(d)

    def _gapped(pt_from, pt_to, gap_start=True, gap_end=True):
        dn = _dir(pt_from, pt_to)
        ent = pt_from + gap * dn if gap_start else pt_from.copy()
        ext = pt_to - gap * dn if gap_end else pt_to.copy()
        d = ext - ent
        L = float(np.linalg.norm(d))
        e_vec = d / L
        n_vec = np.array([-e_vec[1], e_vec[0]], dtype=np.float32)
        return {'entrance': ent, 'exit': ext, 'width': tw, 'n': n_vec}

    WA = [
        np.array([-3.5, 6.0]),
        np.array([-3.2, 4.5]),
        np.array([-3.8, 2.8]),
    ]
    WA_mid = np.array([-2.0, 2.0])
    WB_entry = np.array([-0.5, 4.5])
    J_MA = np.array([-1.0, 1.8])
    J_MB = np.array([0.5, 0.5])

    WD_entry = np.array([2.2, 5.0])
    _dax = J_MB - WD_entry
    _da_dir = _dax / float(np.linalg.norm(_dax))

    stem_len = 1.3
    split_half = np.pi / 7
    split_bl = 1.2

    WD_split = WD_entry + stem_len * _da_dir
    _c, _s = np.cos(split_half), np.sin(split_half)
    _left_dir = np.array([
        _da_dir[0] * _c - _da_dir[1] * _s,
        _da_dir[0] * _s + _da_dir[1] * _c,
    ])
    _right_dir = np.array([
        _da_dir[0] * _c + _da_dir[1] * _s,
        -_da_dir[0] * _s + _da_dir[1] * _c,
    ])
    WD_left = WD_split + split_bl * _left_dir
    WD_right = WD_split + split_bl * _right_dir
    WD_merge = WD_entry + (
        stem_len + 2.0 * split_bl * np.cos(split_half)
    ) * _da_dir

    W12_end = np.array([-0.3, -1.0])
    W13_split = np.array([-0.3, -2.5])
    W14_end = np.array([-1.5, -3.5])
    W15_end = np.array([1.0, -3.5])
    W16_exit = np.array([-1.5, -4.5])
    W17_exit = np.array([1.0, -4.5])

    return [
        _gapped(WA[0], WA[1], gap_start=False),        # C0
        _gapped(WA[1], WA[2]),                          # C1
        _gapped(WA[2], WA_mid),                         # C2
        _gapped(WA_mid, J_MA),                          # C3
        _gapped(WB_entry, J_MA, gap_start=False),       # C4
        _gapped(J_MA, J_MB),                            # C5
        _gapped(WD_entry, WD_split, gap_start=False),   # C6
        _gapped(WD_split, WD_left),                     # C7
        _gapped(WD_split, WD_right),                    # C8
        _gapped(WD_left, WD_merge),                     # C9
        _gapped(WD_right, WD_merge),                    # C10
        _gapped(WD_merge, J_MB),                        # C11
        _gapped(J_MB, W12_end),                         # C12
        _gapped(W12_end, W13_split),                    # C13
        _gapped(W13_split, W14_end),                    # C14
        _gapped(W13_split, W15_end),                    # C15
        _gapped(W14_end, W16_exit, gap_end=False),      # C16
        _gapped(W15_end, W17_exit, gap_end=False),      # C17
    ]


# Strong colors per corridor group (for map overlay)
CORRIDOR_COLORS = {
    # Entry A path
    0: '#E63946', 1: '#E63946', 2: '#E63946', 3: '#E63946',
    # Entry B
    4: '#457B9D',
    # A+B merge feeder
    5: '#F4A261',
    # Entry C stem + diamond
    6: '#2D6A4F', 7: '#2D6A4F', 8: '#2D6A4F',
    9: '#52B788', 10: '#52B788',
    # Diamond exit feeder
    11: '#40916C',
    # Core corridor
    12: '#FFB703', 13: '#FFB703',
    # Split branches
    14: '#8338EC', 15: '#9B5DE5',
    # Final exits
    16: '#9B5DE5', 17: '#8338EC',
}


# ============================================================
# Coordinate conversion: local km → Web Mercator (EPSG:3857)
# ============================================================

def km_to_mercator(coords_km, center_lat, center_lon):
    """Convert (N, 2) array of local km coords to Web Mercator."""
    R = 6371.0
    lat_off = (coords_km[:, 1] / R) * (180.0 / np.pi)
    lon_off = ((coords_km[:, 0] / R) * (180.0 / np.pi)
               / np.cos(np.radians(center_lat)))
    lats = center_lat + lat_off
    lons = center_lon + lon_off
    x = lons * 20037508.34 / 180.0
    y = (np.log(np.tan(np.radians(90.0 + lats) / 2.0))
         * 20037508.34 / np.pi)
    return np.column_stack([x, y])


def pt_to_merc(pt_km, center_lat, center_lon):
    return km_to_mercator(pt_km[np.newaxis, :], center_lat, center_lon)[0]


def scale_width_to_merc(width_km, center_lat):
    """Approximate: 1 km → metres in Web Mercator at given latitude."""
    merc_per_deg_lon = 20037508.34 / 180.0
    km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))
    return width_km * merc_per_deg_lon / km_per_deg_lon


# ============================================================
# Plotting helpers
# ============================================================

def plot_corridor_plain(ax, tube, tube_id, show_label):
    entrance = tube['entrance']
    exit_pos = tube['exit']
    hw = tube['width'] / 2
    n_vec = tube['n']
    corners = [
        entrance + hw * n_vec,
        entrance - hw * n_vec,
        exit_pos - hw * n_vec,
        exit_pos + hw * n_vec,
    ]
    poly = patches.Polygon(
        corners, closed=True,
        facecolor='white', edgecolor='#999999',
        linewidth=1.0, linestyle='dotted',
    )
    ax.add_patch(poly)
    if show_label:
        mid = (entrance + exit_pos) / 2
        ax.text(mid[0], mid[1], f'C{tube_id}',
                fontsize=11, fontweight='bold',
                ha='center', va='center', color='#444444')


def plot_corridor_map(ax, tube, tube_id, show_label,
                      center_lat, center_lon):
    ent_km = tube['entrance']
    ext_km = tube['exit']
    n_vec = tube['n']
    width_km = tube['width']

    ent_m = pt_to_merc(ent_km, center_lat, center_lon)
    ext_m = pt_to_merc(ext_km, center_lat, center_lon)

    # Scale the normal vector to mercator width
    hw_m = scale_width_to_merc(width_km / 2, center_lat)

    # n_vec direction in mercator (approximate: just scale uniformly)
    n_m = np.array([n_vec[0], n_vec[1]])
    n_m = n_m / np.linalg.norm(n_m)

    corners = [
        ent_m + hw_m * n_m,
        ent_m - hw_m * n_m,
        ext_m - hw_m * n_m,
        ext_m + hw_m * n_m,
    ]
    color = CORRIDOR_COLORS.get(tube_id, '#888888')
    poly = patches.Polygon(
        corners, closed=True,
        facecolor=color, edgecolor='white',
        linewidth=0.6, alpha=0.85, zorder=10,
    )
    ax.add_patch(poly)

    if show_label:
        mid = (ent_m + ext_m) / 2
        ax.text(mid[0], mid[1], f'C{tube_id}',
                fontsize=9, fontweight='bold',
                ha='center', va='center', color='white', zorder=11)


# ============================================================
# Main
# ============================================================

LOCATIONS = {
    'boston': {
        'lat': 42.3601, 'lon': -71.0589,
        'zoom': 13, 'name': 'Boston Harbor',
    },
    'sanfrancisco': {
        'lat': 37.7749, 'lon': -122.4194,
        'zoom': 12, 'name': 'San Francisco Bay',
    },
}

ENTRY_COLORS = {
    'Entry A': '#D62728',
    'Entry B': '#1F77B4',
    'Entry C': '#2c7000',
}
EXIT_COLORS = {
    'Exit L': '#9467BD',
    'Exit R': '#8C564B',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str,
                        default='corridor_network.pdf')
    parser.add_argument('--no_labels', action='store_true',
                        help='Omit corridor ID labels (C0, C1, ...)')
    parser.add_argument('--map', action='store_true',
                        help='Overlay on city map (requires contextily)')
    parser.add_argument('--map_file', type=str, default=None,
                        help='Local image file to use as map background')
    parser.add_argument('--location', type=str, default='boston',
                        choices=list(LOCATIONS.keys()),
                        help='Map location (default: boston)')
    parser.add_argument('--map_opacity', type=float, default=0.25,
                        help='Map background opacity (default: 0.25)')
    args = parser.parse_args()

    tubes = build_corridor_geometry()
    print(f"Built {len(tubes)} corridors")

    show_label = not args.no_labels

    if args.map:
        if not HAS_CTX:
            print("ERROR: contextily not installed. Run: pip install contextily")
            return

        loc = LOCATIONS[args.location]
        clat, clon = loc['lat'], loc['lon']
        zoom = loc['zoom']
        loc_name = loc['name']

        fig, ax = plt.subplots(figsize=(8, 10))

        # Collect all corridor points to set map extent
        all_pts_km = []
        for t in tubes:
            all_pts_km.append(t['entrance'])
            all_pts_km.append(t['exit'])
        all_pts_km = np.array(all_pts_km)
        all_pts_m = km_to_mercator(all_pts_km, clat, clon)

        pad = 1500  # metres padding around corridors
        x_min, x_max = all_pts_m[:, 0].min() - pad, all_pts_m[:, 0].max() + pad
        y_min, y_max = all_pts_m[:, 1].min() - pad, all_pts_m[:, 1].max() + pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Fetch grayscale basemap
        print(f"Fetching map tiles ({loc_name}, zoom={zoom})...")
        try:
            ctx.add_basemap(
                ax, crs='EPSG:3857',
                source=ctx.providers.CartoDB.PositronNoLabels,
                zoom=zoom, alpha=args.map_opacity, attribution=False,
            )
            print("Map background added.")
        except Exception as e:
            print(f"Warning: could not fetch map: {e}")

        # Draw corridors
        for i, tube in enumerate(tubes):
            plot_corridor_map(ax, tube, i, show_label, clat, clon)

        # Entry / exit markers in mercator
        entry_tube_ids = {'Entry A': 0, 'Entry B': 4, 'Entry C': 6}
        exit_tube_ids = {'Exit L': 16, 'Exit R': 17}

        for label, tid in entry_tube_ids.items():
            pos_m = pt_to_merc(tubes[tid]['entrance'], clat, clon)
            color = ENTRY_COLORS[label]
            ax.plot(pos_m[0], pos_m[1], 'o', color=color,
                    markersize=10, zorder=15,
                    markeredgecolor='white', markeredgewidth=1.0)
            ax.text(pos_m[0], pos_m[1] + 400, label,
                    fontsize=28, fontweight='bold',
                    ha='center', color=color, zorder=15)

        for label, tid in exit_tube_ids.items():
            pos_m = pt_to_merc(tubes[tid]['exit'], clat, clon)
            color = EXIT_COLORS[label]
            ax.plot(pos_m[0], pos_m[1], 's', color=color,
                    markersize=10, zorder=15,
                    markeredgecolor='white', markeredgewidth=1.0)
            ax.text(pos_m[0], pos_m[1] - 500, label,
                    fontsize=28, fontweight='bold',
                    ha='center', color=color, zorder=15)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f'Combined Scenario over {loc_name}\n'
            '18 Corridors · 3 Entries · 2 Exits · 8 Routes',
            fontsize=15, pad=10,
        )
        ax.text(0.99, 0.01,
                'Map © OpenStreetMap contributors / CartoDB',
                transform=ax.transAxes, fontsize=8,
                ha='right', va='bottom', color='gray')

    elif args.map_file:
        img = Image.open(args.map_file).convert('L')   # grayscale
        img_arr = np.array(img)
        img_h, img_w = img_arr.shape

        # Tight corridor bounding box + small padding
        all_pts = np.array(
            [t['entrance'] for t in tubes] + [t['exit'] for t in tubes]
        )
        x_pad = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.10
        y_pad = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.10
        x_min = all_pts[:, 0].min() - x_pad
        x_max = all_pts[:, 0].max() + x_pad
        y_min = all_pts[:, 1].min() - y_pad
        y_max = all_pts[:, 1].max() + y_pad
        x_span = x_max - x_min
        y_span = y_max - y_min
        plot_aspect = x_span / y_span   # width/height of corridor region

        # Center-crop the image to the same aspect ratio as the corridor
        # bounds, so the image fills the area without any distortion
        if plot_aspect > img_w / img_h:
            # corridor region is wider → crop image height
            new_h = int(img_w / plot_aspect)
            top = (img_h - new_h) // 2
            img_arr = img_arr[top:top + new_h, :]
        else:
            # corridor region is taller → crop image width
            new_w = int(img_h * plot_aspect)
            left = (img_w - new_w) // 2
            img_arr = img_arr[:, left:left + new_w]

        # Figure dimensions match corridor aspect ratio
        fig_h = 9.0
        fig_w = fig_h * plot_aspect
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ax.imshow(
            img_arr, cmap='gray', vmin=0, vmax=255,
            extent=[x_min, x_max, y_min, y_max],
            aspect='equal', alpha=0.45, zorder=0,
        )

        for i, tube in enumerate(tubes):
            plot_corridor_plain(ax, tube, i, show_label)

        entry_pts_local = {
            'Entry A': (tubes[0]['entrance'], ENTRY_COLORS['Entry A']),
            'Entry B': (tubes[4]['entrance'], ENTRY_COLORS['Entry B']),
            'Entry C': (tubes[6]['entrance'], ENTRY_COLORS['Entry C']),
        }
        exit_pts_local = {
            'Exit L': (tubes[16]['exit'], EXIT_COLORS['Exit L']),
            'Exit R': (tubes[17]['exit'], EXIT_COLORS['Exit R']),
        }

        for label, (pos, color) in entry_pts_local.items():
            ax.plot(pos[0], pos[1], 'o', color=color,
                    markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.8)
            ax.text(pos[0], pos[1] + 0.38, label,
                    fontsize=20, fontweight='bold',
                    ha='center', color=color)

        for label, (pos, color) in exit_pts_local.items():
            ax.plot(pos[0], pos[1], 's', color=color,
                    markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.8)
            ax.text(pos[0], pos[1] - 0.44, label,
                    fontsize=20, fontweight='bold',
                    ha='center', color=color)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')   # no axes, ticks, labels, or title

    else:
        fig, ax = plt.subplots(figsize=(7, 9))

        for i, tube in enumerate(tubes):
            plot_corridor_plain(ax, tube, i, show_label)

        entry_pts = {
            'Entry A': (tubes[0]['entrance'], ENTRY_COLORS['Entry A']),
            'Entry B': (tubes[4]['entrance'], ENTRY_COLORS['Entry B']),
            'Entry C': (tubes[6]['entrance'], ENTRY_COLORS['Entry C']),
        }
        exit_pts = {
            'Exit L': (tubes[16]['exit'], EXIT_COLORS['Exit L']),
            'Exit R': (tubes[17]['exit'], EXIT_COLORS['Exit R']),
        }

        for label, (pos, color) in entry_pts.items():
            ax.plot(pos[0], pos[1], 'o', color=color,
                    markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.8)
            ax.text(pos[0], pos[1] + 0.38, label,
                    fontsize=14, fontweight='bold',
                    ha='center', color=color)

        for label, (pos, color) in exit_pts.items():
            ax.plot(pos[0], pos[1], 's', color=color,
                    markersize=9, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.8)
            ax.text(pos[0], pos[1] - 0.44, label,
                    fontsize=16, fontweight='bold',
                    ha='center', color=color)

        ax.set_aspect('equal')
        ax.set_xlabel('X (km)', fontsize=16)
        ax.set_ylabel('Y (km)', fontsize=16)
        ax.set_title(
            'Combined Scenario\n'
            '18 Corridors · 3 Entries · 2 Exits · 8 Routes',
            fontsize=16, pad=12,
        )
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_facecolor('white')

    if not args.map_file:
        plt.tight_layout()

    plt.savefig(args.output_file, dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Saved: {args.output_file}")

    png_file = args.output_file.replace('.pdf', '.png')
    if png_file != args.output_file:
        plt.savefig(png_file, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved: {png_file}")

    plt.close()


if __name__ == '__main__':
    main()
