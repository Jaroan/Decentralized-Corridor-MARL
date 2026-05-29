#!/usr/bin/env python
"""
Render a trajectory NPZ file as an animated GIF with speed-colored
trail history.

Each frame shows:
  - Static corridor network
  - Current agent positions as oriented triangles, colored by speed
  - Faded trail of past N timesteps colored by speed (plasma_r colormap,
    110-175 knots range)

Usage:
    python eval_scripts/render_trajectory_gif.py \
        --trajectory_file eval_combined_graph_40agents/trajectory_40agents_homogeneous_episode_0.npz

    # Customize trail length, fps, and speed range
    python eval_scripts/render_trajectory_gif.py \
        --trajectory_file ... --trail_length 10 --fps 15 \
        --vmin 100 --vmax 175
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import imageio.v2 as imageio
from io import BytesIO
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

KMS_TO_KNOTS = 1.0 / (0.514444 * 0.001)   # km/s → knots


def draw_corridors(ax, entrances, exits, widths, n_vecs):
    for i in range(len(entrances)):
        ent = entrances[i]
        ext = exits[i]
        hw = widths[i] / 2
        n = n_vecs[i]
        corners = [
            ent + hw * n, ent - hw * n,
            ext - hw * n, ext + hw * n,
        ]
        poly = patches.Polygon(
            corners, closed=True,
            facecolor='#F0F0F0', edgecolor='#888888',
            linewidth=1.0, linestyle='--', zorder=1,
        )
        ax.add_patch(poly)


def get_axis_bounds(entrances, exits, widths, n_vecs, pad=1.8):
    pts = []
    for i in range(len(entrances)):
        hw = widths[i] / 2
        n = n_vecs[i]
        pts.append(entrances[i] + hw * n)
        pts.append(entrances[i] - hw * n)
        pts.append(exits[i] + hw * n)
        pts.append(exits[i] - hw * n)
    all_pts = np.array(pts)
    return (
        all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad,
        all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad,
    )


def render_frame(t, positions, headings, speeds_knots, exit_times,
                 corridor_data, trail_length, bounds, vmin, vmax,
                 title='', show_colorbar=True):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=80)

    entrances, exits_pos, widths, n_vecs = corridor_data
    draw_corridors(ax, entrances, exits_pos, widths, n_vecs)

    num_agents = positions.shape[1]
    cmap = plt.get_cmap('plasma_r')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Collect all trail segments across all active agents
    all_segments = []
    all_seg_colors = []

    triangle_patches = []

    for a in range(num_agents):
        exit_t = exit_times[a]
        if 0 <= exit_t < t:
            continue

        trail_start = max(0, t - trail_length)
        trail_pts = positions[trail_start:t + 1, a, :]
        trail_v = speeds_knots[trail_start:t + 1, a]
        n_pts = len(trail_pts)

        if n_pts >= 2:
            for j in range(n_pts - 1):
                # newest segment closest to present, fade older ones
                age = (n_pts - 1 - j) / max(1, n_pts - 1)
                alpha = 0.10 + 0.55 * (1 - age)
                seg_speed = 0.5 * (trail_v[j] + trail_v[j + 1])
                rgba = list(cmap(norm(seg_speed)))
                rgba[3] = alpha
                all_segments.append([trail_pts[j], trail_pts[j + 1]])
                all_seg_colors.append(rgba)

        # Current triangle, colored by current speed
        cx, cy = positions[t, a]
        heading = headings[t, a]
        cur_speed = speeds_knots[t, a]
        tri_color = cmap(norm(cur_speed))

        tri_size = 0.13
        tri = np.array([
            [tri_size, 0.0],
            [-tri_size * 0.6, tri_size * 0.55],
            [-tri_size * 0.6, -tri_size * 0.55],
        ])
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ])
        tri_rot = tri @ rot.T + np.array([cx, cy])
        triangle_patches.append((tri_rot, tri_color))

    if all_segments:
        lc = LineCollection(
            np.array(all_segments), colors=all_seg_colors,
            linewidths=2.4, capstyle='round', zorder=4,
        )
        ax.add_collection(lc)

    for tri_rot, tri_color in triangle_patches:
        ax.add_patch(patches.Polygon(
            tri_rot, closed=True,
            facecolor=tri_color, edgecolor='black',
            linewidth=0.7, zorder=6,
        ))

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.set_facecolor('white')
    ax.set_title(f'{title}  |  t = {t}', fontsize=13)
    ax.tick_params(labelsize=9)

    if show_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('Velocity (knots)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert('RGB'))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--trail_length', type=int, default=8)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--vmin', type=float, default=110.0,
                        help='Min speed for colormap (knots, default: 110)')
    parser.add_argument('--vmax', type=float, default=175.0,
                        help='Max speed for colormap (knots, default: 175)')
    parser.add_argument('--no_colorbar', action='store_true')
    args = parser.parse_args()

    print(f"Loading: {args.trajectory_file}")
    data = np.load(args.trajectory_file, allow_pickle=True)
    positions = data['positions']
    headings = data['headings']
    velocities = data['velocities']           # km/s, scalar magnitude
    speeds_knots = velocities * KMS_TO_KNOTS
    exit_times = data['agent_exit_times']
    entrances = data['corridor_entrances']
    exits_pos = data['corridor_exits']
    widths = data['corridor_widths']
    n_vecs = data['corridor_n_vecs']

    num_timesteps, num_agents, _ = positions.shape
    print(f"Timesteps: {num_timesteps}, Agents: {num_agents}, "
          f"Corridors: {len(entrances)}")
    print(f"Speed range observed: "
          f"{speeds_knots[speeds_knots > 0].min():.1f} – "
          f"{speeds_knots.max():.1f} knots")

    if args.output_file is None:
        parent = os.path.basename(
            os.path.dirname(os.path.abspath(args.trajectory_file)))
        scenario = parent
        if scenario.startswith('eval_'):
            scenario = scenario[len('eval_'):]
        parts = scenario.rsplit('_', 1)
        if len(parts) == 2 and parts[1].endswith('agents'):
            scenario = parts[0]
        args.output_file = f"{scenario}_{num_agents}agents.gif"

    if args.title is None:
        args.title = os.path.splitext(os.path.basename(
            args.trajectory_file))[0]

    bounds = get_axis_bounds(entrances, exits_pos, widths, n_vecs, pad=1.8)
    corridor_data = (entrances, exits_pos, widths, n_vecs)

    last_active_t = num_timesteps - 1
    valid_exits = exit_times[(exit_times >= 0) & (exit_times < num_timesteps)]
    if len(valid_exits) > 0:
        last_active_t = min(num_timesteps - 1,
                            int(valid_exits.max()) + args.trail_length + 2)

    frames = []
    t_values = list(range(0, last_active_t + 1, args.frame_skip))
    print(f"Rendering {len(t_values)} frames (up to t={last_active_t})...")
    for idx, t in enumerate(t_values):
        frame = render_frame(
            t, positions, headings, speeds_knots, exit_times,
            corridor_data, args.trail_length, bounds,
            args.vmin, args.vmax,
            title=args.title, show_colorbar=not args.no_colorbar,
        )
        frames.append(frame)
        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{len(t_values)} frames")

    print(f"Writing GIF: {args.output_file}")
    duration_ms = int(1000.0 / args.fps)
    imageio.mimsave(args.output_file, frames, duration=duration_ms, loop=0)
    size_mb = os.path.getsize(args.output_file) / 1e6
    print(f"Done. Size: {size_mb:.1f} MB, frames: {len(frames)}")


if __name__ == '__main__':
    main()
