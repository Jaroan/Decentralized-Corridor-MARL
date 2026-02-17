#!/usr/bin/env python
"""
Evaluate heterogeneous speed scenarios with mixed fast/slow agents.

This script modifies agent max speeds AFTER environment initialization
to test how fast agents adapt to slow agents.

Example usage:
    # 10 agents, agents 0-2 are slow (110 knots), rest are fast (175 knots)
    python eval_scripts/eval_heterogeneous_speeds.py \
        --num_agents 10 \
        --slow_agent_ids 0 1 2 \
        --slow_speed 110 \
        --fast_speed 175 \
        --save_trajectories

    # 40 agents, 25% slow
    python eval_scripts/eval_heterogeneous_speeds.py \
        --num_agents 40 \
        --slow_percentage 25 \
        --slow_speed 140 \
        --fast_speed 175 \
        --save_trajectories
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_heterogeneous_config(num_agents, slow_agent_ids=None, slow_percentage=None,
                                 slow_speed=110, fast_speed=175):
    """
    Create agent speed configuration.

    Args:
        num_agents: total number of agents
        slow_agent_ids: list of agent IDs to be slow (or None to use percentage)
        slow_percentage: percentage of agents to be slow (if slow_agent_ids not specified)
        slow_speed: speed for slow agents in knots
        fast_speed: speed for fast agents in knots

    Returns:
        agent_speeds: array of shape (num_agents,) with speeds in km/s
    """
    # Convert knots to km/s
    slow_speed_kms = slow_speed * 0.514444 * 0.001
    fast_speed_kms = fast_speed * 0.514444 * 0.001

    # Initialize all agents as fast
    agent_speeds = np.full(num_agents, fast_speed_kms)

    # Determine which agents are slow
    if slow_agent_ids is not None:
        slow_ids = slow_agent_ids
    elif slow_percentage is not None:
        num_slow = int(num_agents * slow_percentage / 100.0)
        # Spread slow agents evenly (not just first N)
        slow_ids = np.linspace(0, num_agents-1, num_slow, dtype=int).tolist()
    else:
        slow_ids = []

    # Set slow agent speeds
    for agent_id in slow_ids:
        if agent_id < num_agents:
            agent_speeds[agent_id] = slow_speed_kms

    print(f"\nSpeed Configuration:")
    print(f"  Fast agents ({fast_speed} knots): {num_agents - len(slow_ids)} agents")
    print(f"  Slow agents ({slow_speed} knots): {len(slow_ids)} agents")
    print(f"  Slow agent IDs: {slow_ids}")

    return agent_speeds, slow_ids


def main():
    parser = argparse.ArgumentParser(description='Evaluate heterogeneous speed scenarios')

    # Basic parameters
    parser.add_argument('--scenario_name', type=str, default='simple_combined_graph',
                        help='Scenario to evaluate')
    parser.add_argument('--num_agents', type=int, default=10,
                        help='Total number of agents')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to run')

    # Speed configuration
    parser.add_argument('--slow_agent_ids', type=int, nargs='+', default=None,
                        help='Specific agent IDs to be slow (e.g., --slow_agent_ids 0 1 2)')
    parser.add_argument('--slow_percentage', type=float, default=None,
                        help='Percentage of agents to be slow (alternative to --slow_agent_ids)')
    parser.add_argument('--slow_speed', type=float, default=110,
                        help='Speed for slow agents in knots (default: 110)')
    parser.add_argument('--fast_speed', type=float, default=175,
                        help='Speed for fast agents in knots (default: 175)')

    # Evaluation options
    parser.add_argument('--model_dir', type=str,
                        default='model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/',
                        help='Directory containing trained model')
    parser.add_argument('--output_dir', type=str, default='eval_heterogeneous',
                        help='Directory to save results')
    parser.add_argument('--save_trajectories', action='store_true',
                        help='Save trajectory data for visualization')
    parser.add_argument('--save_gifs', action='store_true',
                        help='Save GIF animations')

    # Environment parameters
    parser.add_argument('--world_size', type=float, default=5.0,
                        help='World size in km')
    parser.add_argument('--episode_length', type=int, default=350,
                        help='Episode length in steps')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()

    # Validate input
    if args.slow_agent_ids is None and args.slow_percentage is None:
        print("Error: Must specify either --slow_agent_ids or --slow_percentage")
        return

    # Create speed configuration
    agent_speeds, slow_ids = create_heterogeneous_config(
        args.num_agents,
        slow_agent_ids=args.slow_agent_ids,
        slow_percentage=args.slow_percentage,
        slow_speed=args.slow_speed,
        fast_speed=args.fast_speed
    )

    # Save speed configuration
    os.makedirs(args.output_dir, exist_ok=True)
    speed_config_file = os.path.join(args.output_dir, 'speed_config.npz')
    np.savez(speed_config_file,
             agent_speeds=agent_speeds,
             slow_ids=slow_ids,
             slow_speed_knots=args.slow_speed,
             fast_speed_knots=args.fast_speed,
             num_agents=args.num_agents)
    print(f"Saved speed configuration to {speed_config_file}")

    # Build evaluation command
    cmd = [
        'python', 'onpolicy/scripts/eval_mpe.py',
        f'--model_dir={args.model_dir}',
        f'--scenario_name={args.scenario_name}',
        f'--num_agents={args.num_agents}',
        f'--render_episodes={args.num_episodes}',
        f'--world_size={args.world_size}',
        f'--episode_length={args.episode_length}',
        f'--seed={args.seed}',
        '--dynamics_type=air_taxi',
        '--use_dones=False',
        '--collaborative=False',
        '--goal_rew=20',
        '--collision_rew=20',
        '--formation_rew=10',
        '--num_walls=0',
        '--eval_mode',
        f'--eval_output_dir={args.output_dir}',
    ]

    if args.save_trajectories:
        cmd.append('--save_trajectories')

    if args.save_gifs:
        cmd.append('--save_gifs')

    print(f"\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}\n")

    import subprocess
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ Evaluation completed successfully!")
        print(f"\nResults saved to: {args.output_dir}/")
        print(f"\nNext steps:")
        print(f"  1. View metrics: cat {args.output_dir}/eval_summary.csv")

        if args.save_trajectories:
            print(f"  2. Visualize trajectories:")
            print(f"     python eval_scripts/plot_trajectories.py \\")
            print(f"       --trajectory_file {args.output_dir}/trajectory_episode_0.npz \\")
            print(f"       --highlight_agents {' '.join(map(str, slow_ids[:5]))}")  # Highlight first 5 slow agents
    else:
        print(f"\n✗ Evaluation failed with return code {result.returncode}")


if __name__ == '__main__':
    main()
