#!/usr/bin/env python
"""
Comprehensive evaluation script for AAM corridor scenarios.
Runs 100 episodes per configuration and computes performance metrics:
- Conformance to corridor boundaries (C%)
- Success rate (S%)
- Completion time per episode (T, s)
- Violation of separation minimum (Δd, m)
- Need for tactical intervention (I%)
"""

import os
import sys
import subprocess
import argparse
import numpy as np
from pathlib import Path

# Scenario configurations
SCENARIOS = {
    'merge': {
        'name': 'three_phase_graph_merge',
        'world_size_base': 5,
        'episode_length_base': 200,
    },
    # 'double_merge': {
    #     'name': 'three_phase_graph_double_merge',
    #     'world_size_base': 5,
    #     'episode_length_base': 200,
    # },
    # 'split_merge': {
    #     'name': 'three_phase_graph_split_and_merge',
    #     'world_size_base': 5,
    #     'episode_length_base': 200,
    # },
    # 'combined': {
    #     'name': 'simple_combined_graph',
    #     'world_size_base': 8,
    #     'episode_length_base': 350,
    # },
}

# Agent counts to test
AGENT_COUNTS = [30, 40]

# Model configurations
MODELS = {
    'standard': {
        'dir': 'model_weights/tube/rot_inv/airtaxi/try/three/test2026/5ag/low_width/',
        'max_speed': 175 * 0.0003048,  # 175 ft/s converted to km/s
        'description': 'Standard speed model'
    },
    # Add heterogeneous models later
    # 'slow_v1': {
    #     'dir': 'model_weights/tube/rot_inv/airtaxi/slow_v1',
    #     'max_speed': 1.5,
    #     'description': 'Reduced speed model v1'
    # },
    # 'slow_v2': {
    #     'dir': 'model_weights/tube/rot_inv/airtaxi/slow_v2',
    #     'max_speed': 1.2,
    #     'description': 'Reduced speed model v2'
    # },
}

def get_world_size(base_size, num_agents):
    """Scale world size based on number of agents."""
    # Scale by sqrt of agent ratio to maintain density
    scale_factor = 1.0  # np.sqrt(num_agents / 10)
    return int(base_size * scale_factor)

def get_episode_length(base_length, num_agents, scenario_name):
    """Scale episode length based on number of agents."""
    # # Increase episode length for more agents to allow completion
    # scale_factor = 1.0 + 0.1 * (num_agents - 10) / 10
    # for merges: 10 agents: 200 steps, 20 agents: 250 steps, 30 agents: 350 steps, 40 agents: 450 steps
    if scenario_name in ['merge', 'double_merge']:
        if num_agents == 10:
            scale_factor = 1.0
        elif num_agents == 20:
            scale_factor = 1.25
        elif num_agents == 30:
            scale_factor = 1.75
        elif num_agents == 40:
            scale_factor = 2.25
        else:
            scale_factor = 1.0
    return int(base_length * scale_factor)

def run_evaluation(scenario_name, scenario_config, num_agents, model_config,
                   output_dir, seed=0, render=False):
    """Run a single evaluation configuration."""

    world_size = get_world_size(scenario_config['world_size_base'], num_agents)
    episode_length = get_episode_length(scenario_config['episode_length_base'], num_agents, scenario_name)

    # Create output directory
    eval_name = f"{scenario_name}_agents{num_agents}_seed{seed}"
    eval_output = os.path.join(output_dir, eval_name)
    os.makedirs(eval_output, exist_ok=True)

    # Build command
    cmd = [
        'python', 'onpolicy/scripts/eval_mpe.py',
        f'--model_dir={model_config["dir"]}',
        '--render_episodes=100',
        f'--world_size={world_size}',
        f'--num_agents={num_agents}',
        '--num_obstacles=0',
        f'--seed={seed}',
        '--num_landmarks=1',
        f'--episode_length={episode_length}',
        '--use_dones=False',
        '--collaborative=False',
        '--model_name=Combined',
        f'--scenario_name={scenario_config["name"]}',
        '--dynamics_type=air_taxi',
        '--goal_rew=20',
        '--collision_rew=20',
        '--fair_rew=1',
        '--formation_rew=10',
        '--num_walls=0',
        '--zeroshift=5',
        '--min_obs_dist=0.5',
        '--total_actions=9',
        '--formation_type=point',
        f'--eval_output_dir={eval_output}',
        '--eval_mode',  # IMPORTANT: Enables metrics logging
    ]

    # if render:
    #     cmd.append('--use_render')
    #     cmd.append('--save_gifs')

    print(f"\n{'='*80}")
    print(f"Running: {eval_name}")
    print(f"World size: {world_size}, Episode length: {episode_length}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Completed: {eval_name}")

        # Save stdout/stderr
        with open(os.path.join(eval_output, 'stdout.txt'), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(eval_output, 'stderr.txt'), 'w') as f:
            f.write(result.stderr)

        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {eval_name}")
        print(f"Error: {e}")

        # Save error output
        with open(os.path.join(eval_output, 'error.txt'), 'w') as f:
            f.write(f"Return code: {e.returncode}\n")
            f.write(f"stdout:\n{e.stdout}\n")
            f.write(f"stderr:\n{e.stderr}\n")

        return False

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive AAM corridor evaluations')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--scenarios', type=str, nargs='+',
                        choices=list(SCENARIOS.keys()) + ['all'],
                        default=['all'],
                        help='Scenarios to evaluate')
    parser.add_argument('--agent_counts', type=int, nargs='+',
                        default=AGENT_COUNTS,
                        help='Number of agents to test')
    parser.add_argument('--model', type=str, default='standard',
                        choices=list(MODELS.keys()),
                        help='Model to use for evaluation')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Enable rendering and gif saving (WARNING: slow)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configurations without running')

    args = parser.parse_args()

    # Determine which scenarios to run
    if 'all' in args.scenarios:
        scenarios_to_run = list(SCENARIOS.keys())
    else:
        scenarios_to_run = args.scenarios

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Get model config
    model_config = MODELS[args.model]

    # Print experiment plan
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION PLAN")
    print(f"{'='*80}")
    print(f"Scenarios: {', '.join(scenarios_to_run)}")
    print(f"Agent counts: {args.agent_counts}")
    print(f"Model: {args.model} - {model_config['description']}")
    print(f"Episodes per config: 100")
    print(f"Output directory: {output_dir}")
    print(f"Rendering: {'Enabled (WARNING: will be very slow)' if args.render else 'Disabled'}")
    print(f"{'='*80}\n")

    # Calculate total runs
    total_runs = len(scenarios_to_run) * len(args.agent_counts)
    print(f"Total evaluation runs: {total_runs}")
    print(f"Estimated time: ~{total_runs * 10} minutes (assuming ~10 min per 100 episodes)\n")

    if args.dry_run:
        print("DRY RUN - No evaluations will be executed")
        return

    # Confirm to proceed
    # response = input("Proceed with evaluations? [y/N]: ")
    # if response.lower() != 'y':
    #     print("Aborted.")
    #     return

    # Run evaluations
    results = []
    completed = 0
    failed = 0

    for scenario_name in scenarios_to_run:
        scenario_config = SCENARIOS[scenario_name]

        for num_agents in args.agent_counts:
            success = run_evaluation(
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                num_agents=num_agents,
                model_config=model_config,
                output_dir=output_dir,
                seed=args.seed,
                render=args.render
            )

            results.append({
                'scenario': scenario_name,
                'num_agents': num_agents,
                'success': success
            })

            if success:
                completed += 1
            else:
                failed += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    # Print failed runs
    if failed > 0:
        print("Failed runs:")
        for r in results:
            if not r['success']:
                print(f"  - {r['scenario']} with {r['num_agents']} agents")

    print("\nNext steps:")
    print("1. Run the metrics computation script to analyze results:")
    print(f"   python eval_scripts/compute_metrics_simple.py --results_dir {output_dir}")
    print("2. Generate plots and tables for the paper")

    # Display summary of metrics
    print(f"\n{'='*80}")
    print("METRICS SUMMARY")
    print(f"{'='*80}\n")

    import pandas as pd

    for r in results:
        if r['success']:
            scenario_name = r['scenario']
            num_agents = r['num_agents']
            eval_name = f"{scenario_name}_agents{num_agents}_seed{args.seed}"
            csv_file = os.path.join(output_dir, eval_name, 'eval_summary.csv')

            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    print(f"--- {scenario_name} ({num_agents} agents) ---")
                    print(f"  Episodes:        {df['num_episodes'].values[0]}")
                    print(f"  Success Rate:    {df['success_rate_mean'].values[0]:.1f}%")
                    print(f"  Conformance:     {df['conformance_pct_mean'].values[0]:.1f}%")
                    print(f"  Completion Time: {df['completion_time_mean'].values[0]:.1f} ± {df['completion_time_std'].values[0]:.1f} s")
                    print(f"  Throughput:      {df['throughput_mean'].values[0]:.2f} agents/min")
                    print(f"  Δd (violations): {df['delta_d_mean'].values[0]:.2f} ± {df['delta_d_std'].values[0]:.2f} m")
                    print(f"  Intervention:    {df['spacing_violations_mean'].values[0]:.1f}%")
                    print()
                except Exception as e:
                    print(f"  ✗ Error reading {csv_file}: {e}\n")

    print(f"{'='*80}")

if __name__ == '__main__':
    main()
