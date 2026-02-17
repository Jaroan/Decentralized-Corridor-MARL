"""
Evaluation metrics logger - ONLY activated during formal evaluations.
Does not interfere with training or debugging runs.

This module can be optionally imported and used only when --eval_mode flag is set.
"""

import os
import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List


class EvalMetricsLogger:
    """
    Lightweight metrics logger for formal evaluations only.

    Only tracks data when explicitly enabled via --eval_mode flag.
    Does not interfere with normal training or debugging.
    """

    def __init__(self, output_dir: str, enabled: bool = False, num_agents: int = 0, heterogeneous: bool = False):
        """
        Initialize metrics logger.

        Args:
            output_dir: Directory to save evaluation metrics
            enabled: If False, all methods are no-ops (no overhead)
            num_agents: Number of agents in simulation
            heterogeneous: Whether this is a heterogeneous speed run
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.num_agents = num_agents
        self.heterogeneous = heterogeneous

        if not self.enabled:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Episode-level data storage
        self.episodes_data = {
            'conformance_pct': [],
            'success_rate': [],
            'completion_time': [],
            'delta_d': [],
            'spacing_violations': [],
            'throughput': [],  # agents per minute at exit
            'num_agents': None,
            'world_size': None,
            'episode_length': None,
        }

    def log_episode(self,
                    conformance_pct: float,
                    success_rate: float,
                    completion_time: float,
                    delta_d: float,
                    spacing_violations: float,
                    throughput: float = 0.0):
        """
        Log metrics for a single episode.

        Args:
            conformance_pct: Conformance to corridor boundaries (0-100)
            success_rate: Success rate (0-100)
            completion_time: Time to complete episode (seconds)
            delta_d: Mean separation violation when violations occur (meters)
            spacing_violations: Fraction of time with violations (0-100)
            throughput: Exit throughput in agents per minute
        """
        if not self.enabled:
            return

        self.episodes_data['conformance_pct'].append(conformance_pct)
        self.episodes_data['success_rate'].append(success_rate)
        self.episodes_data['completion_time'].append(completion_time)
        self.episodes_data['delta_d'].append(delta_d)
        self.episodes_data['spacing_violations'].append(spacing_violations)
        self.episodes_data['throughput'].append(throughput)

    def set_config(self, num_agents: int, world_size: float, episode_length: int):
        """Store configuration parameters."""
        if not self.enabled:
            return

        self.episodes_data['num_agents'] = num_agents
        self.episodes_data['world_size'] = world_size
        self.episodes_data['episode_length'] = episode_length

    def save_summary(self):
        """
        Save aggregated summary statistics to CSV.
        Only saves data collected from formal evaluation runs.
        """
        if not self.enabled:
            return

        # Compute summary statistics
        summary = {
            'num_agents': self.episodes_data['num_agents'],
            'world_size': self.episodes_data['world_size'],
            'episode_length': self.episodes_data['episode_length'],
            'num_episodes': len(self.episodes_data['conformance_pct']),
        }

        # Compute mean and std for each metric
        for metric in ['conformance_pct', 'success_rate', 'completion_time',
                       'delta_d', 'spacing_violations', 'throughput']:
            data = self.episodes_data[metric]
            if len(data) > 0:
                summary[f'{metric}_mean'] = np.mean(data)
                summary[f'{metric}_std'] = np.std(data)
                summary[f'{metric}_median'] = np.median(data)
                summary[f'{metric}_min'] = np.min(data)
                summary[f'{metric}_max'] = np.max(data)
            else:
                summary[f'{metric}_mean'] = 0.0
                summary[f'{metric}_std'] = 0.0
                summary[f'{metric}_median'] = 0.0
                summary[f'{metric}_min'] = 0.0
                summary[f'{metric}_max'] = 0.0

        # Save to CSV with descriptive filename
        run_type = 'heterogeneous' if self.heterogeneous else 'homogeneous'
        csv_filename = f'eval_summary_{self.num_agents}agents_{run_type}.csv'
        csv_path = os.path.join(self.output_dir, csv_filename)

        # Write header if file doesn't exist
        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(summary)

        print(f"✓ Evaluation metrics saved to: {csv_path}")

        # Also save raw episode data
        self._save_raw_data()

    def _save_raw_data(self):
        """Save raw per-episode data for detailed analysis."""
        import pickle

        run_type = 'heterogeneous' if self.heterogeneous else 'homogeneous'
        raw_filename = f'eval_raw_data_{self.num_agents}agents_{run_type}.pkl'
        raw_path = os.path.join(self.output_dir, raw_filename)

        with open(raw_path, 'wb') as f:
            pickle.dump(self.episodes_data, f)

        print(f"✓ Raw episode data saved to: {raw_path}")

    def get_summary_string(self) -> str:
        """Get a quick summary string for printing."""
        if not self.enabled:
            return "Metrics logging disabled"

        if len(self.episodes_data['conformance_pct']) == 0:
            return "No episodes logged yet"

        return (
            f"Episodes: {len(self.episodes_data['conformance_pct'])} | "
            f"Success: {np.mean(self.episodes_data['success_rate']):.1f}% | "
            f"Conformance: {np.mean(self.episodes_data['conformance_pct']):.1f}% | "
            f"Time: {np.mean(self.episodes_data['completion_time']):.1f}s"
        )


def create_logger(all_args) -> EvalMetricsLogger:
    """
    Factory function to create metrics logger based on args.

    Args:
        all_args: Argument namespace with eval_mode and eval_output_dir

    Returns:
        EvalMetricsLogger instance (enabled only if eval_mode=True)
    """
    # Only enable if explicitly in evaluation mode
    enabled = getattr(all_args, 'eval_mode', False)
    num_agents = getattr(all_args, 'num_agents', 0)
    heterogeneous = getattr(all_args, 'heterogeneous_speeds', False)

    if enabled:
        output_dir = getattr(all_args, 'eval_output_dir', 'eval_results')
        run_type = 'heterogeneous' if heterogeneous else 'homogeneous'
        print(f"📊 Evaluation metrics logging ENABLED")
        print(f"   Output: {output_dir}")
        print(f"   Agents: {num_agents} ({run_type})")
    else:
        output_dir = None

    return EvalMetricsLogger(output_dir=output_dir, enabled=enabled,
                            num_agents=num_agents, heterogeneous=heterogeneous)
