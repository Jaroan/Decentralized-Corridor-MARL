"""
Metrics tracking utility for AAM corridor evaluations.
This module tracks episode data needed for computing performance metrics.
"""

import numpy as np
import pickle
import os
from pathlib import Path


class MetricsTracker:
    """Track metrics data during evaluation episodes."""

    def __init__(self, output_dir=None):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics data
        """
        self.output_dir = output_dir
        self.reset()

    def reset(self):
        """Reset all tracked data."""
        # Per-episode data
        self.episode_data = {
            'agent_positions': [],        # List of (num_steps, num_agents, 2)
            'agent_velocities': [],       # List of (num_steps, num_agents, 2)
            'agent_tubes': [],            # List of (num_steps, num_agents) - current tube
            'in_tube': [],                # List of (num_steps, num_agents) - bool
            'lateral_deviation': [],      # List of (num_steps, num_agents) - y-coordinate
            'pairwise_distances': [],     # List of (num_steps, num_agents, num_agents)
            'goal_reached': [],           # List of (num_agents,) - bool
            'completion_times': [],       # List of floats - episode completion time
            'tube_params': None,          # Tube parameters (shared across episodes)
        }

        # Current episode buffer
        self.current_episode = {
            'positions': [],
            'velocities': [],
            'tubes': [],
            'in_tube': [],
            'lateral_dev': [],
            'distances': [],
        }

    def start_episode(self):
        """Start tracking a new episode."""
        self.current_episode = {
            'positions': [],
            'velocities': [],
            'tubes': [],
            'in_tube': [],
            'lateral_dev': [],
            'distances': [],
        }

    def record_step(self, world, scenario):
        """
        Record data for a single timestep.

        Args:
            world: World object from environment
            scenario: Scenario object
        """
        num_agents = len(world.agents)

        # Positions
        positions = np.array([agent.state.p_pos for agent in world.agents])
        self.current_episode['positions'].append(positions)

        # Velocities
        velocities = np.array([agent.state.p_vel for agent in world.agents])
        self.current_episode['velocities'].append(velocities)

        # Current tube for each agent
        if hasattr(scenario, 'current_tube'):
            tubes = scenario.current_tube.copy()
            self.current_episode['tubes'].append(tubes)

        # In-tube status
        in_tube = np.zeros(num_agents, dtype=bool)
        lateral_dev = np.zeros(num_agents)

        for i, agent in enumerate(world.agents):
            if hasattr(scenario, 'current_tube'):
                tube_idx = scenario.current_tube[i]
                pos = agent.state.p_pos

                # Check if in tube and get lateral deviation
                if hasattr(scenario, '_tube_coords'):
                    s, y, L, half_w = scenario._tube_coords(world, pos, tube_idx)
                    in_tube[i] = scenario._in_tube_rect(s, y, L, half_w)
                    lateral_dev[i] = y

        self.current_episode['in_tube'].append(in_tube)
        self.current_episode['lateral_dev'].append(lateral_dev)

        # Pairwise distances
        distances = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist

        self.current_episode['distances'].append(distances)

    def end_episode(self, world, scenario, episode_time):
        """
        End current episode and save data.

        Args:
            world: World object
            scenario: Scenario object
            episode_time: Total time for episode (seconds)
        """
        # Convert lists to arrays
        self.episode_data['agent_positions'].append(
            np.array(self.current_episode['positions'])
        )
        self.episode_data['agent_velocities'].append(
            np.array(self.current_episode['velocities'])
        )

        if len(self.current_episode['tubes']) > 0:
            self.episode_data['agent_tubes'].append(
                np.array(self.current_episode['tubes'])
            )

        self.episode_data['in_tube'].append(
            np.array(self.current_episode['in_tube'])
        )
        self.episode_data['lateral_deviation'].append(
            np.array(self.current_episode['lateral_dev'])
        )
        self.episode_data['pairwise_distances'].append(
            np.array(self.current_episode['distances'])
        )

        # Goal reached status
        goal_reached = np.array([agent.status for agent in world.agents])
        self.episode_data['goal_reached'].append(goal_reached)

        # Completion time
        self.episode_data['completion_times'].append(episode_time)

        # Save tube params (once)
        if self.episode_data['tube_params'] is None and hasattr(world, 'tube_params'):
            self.episode_data['tube_params'] = world.tube_params

    def save(self, filename='eval_data.pkl'):
        """
        Save all tracked data to file.

        Args:
            filename: Name of file to save to
        """
        if self.output_dir is None:
            print("Warning: No output directory specified, cannot save metrics data")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(self.episode_data, f)

        print(f"Saved metrics data to: {filepath}")

        # Also save summary statistics as JSON for quick inspection
        summary = {
            'num_episodes': len(self.episode_data['completion_times']),
            'avg_completion_time': float(np.mean(self.episode_data['completion_times'])),
            'avg_success_rate': float(np.mean([
                np.mean(goals) for goals in self.episode_data['goal_reached']
            ])) * 100,
        }

        import json
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to: {summary_path}")

    def get_summary_stats(self):
        """Get quick summary statistics."""
        if len(self.episode_data['completion_times']) == 0:
            return "No episodes tracked yet"

        num_episodes = len(self.episode_data['completion_times'])
        avg_time = np.mean(self.episode_data['completion_times'])
        success_rate = np.mean([
            np.mean(goals) for goals in self.episode_data['goal_reached']
        ]) * 100

        return (f"Tracked {num_episodes} episodes | "
                f"Avg completion time: {avg_time:.1f}s | "
                f"Success rate: {success_rate:.1f}%")
