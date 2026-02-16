"""
Throughput calculation for AAM corridor evaluations.

Throughput is measured as the rate of agents successfully exiting
the final corridor (agents per minute).
"""

import numpy as np
from typing import List, Tuple


def compute_throughput(exit_times: List[float],
                       total_time: float,
                       time_window: float = 60.0) -> float:
    """
    Compute exit throughput in agents per minute.

    Args:
        exit_times: List of times (in seconds) when each agent exited
        total_time: Total episode time in seconds
        time_window: Time window for rate calculation (default: 60s)

    Returns:
        Throughput in agents per minute

    Example:
        If 10 agents exit over 120 seconds:
        throughput = 10 / (120/60) = 5 agents/minute
    """
    if len(exit_times) == 0 or total_time <= 0:
        return 0.0

    # Number of agents that successfully exited
    num_exits = len([t for t in exit_times if t > 0])

    if num_exits == 0:
        return 0.0

    # Throughput = (successful exits) / (time in minutes)
    throughput = num_exits / (total_time / time_window)

    return throughput


def compute_peak_throughput(exit_times: List[float],
                            window_size: float = 60.0) -> Tuple[float, float]:
    """
    Compute peak throughput using a sliding window.

    Args:
        exit_times: List of times (in seconds) when each agent exited
        window_size: Sliding window size in seconds

    Returns:
        (peak_throughput, peak_window_start): Peak agents/minute and
                                               when it occurred
    """
    if len(exit_times) == 0:
        return 0.0, 0.0

    # Filter valid exit times
    valid_exits = sorted([t for t in exit_times if t > 0])

    if len(valid_exits) == 0:
        return 0.0, 0.0

    # Slide window through time
    max_throughput = 0.0
    peak_window_start = 0.0

    start_time = valid_exits[0]
    end_time = valid_exits[-1]

    current_time = start_time
    while current_time <= end_time:
        # Count exits in current window
        window_end = current_time + window_size
        exits_in_window = sum(1 for t in valid_exits
                             if current_time <= t < window_end)

        # Throughput for this window (agents per minute)
        window_throughput = exits_in_window / (window_size / 60.0)

        if window_throughput > max_throughput:
            max_throughput = window_throughput
            peak_window_start = current_time

        # Slide window
        current_time += window_size / 10  # 10% overlap

    return max_throughput, peak_window_start


def compute_inter_exit_time_stats(exit_times: List[float]) -> dict:
    """
    Compute statistics on time between consecutive exits.

    Args:
        exit_times: List of times when each agent exited

    Returns:
        dict with mean, std, min, max inter-exit times
    """
    valid_exits = sorted([t for t in exit_times if t > 0])

    if len(valid_exits) < 2:
        return {
            'mean_inter_exit_time': 0.0,
            'std_inter_exit_time': 0.0,
            'min_inter_exit_time': 0.0,
            'max_inter_exit_time': 0.0,
        }

    # Compute time between consecutive exits
    inter_exit_times = np.diff(valid_exits)

    return {
        'mean_inter_exit_time': float(np.mean(inter_exit_times)),
        'std_inter_exit_time': float(np.std(inter_exit_times)),
        'min_inter_exit_time': float(np.min(inter_exit_times)),
        'max_inter_exit_time': float(np.max(inter_exit_times)),
    }


def extract_exit_times_from_env_infos(env_infos: dict,
                                      dt: float = 0.1) -> List[float]:
    """
    Extract exit times from environment info dictionary.

    Args:
        env_infos: Dictionary from environment with agent status
        dt: Timestep duration in seconds

    Returns:
        List of exit times for each agent

    Note:
        This assumes env_infos contains agent status/done information.
        You may need to adapt this based on your environment's info format.
    """
    exit_times = []

    # Check if we have agent done/status information
    # Adapt this based on your actual env_infos structure
    for key, value in env_infos.items():
        if 'exit_time' in key or 'done_time' in key:
            exit_times.append(value[0] if isinstance(value, list) else value)

    return exit_times
