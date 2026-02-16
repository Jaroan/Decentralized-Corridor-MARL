#!/usr/bin/env python
"""
Create plots for AAM corridor evaluation results.
Generates figures suitable for paper submission.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (7, 5)


def plot_metric_vs_agents(df, metric_name, metric_label, ylabel, better_direction='lower',
                           output_path=None):
    """
    Plot a metric vs number of agents for all scenarios.

    Args:
        df: DataFrame with metrics
        metric_name: Column name for the metric
        metric_label: Display label for the metric
        ylabel: Y-axis label
        better_direction: 'lower' or 'higher' to indicate which is better
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = df['scenario'].unique()
    x = sorted(df['num_agents'].unique())

    for scenario in scenarios:
        scenario_data = df[df['scenario'] == scenario]
        y_means = []
        y_stds = []

        for num_agents in x:
            data = scenario_data[scenario_data['num_agents'] == num_agents][metric_name]
            if len(data) > 0:
                y_means.append(data.mean())
                y_stds.append(data.std())
            else:
                y_means.append(np.nan)
                y_stds.append(0)

        y_means = np.array(y_means)
        y_stds = np.array(y_stds)

        # Plot line with error bars
        ax.errorbar(x, y_means, yerr=y_stds, marker='o', label=scenario,
                    linewidth=2, markersize=6, capsize=4)

    ax.set_xlabel('Number of Aircraft', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{metric_label} vs. Traffic Density', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation for better direction
    direction_text = "↓ Lower is better" if better_direction == 'lower' else "↑ Higher is better"
    ax.text(0.02, 0.98, direction_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9, style='italic', alpha=0.7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_all_metrics(df, output_dir):
    """Create plots for all metrics."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Conformance to corridor boundaries
    plot_metric_vs_agents(
        df, 'conformance_pct', 'Conformance to Corridor Boundaries',
        'Conformance (C%)', better_direction='higher',
        output_path=os.path.join(output_dir, 'conformance_vs_agents.pdf')
    )

    # 2. Success rate
    plot_metric_vs_agents(
        df, 'success_pct', 'Success Rate',
        'Success Rate (S%)', better_direction='higher',
        output_path=os.path.join(output_dir, 'success_rate_vs_agents.pdf')
    )

    # 3. Completion time
    plot_metric_vs_agents(
        df, 'completion_time_mean', 'Average Completion Time',
        'Completion Time (s)', better_direction='lower',
        output_path=os.path.join(output_dir, 'completion_time_vs_agents.pdf')
    )

    # 4. Separation violations
    plot_metric_vs_agents(
        df, 'delta_d_mean', 'Separation Violation Magnitude',
        'Δd (m)', better_direction='lower',
        output_path=os.path.join(output_dir, 'separation_violation_vs_agents.pdf')
    )

    # 5. Intervention need
    plot_metric_vs_agents(
        df, 'intervention_pct', 'Need for Tactical Intervention',
        'Intervention Need (I%)', better_direction='lower',
        output_path=os.path.join(output_dir, 'intervention_vs_agents.pdf')
    )


def plot_heatmap(df, metric_name, metric_label, output_path=None):
    """
    Create a heatmap of metric values across scenarios and agent counts.

    Args:
        df: DataFrame with metrics
        metric_name: Column name for the metric
        metric_label: Display label for the metric
        output_path: Path to save figure
    """
    # Pivot table for heatmap
    pivot = df.pivot_table(
        values=metric_name,
        index='scenario',
        columns='num_agents',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': metric_label})

    ax.set_xlabel('Number of Aircraft', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    ax.set_title(f'{metric_label} Heatmap', fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_bar(df, output_path=None):
    """
    Create grouped bar chart comparing all metrics across scenarios.

    Args:
        df: DataFrame with metrics
        output_path: Path to save figure
    """
    # Normalize metrics to 0-100 scale for comparison
    metrics_to_plot = {
        'conformance_pct': 'C%',
        'success_pct': 'S%',
    }

    # Get mean values per scenario
    scenario_means = df.groupby('scenario').mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = scenario_means.index
    x = np.arange(len(scenarios))
    width = 0.35

    for i, (metric, label) in enumerate(metrics_to_plot.items()):
        offset = width * (i - len(metrics_to_plot)/2 + 0.5)
        ax.bar(x + offset, scenario_means[metric], width, label=label)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved comparison bar chart to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot AAM corridor evaluation results')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to raw_metrics.csv file')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for plots')

    args = parser.parse_args()

    # Load metrics
    df = pd.read_csv(args.metrics_file)

    print(f"Loaded {len(df)} evaluation results")
    print(f"Scenarios: {df['scenario'].unique()}")
    print(f"Agent counts: {sorted(df['num_agents'].unique())}")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Update file extension in plot functions if needed
    if args.format != 'pdf':
        # This is a simplified approach - you could make it more robust
        print(f"Note: Plots will be saved as PDF. Convert manually if needed.")

    print("\nGenerating plots...")

    # Generate all plots
    plot_all_metrics(df, output_dir)

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    for metric_name, metric_label in [
        ('conformance_pct', 'Conformance (C%)'),
        ('success_pct', 'Success Rate (S%)'),
        ('intervention_pct', 'Intervention Need (I%)'),
    ]:
        plot_heatmap(df, metric_name, metric_label,
                     output_path=os.path.join(output_dir, f'heatmap_{metric_name}.pdf'))

    # Generate comparison bar chart
    print("\nGenerating comparison chart...")
    plot_comparison_bar(df, output_path=os.path.join(output_dir, 'comparison_bar.pdf'))

    print(f"\n{'='*80}")
    print(f"PLOTTING COMPLETE")
    print(f"{'='*80}")
    print(f"All plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  - conformance_vs_agents.pdf")
    print("  - success_rate_vs_agents.pdf")
    print("  - completion_time_vs_agents.pdf")
    print("  - separation_violation_vs_agents.pdf")
    print("  - intervention_vs_agents.pdf")
    print("  - heatmap_*.pdf (3 heatmaps)")
    print("  - comparison_bar.pdf")


if __name__ == '__main__':
    main()
