#!/usr/bin/env python
"""
Generate LaTeX table from route speed analysis.

Usage:
    python eval_scripts/generate_route_speeds_latex.py \
        --summary_file eval_40agents_combined_graph/analysis/route_speeds_summary.csv \
        --output_file eval_40agents_combined_graph/analysis/route_speeds_table.tex
"""

import argparse
import pandas as pd
import os


def generate_latex_table(summary_file: str, output_file: str):
    """Generate LaTeX table from route speeds summary CSV."""

    print(f"Loading data from: {summary_file}")

    # Read summary data with multi-level header
    df = pd.read_csv(summary_file, header=[0, 1, 2], skiprows=0)

    # Flatten the column names and read again with simpler approach
    # Just read as regular CSV skipping the header rows
    df = pd.read_csv(summary_file, skiprows=2)

    # Column order: route_id, route_name, count, mean, min, max, std,
    #               corridor_opt, corridor_actual, corridor_eff,
    #               opt_total, actual_traj, path_eff, time
    df.columns = ['route_id', 'route_name', 'count', 'mean_speed', 'min_speed',
                  'max_speed', 'std_speed', 'corridor_optimal', 'corridor_actual',
                  'corridor_efficiency', 'optimal_total', 'actual_trajectory',
                  'path_efficiency', 'avg_time']

    latex_content = []

    # Table header
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Corridor navigation performance for simple\\_combined\\_graph scenario with 40 agents. Distances measured from first corridor entry to final exit, excluding approach distance.}")
    latex_content.append("\\label{table:route_speeds}")
    latex_content.append("\\begin{tabular}{llccc}")
    latex_content.append("\\hline")
    latex_content.append("Route & Entry$\\rightarrow$Exit & Avg Speed & Actual Dist & Optimal Dist \\\\")
    latex_content.append(" & & (knots) & (km) & (km) \\\\")
    latex_content.append("\\hline")

    # Read and format data
    for idx, row in df.iterrows():
        route_id = int(row['route_id'])
        route_name = row['route_name']

        # Parse route name (e.g., "A→T16 (left)")
        parts = route_name.split('→')
        entry = parts[0].strip()
        exit_part = parts[1].strip() if len(parts) > 1 else ""

        # Extract statistics (corridor-only distances)
        mean_speed = float(row['mean_speed'])
        corridor_actual = float(row['corridor_actual'])
        corridor_optimal = float(row['corridor_optimal'])

        # Format row
        line = (f"R{route_id} & {entry}$\\rightarrow${exit_part} & "
                f"{mean_speed:.1f} & "
                f"{corridor_actual:.1f} & "
                f"{corridor_optimal:.1f} \\\\")

        latex_content.append(line)

    # Table footer
    latex_content.append("\\hline")
    latex_content.append("\\multicolumn{5}{l}{\\footnotesize Max theoretical speed: 175.0 knots} \\\\")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    # Write to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"\nLaTeX table saved to: {output_file}")
    print("\nTo include in your paper, add:")
    print(f"  \\input{{{output_file}}}")
    print("\nOr copy the contents directly into your .tex file")

    # Also print to console
    print("\n" + "="*80)
    print("LATEX TABLE CONTENT:")
    print("="*80)
    print('\n'.join(latex_content))
    print("="*80)


def generate_figure_latex(analysis_dir: str, output_file: str):
    """Generate LaTeX code for including the figures."""

    latex_content = []

    # Box plot figure
    latex_content.append("% Box plot of route speeds")
    latex_content.append("\\begin{figure}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(f"\\includegraphics[width=0.8\\textwidth]{{{analysis_dir}/route_speeds_boxplot.pdf}}")
    latex_content.append("\\caption{Distribution of agent speeds by route. Box plots show median, quartiles, and outliers. The red dashed line indicates the maximum theoretical speed of 175 knots. Route 5 (CL$\\rightarrow$T17) exhibits high variability and a significant outlier at 87.3 knots, suggesting a bottleneck.}")
    latex_content.append("\\label{fig:route_speeds_boxplot}")
    latex_content.append("\\end{figure}")
    latex_content.append("")

    # Scatter plot figure
    latex_content.append("% Scatter plot of speed vs distance")
    latex_content.append("\\begin{figure}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(f"\\includegraphics[width=0.8\\textwidth]{{{analysis_dir}/speed_vs_distance.pdf}}")
    latex_content.append("\\caption{Agent speed versus total distance traveled, colored by route. Longer routes (Routes 0, 1 from entry A) require agents to traverse greater distances. The outlier at approximately 87 knots on Route 5 indicates severe performance degradation due to conflict resolution or spacing maintenance.}")
    latex_content.append("\\label{fig:speed_vs_distance}")
    latex_content.append("\\end{figure}")

    # Write to file
    fig_file = output_file.replace('.tex', '_figures.tex')
    with open(fig_file, 'w') as f:
        f.write('\n'.join(latex_content))

    print(f"\nFigure LaTeX code saved to: {fig_file}")
    print("\n" + "="*80)
    print("FIGURE LATEX CONTENT:")
    print("="*80)
    print('\n'.join(latex_content))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table and figure code from route speed analysis'
    )
    parser.add_argument('--summary_file', type=str, required=True,
                       help='Path to route_speeds_summary.csv')
    parser.add_argument('--output_file', type=str, default='route_speeds_table.tex',
                       help='Output LaTeX file path')
    parser.add_argument('--analysis_dir', type=str, default='.',
                       help='Directory containing analysis plots (for figure paths)')

    args = parser.parse_args()

    # Generate table
    generate_latex_table(args.summary_file, args.output_file)

    # Generate figure code
    generate_figure_latex(args.analysis_dir, args.output_file)

    print("\n✓ LaTeX generation complete!")
    print("\nNext steps:")
    print("  1. Copy the table into your paper's .tex file")
    print("  2. Copy the figure code into your paper's .tex file")
    print("  3. Make sure the PDF plots are in the correct directory")
    print("  4. Compile your LaTeX document")


if __name__ == '__main__':
    main()
