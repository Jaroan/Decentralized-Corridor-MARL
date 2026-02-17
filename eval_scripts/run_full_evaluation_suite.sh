#!/bin/bash
# Full evaluation suite for AAM corridor scenarios
# This script runs all evaluations, computes metrics, and generates plots

set -e  # Exit on error

# Configuration
OUTPUT_DIR="eval_results_$(date +%Y%m%d_%H%M%S)"
SCENARIOS="all"
AGENT_COUNTS="30 40"
MODEL="standard" ## standard or low speed

echo "=========================================="
echo "AAM Corridor Comprehensive Evaluation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Scenarios: $SCENARIOS"
echo "  Agent counts: $AGENT_COUNTS"
echo "  Model: $MODEL"
echo ""

# Step 1: Run evaluations
echo "=========================================="
echo "STEP 1: Running Evaluations"
echo "=========================================="
echo ""

python eval_scripts/run_comprehensive_eval.py \
    --output_dir "$OUTPUT_DIR" \
    --scenarios $SCENARIOS \
    --agent_counts $AGENT_COUNTS \
    --model $MODEL

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo ""
echo "✓ Evaluations completed successfully"
echo ""

# Step 2: Compute metrics
echo "=========================================="
echo "STEP 2: Computing Metrics"
echo "=========================================="
echo ""

python eval_scripts/compute_metrics.py \
    --results_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Metrics computation failed!"
    exit 1
fi

echo ""
echo "✓ Metrics computed successfully"
echo ""

# # Step 3: Generate plots
# echo "=========================================="
# echo "STEP 3: Generating Plots"
# echo "=========================================="
# echo ""

# python eval_scripts/plot_results.py \
#     --metrics_file "$OUTPUT_DIR/raw_metrics.csv" \
#     --output_dir "$OUTPUT_DIR/plots"

# if [ $? -ne 0 ]; then
#     echo "ERROR: Plot generation failed!"
#     exit 1
# fi

# echo ""
# echo "✓ Plots generated successfully"
# echo ""

# Summary
echo "=========================================="
echo "EVALUATION SUITE COMPLETED"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - raw_metrics.csv          (raw data)"
echo "  - metrics_table.csv        (summary table)"
echo "  - metrics_table.tex        (LaTeX table)"
echo "  - plots/                   (all figures)"
echo ""
echo "Next steps:"
echo "  1. Review metrics_table.csv for results summary"
echo "  2. Include metrics_table.tex in your paper"
echo "  3. Use plots/ figures in your paper"
echo ""
echo "For questions, see eval_scripts/README_COMPREHENSIVE_EVAL.md"
echo ""
