#!/bin/bash
# Run heterogeneous experiments and generate outflow rate plots

echo "========================================="
echo "HETEROGENEOUS COMBINED GRAPH ANALYSIS"
echo "========================================="
echo ""

# Step 1: Run experiments
echo "Step 1: Running experiments..."
./eval_scripts/run_heterogeneous_combined_graph.sh

# Check if experiments succeeded
if [ $? -ne 0 ]; then
    echo "Error: Experiments failed"
    exit 1
fi

echo ""
echo "========================================="
echo "Step 2: Generating outflow rate plot..."
echo "========================================="
echo ""

# Create output directory
mkdir -p eval_heterogeneous_combined_graph/plots

# Step 2: Generate plot
python eval_scripts/plot_outflow_rate_averaged.py \
  --trajectory_dir eval_heterogeneous_combined_graph \
  --num_episodes 5 \
  --output_dir eval_heterogeneous_combined_graph/plots

# Check if plotting succeeded
if [ $? -ne 0 ]; then
    echo "Error: Plotting failed"
    exit 1
fi

echo ""
echo "========================================="
echo "ANALYSIS COMPLETE"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Trajectories: eval_heterogeneous_combined_graph/"
echo "  - Plots: eval_heterogeneous_combined_graph/plots/"
echo ""
