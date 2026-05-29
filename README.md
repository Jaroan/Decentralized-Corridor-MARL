# Decentralized Autonomous Traffic Management through Corridor Networks

This repository contains research code for decentralized coordination of autonomous aircraft through air mobility corridors using multi-agent reinforcement learning (MARL).

## Overview

This work addresses the challenge of coordinating multiple autonomous vehicles through shared corridor networks without centralized control. The approach uses decentralized multi-agent reinforcement learning to enable aircraft to safely and efficiently navigate shared airspace while maintaining spacing constraints and minimizing conflicts.

## Research Papers

### 1. Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors
**Authors:** Jasmine J. Aloor and Hamsa Balakrishnan  
**Presented at:** AIAA 2026 SciTech Forum  
**Session:** Air Traffic Management for Advanced Aircraft Concepts

### 2. Decentralized Autonomous Traffic Management through Corridor Networks
**Authors:** Jasmine Jerry Aloor, Aadarsh Govada, and Hamsa Balakrishnan  
**Accepted for presentation at:** Second US-Europe Air Transportation Research and Development Symposium (ATRDS 2026), June 2026

**Contact:** jjaloor@mit.edu

## Features

- **Decentralized Control**: Agents learn policies without centralized coordination
- **Safety Constraints**: Maintains minimum spacing between aircraft in corridors
- **Scalability**: Multi-agent approach enables scaling to large numbers of vehicles
- **Corridor Networks**: Models realistic corridor-based traffic structures
- **Multi-Agent Reinforcement Learning**: Uses MARL algorithms for policy learning

## Installation

```bash
# Clone the repository
git clone https://github.com/Jaroan/Decentralized-Corridor-MARL.git
cd Decentralized-Corridor-MARL

# Create conda environment
conda create -n corridor-marl python=3.11
conda activate corridor-marl

# Install dependencies
pip install pip==23.1.2
pip install torch==2.0.1
pip install -r requirements.txt
```

### Dependencies
- Python 3.11+
- PyTorch 2.0.1+
- OpenAI Gym
- Multi-agent environment libraries

## Usage

### Training

Train agents on a corridor scenario:

```bash
python -u onpolicy/scripts/train_mpe.py \
  --project_name "corridor-experiment" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed 2 \
  --experiment_name "decentralized-corridor" \
  --scenario_name "three_phase_graph"
```

### Evaluation

Evaluate trained models:

```bash
python onpolicy/scripts/eval_mpe.py \
  --model_dir='model_weights/corridor_model' \
  --render_episodes=5 \
  --num_agents=4 \
  --scenario_name='three_phase_graph' \
  --save_gifs \
  --use_render
```

## Code Structure

```
.
├── README.md                      # Project overview
├── requirements.txt               # Dependencies
├── multiagent/                    # Core environment and scenarios
│   ├── custom_scenarios/          # Corridor scenario definitions
│   │   └── three_phase_graph.py   # Three-phase corridor network
│   ├── environment.py             # Multi-agent environment
│   └── utils.py                   # Utility functions
├── onpolicy/                      # Training and evaluation scripts
│   ├── scripts/
│   │   ├── train_mpe.py          # Training script
│   │   └── eval_mpe.py           # Evaluation script
│   └── algorithms/               # MARL algorithm implementations
├── model_weights/                # Trained model checkpoints
└── eval_scripts/                 # Evaluation utilities and plotting
```

## Key Results

- **Decentralized coordination** of multiple agents in shared corridor networks
- **Safety maintenance** through learned spacing policies
- **Scalable performance** across varying numbers of agents

## Citation

If you use this code or build upon this work, please cite the relevant papers:

```bibtex
@inproceedings{aloor2026decentralized_corridors,
  title={Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors},
  author={Aloor, Jasmine Jerry and Balakrishnan, Hamsa},
  booktitle={AIAA SciTech Forum},
  year={2026}
}

@inproceedings{aloor2026corridor_networks,
  title={Decentralized Autonomous Traffic Management through Corridor Networks},
  author={Aloor, Jasmine Jerry and Govada, Aadarsh and Balakrishnan, Hamsa},
  booktitle={Second US-Europe Air Transportation Research and Development Symposium (ATRDS)},
  year={2026}
}
```

## Acknowledgements

This work was supported in part by NASA grants and the Department of the Air Force Artificial Intelligence Accelerator.

## Questions & Contact

For questions or feedback about this code, please contact:
- **Jasmine Jerry Aloor** (jjaloor@mit.edu)
- File an issue on GitHub for bug reports and feature requests

## License

See LICENSE file for details.
