# Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation

This project focuses on improving trust strategies in simulations through forgiveness mechanisms, particularly in language-based persuasion games. The project implements various neural architectures (LSTM, Transformer) to model and predict human choices in interactive scenarios.

## Project Overview

The project consists of several key components:
- **Environments**: Custom environments for different neural architectures (LSTM, FC, Transformer)
- **Simulation**: Implementation of decision-making strategies and game simulations
- **Data Processing**: Tools for handling game reviews and user vectors
- **Analysis**: Jupyter notebooks for analyzing results and visualizing data

## Project Structure
```
├── environments/         # Neural network environment implementations
├── Simulation/          # Game simulation and strategy implementations
├── utils/              # Helper functions and data processing utilities
├── data/               # Dataset files and game reviews
├── RunningScripts/     # Analysis and execution scripts
└── sweeps_csvs/        # Experiment results and sweeps
```

## Getting Started


### Prerequisites

Before you begin, ensure you have the following tools installed on your system:
- Python 3.8 or higher
- Git
- Anaconda or Miniconda

### Requirements
The project requires the following main dependencies:
- PyTorch 1.12.1
- Transformers
- Pandas
- Matplotlib
- Weights & Biases (for experiment tracking)
- Scikit-learn

All dependencies are specified in `requirements.yml`.

### Installation

To install and run the code on your local machine, follow these steps:

1. **Clone the repository**

   First, clone the repository to your local machine using Git. Open a terminal and run the following command:
   ```bash
   git clone https://github.com/eilamshapira/HumanChoicePrediction
    ```
2. **Create and activate the conda environment**

    After cloning the repository, navigate into the project directory:

    ```bash
    cd HumanChoicePrediction
    ```

    Then, use the following command to create a conda environment from the requirements.yml file provided in the project:
    ```bash
    conda env create -f requirements.yml
    ```
3. **Log in to Weights & Biases (W&B)**

   Weights & Biases is a machine learning platform that helps you track your experiments, visualize data, and share your findings. Logging in to W&B is essential for tracking the experiments in this project. If you haven't already, you'll need to create a W&B account. 
   Use the following command to log in to your account:
    ```bash
    wandb login
    ```

## Usage

### Running Experiments
1. To initialize your experiment:
```bash
cd RunningScripts
./clone_and_init_YOUR_UID.sh
```

2. To run a parameter sweep:
```bash
python final_sweep_YOUR_UID.py
```

3. To analyze results:
- Open `RunningScripts/analyze_results.ipynb` in Jupyter Notebook
- Follow the notebook cells to visualize and analyze your experimental results

### Key Components
- `SpecialLSTM.py`: Implementation of the custom LSTM architecture
- `StrategyTransfer.py`: Core logic for strategy transfer and adaptation
- `environments/*.py`: Different environment implementations for various neural architectures
- `Simulation/dm_strategies.py`: Decision-making strategy implementations

## Results Analysis
The project includes comprehensive analysis tools in the `RunningScripts` directory:
- `analyze_results.ipynb`: Jupyter notebook for visualizing results
- `read_wandb.py`: Script for fetching and processing W&B logs

## Citation

If you find this work useful, please cite our paper:

    @misc{shapira2024human,
          title={Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation}, 
          author={Eilam Shapira and Reut Apel and Moshe Tennenholtz and Roi Reichart},
          year={2024},
          eprint={2305.10361},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }