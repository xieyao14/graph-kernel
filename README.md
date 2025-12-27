# Deep Graph Kernel Point Processes

This repository contains the implementation of the Spatio-Temporal Point Process model with deep graph kernels. The code implements a deep learning framework for modeling event dynamics on graph structures using various graph kernel bases (ChebNet, L3Net, GAT) and temporal point processes.

## Directory Structure

Ensure your directory is organized as follows before running the code:

```
Package/
├── config/                 # Configuration files (YAML)
    ├── train_16-node-graph_mle.yaml
    ├── ... (other config yaml files)
├── data/                   # Dataset files
│   └── [data_name]/        # Subfolder matching the 'data' key in config
├── results/                # Output logs, saved models, and metrics
└── src/                    # Source code
    ├── model.py
    ├── utils.py
    ├── visualization.py
    ├── train_synthetic_data_L3.py  # main model training script
    ├── evaluation.py               # Script for model evaluation and visualization
    └── ... (other .py files)
```

## Installation & Environment Setup

To ensure a clean environment as requested during the review process, we recommend using Conda.

1. Create a new environment:
   ```bash
   conda create -n GraDK_pp python=3.9
   conda activate GraDK_pp
   ```

2. Install Dependencies:
   
   First, install PyTorch (adjust the command below based on your CUDA version/CPU availability):
   
   ### Example for CPU-only
   ```bash
   pip install torch torchvision torchaudio
   ```
   
   ### Example for CUDA 11.8
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Next, install PyTorch Geometric and other requirements:
   ```bash
   pip install -r requirements.txt
   ```
   
   *Note: Depending on your system, torch_geometric may require additional dependencies like torch-scatter or torch-sparse. Please refer to the PyG Installation Guide if you encounter errors regarding these libraries.*

## Data Preparation

The code expects data to be located in the data/ directory. 
- For synthetic experiments, ensure the graph structure (graph.pt) and event data (.npy) are located in data/[data_name]/.
- The data_name is defined in the configuration YAML file (e.g., data: "data-basecos_exp-ring_graph...").

## Usage

All scripts should be executed from the root directory of the package (i.e., Package/) to ensure relative paths for data and configuration files are resolved correctly.

### Training on Synthetic Data

To train the model using the L3Net basis on synthetic data:
```bash
python src/train_synthetic_data_L3.py -config_yaml_file config/train_16-node-graph_mle.yaml
```

To train using the GAT basis:
```bash
python src/train_synthetic_data_GAT.py -config_yaml_file config/train_16-node-graph_mle.yaml
```

### Training on Real Data

To train on real-world datasets:
```bash
python src/train_real_data_L3.py -config_yaml_file config/[your_real_data_config].yaml
```

### Training with L2 Loss

To train with L2 loss function instead of log-likelihood, switch the ``loss`` in the config yaml file from ```"likelihood"``` to ```"cont_l2_loss"```.


### Evaluation & Visualization
To evaluate a trained model, calculate metrics (Log-Likelihood, MAE, KLD), and generate visualization plots:

```bash
python src/evaluation.py
```

Note: The evaluation script loads a specific model checkpoint. Please open src/evaluation.py and ensure the model_name and checkpoint epoch match your trained model in results/saved_models/.

## Configuration

The model hyperparameters and training settings are controlled via YAML files located in the config/ directory. Key parameters include:

- Data: 
  - data: Name of the dataset folder in data/.
  - T0, T1: Time horizon start and end.
- Model Architecture:
  - n_basis_time: Number of temporal basis functions.
  - n_basis_loc: Number of graph basis.
- Training:
  - loss: Loss function type (e.g., likelihood, l2_loss).
  - batch_size, lr, epoch: Standard training parameters.
  - device: Compute device (e.g., cuda:0 or cpu).
  - t_init, t_upp, t_mul, b_bar, b_upp: Log-barrier training parameters.

## Outputs

After training, results are saved to the results/ directory:
- saved_models/: Contains model checkpoints (.pth), loss logs (losses.npy), and evaluation metrics (metrics.npy).
- Images: If visualization is enabled, plots of intensity evolution are saved here.
- Sampled Data: out-of-sample data generated from the learned model (model_generated_data.npy).
