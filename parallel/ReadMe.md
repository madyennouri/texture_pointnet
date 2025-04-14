This repository contains a parallelized implementation of TexturePointNet, a deep learning approach for texture analysis using PointNet++ architecture. The code has been optimized to run efficiently on multi-node clusters using distributed training techniques.
Table of Contents

Installation
Parallelization Strategy
Directory Structure
Usage

Distributed Training
Distributed Hyperparameter Tuning
Distributed Evaluation


Advanced Configuration
Results and Logs
Troubleshooting

Installation
First, clone the repository:
bashCopygit clone https://github.com/madyennouri/texture_pointnet.git
cd texture_pointnet
Create a virtual environment (optional but recommended):
bashCopypython -m venv texture_env
source texture_env/bin/activate  # On Windows: texture_env\Scripts\activate
Install the required packages:
bashCopypip install -r requirements.txt
pip install mpi4py torch_xla
Parallelization Strategy
The code has been parallelized in three primary ways:

Distributed Data-Parallel Training: Using PyTorch's DistributedDataParallel (DDP) to train the model across multiple nodes/GPUs with synchronized gradient updates.
Distributed Hyperparameter Tuning: Using MPI to distribute hyperparameter combinations across multiple nodes, with each node evaluating different combinations independently.
Parallel Data Processing: Optimized data loading and preprocessing with multi-process workers and prefetching.

Directory Structure
Copytexture_pointnet/
├── models/                     # Model architecture definitions
│   ├── pointnet_modules.py     # PointNet++ modules
│   └── texture_pointnet.py     # Main model implementation
├── utils/                      # Utility functions
│   ├── data_utils.py           # Data loading and preprocessing
│   └── visualization.py        # Visualization utilities
├── parallel/                   # Parallelization components
│   ├── distributed_train.py    # Distributed training implementation
│   ├── distributed_hypertuning.py # Distributed hyperparameter tuning
│   ├── distributed_evaluate.py # Distributed model evaluation
│   ├── data_parallel.py        # Parallel data processing utilities
│   └── utils.py                # Distributed utilities
├── scripts/                    # Slurm job scripts
│   ├── run_distributed_train.sh # Script for distributed training
│   ├── run_hypertuning.sh      # Script for hyperparameter tuning
│   └── run_distributed_eval.sh # Script for distributed evaluation
├── data/                       # Data directory
│   └── ml_dataset.mat          # Dataset file
├── requirements.txt            # Package dependencies
└── README.md                   # This file
Usage
Distributed Training
To run distributed training on the cluster, use the provided job script:
bashCopysbatch scripts/run_distributed_train.sh
Or submit an interactive job:
bashCopy# Request interactive allocation
interactive -a oanacazacu -N 4 -n 16 -t 48:00:00

# Inside the interactive session
mkdir -p results/distributed_training logs
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=16

# Launch training with srun
srun -n 16 python -m parallel.distributed_train \
    --data_path data/ml_dataset.mat \
    --output_dir results/distributed_training \
    --batch_size 32 \
    --learning_rate 0.001 \
    --workers 4 \
    --epochs 100 \
    --dist_url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
    --dist_backend nccl \
    --world_size ${WORLD_SIZE}
Distributed Hyperparameter Tuning
To run distributed hyperparameter tuning on the cluster:
bashCopysbatch scripts/run_hypertuning.sh
Or in an interactive session:
bashCopy# Request interactive allocation
interactive -a oanacazacu -N 16 -n 16 -t 72:00:00

# Inside the interactive session
mkdir -p results/hypertuning logs

# Launch hyperparameter tuning with mpirun
mpirun -n 16 python -m parallel.distributed_hypertuning \
    --data_path data/ml_dataset.mat \
    --output_dir results/hypertuning \
    --tuning_epochs 30
Distributed Evaluation
To evaluate a trained model in parallel:
bashCopysbatch scripts/run_distributed_eval.sh
Or in an interactive session:
bashCopy# Request interactive allocation
interactive -a oanacazacu -N 2 -n 8 -t 12:00:00

# Inside the interactive session
mkdir -p results/evaluation logs
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=8

# Launch evaluation with srun
srun -n 8 python -m parallel.distributed_evaluate \
    --data_path data/ml_dataset.mat \
    --model_path results/distributed_training/model_best.pth \
    --output_dir results/evaluation \
    --batch_size 32 \
    --workers 4 \
    --dist_url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
    --dist_backend nccl \
    --world_size ${WORLD_SIZE}
Advanced Configuration
The scripts allow for various customizations:
Training Parameters

--batch_size: Batch size per process
--learning_rate: Base learning rate (scaled by world size)
--epochs: Number of training epochs
--dropout_rate: Dropout rate for regularization
--patience: Patience for early stopping
--workers: Number of data loading workers per process

Hyperparameter Tuning

--tuning_epochs: Number of epochs for each hyperparameter combination
--max_combinations: Maximum number of combinations to try

Evaluation

--model_path: Path to the trained model checkpoint
--batch_size: Batch size for evaluation
--workers: Number of data loading workers

Results and Logs
Training and evaluation logs are stored in the logs/ directory with job IDs as suffixes. Results, checkpoints, and visualizations are saved in their respective output directories.
Model Checkpoints
During training, the best model (based on validation loss) is saved as model_best.pth, and the latest checkpoint is saved as checkpoint.pth.
Training Curves
Learning curves are saved as learning_curves.png in the output directory, showing training and validation loss over epochs.
Evaluation Results
Evaluation metrics and predictions vs. true values are saved in the output directory specified in the evaluation command.
Troubleshooting
Common Issues

NCCL Timeouts: If experiencing timeouts during distributed training:

Increase timeout values with: export NCCL_SOCKET_IFNAME=eth0; export NCCL_DEBUG=INFO


Out of Memory: If encountering OOM errors:

Reduce batch size
Try gradient accumulation by adding --gradient_accumulation_steps 2 to the command


Load Balancing: If some nodes finish much earlier than others during hyperparameter tuning:

Implement a dynamic work-stealing approach with MPI



Debugging Distributed Training
To debug distributed training, set the following environment variables:
bashCopyexport NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
