#!/bin/bash
#SBATCH --job-name=texturenet_tune
#SBATCH --output=logs/tuning_%j.out
#SBATCH --error=logs/tuning_%j.err
#SBATCH --nodes=16               # Number of nodes to use
#SBATCH --ntasks=16              # Total number of tasks
#SBATCH --time=72:00:00          # Maximum execution time
#SBATCH --account=oanacazacu     # Project account
#SBATCH --partition=standard     # Partition/queue to use

# Load modules (adjust based on your environment)
module load python/3.8
module load pytorch/1.9.0-cuda11.1
module load mpi4py/3.0.3

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Configuration
DATA_PATH="data/ml_dataset.mat"
OUTPUT_DIR="results/hypertuning"
TUNING_EPOCHS=30

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Run the distributed hyperparameter tuning using MPI
mpirun -n ${SLURM_NTASKS} python -m parallel.distributed_hypertuning \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --tuning_epochs ${TUNING_EPOCHS}

echo "Hyperparameter tuning completed!"