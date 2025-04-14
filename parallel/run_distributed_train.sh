#!/bin/bash
#SBATCH --job-name=texturenet_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=4                # Number of nodes to use
#SBATCH --ntasks-per-node=4      # Number of tasks per node
#SBATCH --cpus-per-task=4        # CPUs per task
#SBATCH --time=48:00:00          # Maximum execution time
#SBATCH --account=oanacazacu     # Project account
#SBATCH --partition=standard     # Partition/queue to use

# Load modules (adjust based on your environment)
module load python/3.8
module load pytorch/1.9.0-cuda11.1

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Training configurations
DATA_PATH="data/ml_dataset.mat"
OUTPUT_DIR="results/distributed_training"
BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_WORKERS=4
NUM_EPOCHS=100

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Get the number of nodes and tasks
WORLD_SIZE=$SLURM_NTASKS
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"

# Run the distributed training on all nodes
srun python -m parallel.distributed_train \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --workers ${NUM_WORKERS} \
    --epochs ${NUM_EPOCHS} \
    --dist_url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
    --dist_backend nccl \
    --world_size ${WORLD_SIZE}

echo "Distributed training completed!"