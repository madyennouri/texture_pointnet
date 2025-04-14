#!/bin/bash
#SBATCH --job-name=texturenet_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=2                # Number of nodes to use
#SBATCH --ntasks-per-node=4      # Number of tasks per node
#SBATCH --cpus-per-task=4        # CPUs per task
#SBATCH --time=12:00:00          # Maximum execution time
#SBATCH --account=oanacazacu     # Project account
#SBATCH --partition=standard     # Partition/queue to use

# Load modules (adjust based on your environment)
module load python/3.8
module load pytorch/1.9.0-cuda11.1

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Evaluation configurations
DATA_PATH="data/ml_dataset.mat"
MODEL_PATH="results/distributed_training/model_best.pth"
OUTPUT_DIR="results/evaluation"
BATCH_SIZE=32
NUM_WORKERS=4

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

# Run the distributed evaluation
srun python -m parallel.distributed_evaluate \
    --data_path ${DATA_PATH} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --workers ${NUM_WORKERS} \
    --dist_url "tcp://${MASTER_ADDR}:${MASTER_PORT}" \
    --dist_backend nccl \
    --world_size ${WORLD_SIZE}

echo "Distributed evaluation completed!"