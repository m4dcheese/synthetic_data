#!/bin/bash
#SBATCH --account=def-mijungp
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=5:0:0
#SBATCH --array=0-2 # Array of 3 jobs (0, 1, 2)

# Define arguments for each job
ARGS=(0, 1, 2)

# Load the appropriate environment
source ../env/bin/activate

# Get the argument corresponding to the current array task
CURRENT_ARG=${ARGS[$SLURM_ARRAY_TASK_ID]}

# Execute the Python script with the current argument
python main.py --run "$CURRENT_ARG"
