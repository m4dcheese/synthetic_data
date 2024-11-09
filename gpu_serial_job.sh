#!/bin/bash
#SBATCH --account=def-mijungp
#SBATCH --gpus-per-node=1
#SBATCH --mem=4000M
#SBATCH --time=1:0:0

source ../env/bin/activate
python main.py >> output.txt
