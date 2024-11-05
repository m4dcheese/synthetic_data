#!/bin/bash

# Prompt the user to select the partition
echo "Select the partition you want to work on:"
echo "1 = dev_gpu_4_a100"
echo "2 = dev_single"
echo "3 = dev_gpu_4"
read -p "Enter the number (1, 2, or 3): " PARTITION_CHOICE

# Set the partition and time limit based on user choice
TIME_LIMIT="00:25:00"  # Default time limit; adjust per partition as needed

if [[ "$PARTITION_CHOICE" == "1" ]]; then
  PARTITION="dev_gpu_4_a100"
elif [[ "$PARTITION_CHOICE" == "2" ]]; then
  PARTITION="dev_single"
elif [[ "$PARTITION_CHOICE" == "3" ]]; then
  PARTITION="dev_gpu_4"
else
  echo "Error: Please enter a valid partition choice (1, 2, or 3)."
  exit 1
fi

# Prompt the user to enter the number of GPUs
read -p "Enter the number of GPUs you want to use: " NUM_GPUS

# Validate if the input is a number
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  echo "Error: Please enter a valid number for GPUs."
  exit 1
fi



# Run the srun command with the specified partition, number of GPUs, and number of CPUs
srun --partition=$PARTITION --gres=gpu:$NUM_GPUS  --time=$TIME_LIMIT --nodes=1 --ntasks=1 --pty bash 



