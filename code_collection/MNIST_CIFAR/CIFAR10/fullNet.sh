#!/bin/bash
#SBATCH --job-name=hm1       # Job name
#SBATCH --output=jupyter_output_%j.log   # Standard output and error log (%j is job ID)
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=64G                        # Total memory per node
#SBATCH --gres=gpu:A6000:1               # Request 1 A6000 GPU
#SBATCH --partition=long                 # Partition queue name

# Load necessary modules
module load conda/24.1.2 cuda/cuda12.4

# Activate the Conda environment
source activate dl-gpu

# # Printing some info
# echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
# echo "Home directory: ${HOME}"
# echo "Working directory: $PWD"
# echo "Current node: ${SLURM_NODELIST}"

# for debugging
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python '/home/harikrishnam/dv/the_project/zero0.3/CIFAR10/fullNet.py'