#!/bin/bash
#SBATCH --job-name=run_notebook          # Job name
#SBATCH --output=notebook_output_%j.log  # Output log file
#SBATCH --error=notebook_error_%j.err    # Error log file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=16G                        # Memory per node
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=long                 # Partition name
#SBATCH --time=01:00:00                  # Time limit

# Load necessary modules
module load conda/24.1.2 cuda/cuda12.4

# Activate the Conda environment
source activate dl-gpu

# Debugging information (optional)
python --version
nvcc -V

# Convert and run the notebook
jupyter nbconvert --to notebook --execute /home/harikrishnam/dv/the_project/zero0.3/MNIST/m_full.ipynb --output executed1_m_full.ipynb.ipynb
