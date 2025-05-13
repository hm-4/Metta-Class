#!/bin/bash
#SBATCH --job-name=mgen         # Job name
#SBATCH --output=jupyter_vscode_%j.log   # Standard output log
#SBATCH --error=jupyter_vscode_%j.err    # Standard error log
#SBATCH --partition=long                  # Partition name (adjust as needed)
#SBATCH --time=24:00:00                  # Time limit
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPUs per task
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem=64GB                       # Memory per node

# Load necessary modules
source /etc/profile
module load conda/24.1.2 cuda/cuda12.4

# Activate your Conda environment
source activate dl-gpu  # Replace with your Conda environment name
echo $PATH
which python
which jupyter

# Find an available port
PORT=$(shuf -i 8000-8999 -n 1)

# Print SSH tunneling instructions
echo "To connect to the Jupyter Notebook, run this command on your local machine:"
echo "ssh -L $PORT:$(hostname):$PORT $USER@10.24.56.235"
echo "Then use this URL in VSCode or your browser: http://localhost:$PORT"

# Start Jupyter Notebook
jupyter notebook --no-browser --ip=0.0.0.0 --port=$PORT