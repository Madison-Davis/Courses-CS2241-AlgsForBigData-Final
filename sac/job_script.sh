#!/bin/bash
#SBATCH --job-name=profile_job             # Job name
#SBATCH --output=profile_job_%j.out         # Standard output (%j expands to job ID)
#SBATCH --error=profile_job_%j.err          # Standard error
#SBATCH --partition=gpu                    # Partition name
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=4                   # Request 4 CPU cores
#SBATCH --time=02:00:00                     # Maximum run time

# Load required Python module.
module load python/3.12.5-fasrc01

# Change to the directory where your virtual environment and scripts are located.
cd /n/holylabs/LABS/janapa_reddi_lab/Lab/atschand/Courses-CS2241-AlgsForBigData-Final

# Activate the virtual environment.
# Update the path below if your virtual environment is located elsewhere.
source sac_env/bin/activate

cd sac

echo "Starting sac_baseline_profile.py"
python3 sac_baseline_profile.py

echo "Finished sac_baseline_profile.py; now starting plot_profiles.py"
python3 plot_profiles.py
