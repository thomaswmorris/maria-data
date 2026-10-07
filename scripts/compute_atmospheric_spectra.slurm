#!/bin/bash -l

#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=3:00:00
#SBATCH --job-name=maria
#SBATCH --output=/home/tm12/slurm/maria/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thomas.w.morris@yale.edu

set -e
export PYTHONUNBUFFERED=TRUE

conda activate sotorch

srun python /scratch/gpfs/SIMONSOBS/users/tm12/maria-data/scripts/compute_atmospheric_spectrum.py --tag v4
