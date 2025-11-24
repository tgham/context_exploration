#!/bin/bash -l

#SBATCH -J gp_worlds
#SBATCH -D /home/tgraham/GP_worlds/

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.graham@tuebingen.mpg.de


micromamba activate chickpeas

srun python -u ./expt_optimisation.py 
# srun python -u ./generate_seqs.py 