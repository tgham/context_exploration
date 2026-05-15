#!/bin/bash -l

#SBATCH -J gp_worlds
#SBATCH -D /home/tgraham/GP_worlds/

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.graham@tuebingen.mpg.de


# micromamba activate chickpeas

# srun python -u ./generate_seqs.py 
# srun python -u ./expt_optimisation.py 
# srun python -u ./bandit_expt.py 
# srun python -u ./gittins_expt.py --n_sims 100 --n_workers 50 --n_alphas 20 --n_samples 500000
# srun python -u ./bandit_single_search.py --n_sims 100 --n_workers 100 --max_beta 8 --n_samples 1000000 --exploration_constant 3.01

micromamba activate sbi_env
# srun python -u ./SBI/hpc/tesbi_e2e.py --stage ppc
# srun python -u ./SBI/hpc/tesbi_e2e.py --stage recover --K 50 --num_post 1000
srun python -u ./SBI/hpc/tesbi.py --stage all --density nsf --n_samples 30000


