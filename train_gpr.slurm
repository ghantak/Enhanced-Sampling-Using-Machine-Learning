#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=20:00:00
#SBATCH --job-name=test-gpr-par
#SBATCH --account=sschwartz
#SBATCH --qos=user_qos_sschwartz
#SBATCH --partition=gpu_standard #high_priority
#SBATCH --output=slurm-%j.o
#SBATCH --mail-type=ALL
#SBATCH --mail-user=krishnaghanta@arizona.edu

module load gnu8/8.3.0
module load python
#python train_gpr.py
#python train_gpr_parallel.py
python train_gpr_gpu.py

