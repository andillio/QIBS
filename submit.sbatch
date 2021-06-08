#!/bin/bash
#SBATCH --job-name=QAnd
#SBATCH --time=1-00:00:00
#SBATCH --output=QAnd
#SBATCH -p kipac,hns,iric,normal
#SBATCH --ntasks=200
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1

conda init bash
conda activate

module load openmpi

cd /oak/stanford/orgs/kipac/users/pizza/QuantumStuff/QuantumAndrewnomics
mpirun -np 200 --map-by core /oak/stanford/orgs/kipac/users/pizza/conda/bin/python coherentSimMP.py
