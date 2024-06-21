#!/bin/sh
#
#SBATCH --job-name="bad multi Overijssel"
#SBATCH --account=education-tpm-msc-epa
#SBATCH --mail-user=y.hiensch@student.tudelft.nl
#SBATCH --mail-type=all

#SBATCH --time=03:58:00  # Adjust as needed
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4GB

module load 2023r1
module load openmpi
module load python
module load py-numpy
module load py-scipy
module load py-mpi4py
module load py-pip

pip install --user --upgrade ema_workbench
pip install --user networkx
pip install --user openpyxl
pip install --user xlrd

# Run the Python script
python EXPL_Multi_MORDM_Overijssel.py
mpiexec -n 1 python3 EXPL_Multi_MORDM_Overijssel.py

# Deactivate the virtual environment
deactivate


