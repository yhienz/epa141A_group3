#!/bin/sh
#
#SBATCH --job-name="Policy_MP_GRP3"
#SBATCH --account=education-tpm-msc-epa
#SBATCH --mail-user=y.hiensch@student.tudelft.nl
#SBATCH --mail-type=all

#SBATCH --time=00:02:00
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

python -m venv venv
source venv/bin/activate
python -m pip install --upgrade ema_workbench
python -m pip install ipyparallel
python -m pip install networkx
python -m pip install openpyxl
python -m pip install xlrd

python MORDM.py
mpiexec -n 1 python3 MORDM.py