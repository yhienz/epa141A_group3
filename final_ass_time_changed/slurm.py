#!/bin/sh
#
#SBATCH --job-name="Policy_MP"
#SBATCH --account=education-tpm-ms-epa
#SBATCH --mail-user=y.hiensch6@student.tudelft.nl
#SBATCH --mail-type=all

#SBATCH --partition=compute
#SBATCH --time=
#SBATCH --ntasks=
#SBATCH --cpus-per-task=4
#SBATCH --mem-percpu=4G

module load 2023r1
module load openmpi
module load python
module py-numpy
module py-scipy
module py-mpi4py
module py-pip

python -m venv venv
source venv/bin/activate
python -m pip install --upgrade ema_workbench
python -m pip install ipyparallel
python -m pip install networkx
python -m pip install openpyxl
python -m pip install xlrd

python MORDM.py
# mpiexec -n 1 python3 MORDM.py