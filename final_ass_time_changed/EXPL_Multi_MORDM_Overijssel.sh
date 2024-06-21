#!/bin/sh
#
#SBATCH --job-name="bad multi Overijssel"
#SBATCH --account=education-tpm-msc-epa
#SBATCH --mail-user=y.hiensch@student.tudelft.nl
#SBATCH --mail-type=all

#SBATCH --time=02:58:00  # Adjust as needed
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

# Log the start time
echo "Job started at: $(date)"

# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install necessary Python packages
#python -m pip install --upgrade --force-reinstall numpy==1.26.4
python -m pip install --upgrade --force-reinstall numpy==1.23
python -m pip install --upgrade ema_workbench
python -m pip install ipyparallel
python -m pip install networkx
python -m pip install openpyxl
python -m pip install xlrd

# Ensure the destination directory exists
venv_python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
target_dir="venv/lib/python${venv_python_version}/site-packages/ema_workbench/em_framework"

# Create the target directory if it does not exist
mkdir -p $target_dir

# Copy the optimization.py file to the ema_workbench package
if [ -f "optimization.py" ]; then
    cp "optimization.py" "$target_dir/optimization.py"
    echo "File copied successfully to $target_dir"
else
    echo "File copy failed: optimization.py does not exist"
    exit 1
fi

# Run the Python script
python EXPL_Multi_MORDM_Overijssel.py
mpiexec -n 1 python3 EXPL_Multi_MORDM_Overijssel.py

# Deactivate the virtual environment
deactivate

# Log the end time
echo "Job ended at: $(date)"
