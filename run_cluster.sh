#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00

module load gcc/13.2.0-iqpfkya
module load cuda/12.4.0-zk32gam

CU_FILE=${1:-video_pml.cu}
PY_FILE=${2:-run_all.sh}
SIM_MODE=${3:-0}
EXE_NAME="${CU_FILE%.*}"

echo "Compiling $CU_FILE..."
nvcc $CU_FILE -o $EXE_NAME -O3

echo "Running $EXE_NAME with SIM_MODE=$SIM_MODE..."
./$EXE_NAME $SIM_MODE

echo "Running $PY_FILE with SIM_MODE=$SIM_MODE..."
source ../venv/bin/activate
if [[ "$PY_FILE" == *.sh ]]; then
    bash $PY_FILE $SIM_MODE
else
    python3 $PY_FILE $SIM_MODE
fi
echo "Done!"
