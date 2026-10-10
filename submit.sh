#!/bin/bash

CU_FILE=${1:-video_pml.cu}
PY_FILE=${2:-run_all.sh}
SIM_MODE=${3:-0}

mkdir -p outputs errors video_H video_E frames_e frames_h video_S frames_s

# Find the next available ID based on output files
NEXT_ID=1
for f in outputs/output_*.txt; do
    if [ -e "$f" ]; then
        ID=$(basename "$f" | tr -dc '0-9')
        if [ "$ID" -ge "$NEXT_ID" ]; then
            NEXT_ID=$((ID + 1))
        fi
    fi
done

echo "Submitting job with ID: $NEXT_ID for file: $CU_FILE, $PY_FILE, SIM_MODE=$SIM_MODE"
touch outputs/output_${NEXT_ID}.txt errors/error_${NEXT_ID}.txt
sbatch --output=outputs/output_${NEXT_ID}.txt --error=errors/error_${NEXT_ID}.txt run_cluster.sh $CU_FILE $PY_FILE $SIM_MODE
