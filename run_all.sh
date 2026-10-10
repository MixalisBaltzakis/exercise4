#!/bin/bash
SIM_MODE=${1:-0}
source ./venv/bin/activate
python3 make_video_fast.py $SIM_MODE
python3 make_powerflow.py $SIM_MODE
