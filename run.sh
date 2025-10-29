#!/bin/bash
# Launcher script for DexGraspMaker interactive gripper tool

# Set required environment variables for OpenGL
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# Activate conda environment and run the application
eval "$(conda shell.bash hook)"
conda activate dgm
python interactive_gripper_tool/main.py "$@"
