#!/bin/bash
# Run the LocateAnything ROS2 node
# Activates sVObjTrack venv for numpy 1.x (cv_bridge compat) + torch
# TRT libraries loaded via LD_LIBRARY_PATH
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
LA_PROJECT="/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/locate_anything"
LA_TRT="$LA_PROJECT/model/tensorRT"

# 1. Activate sVObjTrack venv (numpy 1.x for cv_bridge, torch, transformers)
source "$DIR/.venv/bin/activate"

# 2. ROS2 Humble (rclpy, cv_bridge)
source /opt/ros/humble/setup.bash

# 3. TRT shared libraries (libnvinfer.so.10, libcudnn.so.9)
export LD_LIBRARY_PATH="$LA_TRT/.venv/lib/python3.10/site-packages/tensorrt_libs:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"

# 4. Colcon workspace
source "$DIR/install/setup.bash" 2>/dev/null || {
  echo "Run 'colcon build --symlink-install' first"
  exit 1
}

echo "[run_la_node] Python: $(which python3) $(python3 --version)"
echo "[run_la_node] VENV: $VIRTUAL_ENV"
python3 -c "import torch; print(f'[run_la_node] torch {torch.__version__} @ {torch.__file__}')" 2>/dev/null
echo "[run_la_node] Starting locate_anything_ros2 node..."
python3 -m locate_anything_ros2.la_node "$@"
