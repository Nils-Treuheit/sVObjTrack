#!/bin/bash
# Run the YOLO ROS2 node
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Activate sVObjTrack venv
source "$DIR/.venv/bin/activate"

# 2. ROS2 Humble
source /opt/ros/humble/setup.bash

# 3. Colcon workspace
source "$DIR/install/setup.bash" 2>/dev/null || {
  echo "Run 'colcon build --symlink-install' first"
  exit 1
}

echo "[run_yolo_node] Python: $(which python3) $(python3 --version)"
echo "[run_yolo_node] Starting yolo_ros2 node..."
python3 -m yolo_ros2.yolo_node "$@"
