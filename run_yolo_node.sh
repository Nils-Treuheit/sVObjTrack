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

# If first arg looks like a model shortcut (not a --ros-args flag), convert it
if [ $# -gt 0 ] && [[ "$1" != "--ros-args" ]] && [[ "$1" != "-p" ]]; then
  echo "[run_yolo_node] Model: $1"
  # YAML-single-quote the value so ROS2 doesn't parse [fusion, list] as STRING_ARRAY
  python3 -m yolo_ros2.yolo_node --ros-args -p "model_id:='$1'"
else
  python3 -m yolo_ros2.yolo_node "$@"
fi
