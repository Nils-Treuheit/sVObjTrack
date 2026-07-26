#!/bin/bash
# Run the RF-DETR ROS2 node
#
# Usage:
#   ./run_rfdetr_node.sh                    # default: nano
#   ./run_rfdetr_node.sh small              # RFDETRSmall
#   ./run_rfdetr_node.sh medium             # RFDETRMedium
#   ./run_rfdetr_node.sh large              # RFDETRLarge
#   ./run_rfdetr_node.sh --ros-args -p model_size:=medium
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
RFDETR_VENV="/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/rfdetr/.venv"

# 1. Activate sVObjTrack venv (for rclpy, cv_bridge, etc.)
source "$DIR/.venv/bin/activate"

# 2. ROS2 Humble
source /opt/ros/humble/setup.bash

# 3. Colcon workspace
source "$DIR/install/setup.bash" 2>/dev/null || {
  echo "Run 'colcon build --symlink-install' first"
  exit 1
}

# 4. Add RF-DETR venv to PYTHONPATH so the node can import rfdetr
export PYTHONPATH="$RFDETR_VENV/lib/python3.10/site-packages:$PYTHONPATH"

echo "[run_rfdetr_node] Python: $(which python3) $(python3 --version)"
echo "[run_rfdetr_node] Starting rfdetr_ros2 node..."

# If first arg looks like a model shortcut, convert to ROS2 param
if [ $# -gt 0 ] && [[ "$1" != "--ros-args" ]] && [[ "$1" != "-p" ]]; then
  echo "[run_rfdetr_node] Model size: $1"
  python3 -m rfdetr_ros2.rfdetr_node --ros-args -p "model_size:='$1'"
else
  python3 -m rfdetr_ros2.rfdetr_node "$@"
fi
