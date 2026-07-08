#!/bin/bash
# Run the Orchestrator node.
#
# Spawns YOLO detection (yolo26 + yolo26-obb + yolo26-pose by default)
# and forwards text queries to LocateAnything.
#
# Usage:
#   ./run_orchestrator_node.sh                          # default fusion
#   ./run_orchestrator_node.sh yolo11                   # single model
#   ./run_orchestrator_node.sh '[yolo26, cubified]'     # custom fusion
#   ./run_orchestrator_node.sh '[yolo26, yolo11-pose]'  # AABB + pose fusion
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Activate sVObjTrack venv (numpy 1.x, torch, rclpy)
source "$DIR/.venv/bin/activate"

# 2. ROS2 Humble
source /opt/ros/humble/setup.bash

# 3. Colcon workspace
source "$DIR/install/setup.bash" 2>/dev/null || {
  echo "Run 'colcon build --symlink-install' first"
  exit 1
}

echo "[run_orchestrator] Python: $(which python3) $(python3 --version)"
echo "[run_orchestrator] Starting orchestrator_node..."

# If first arg looks like a model shortcut, forward it as model_id parameter
if [ $# -gt 0 ] && [[ "$1" != "--ros-args" ]] && [[ "$1" != "-p" ]]; then
  echo "[run_orchestrator] Model: $1"
  python3 -m orchestrator.orchestrator_node --ros-args -p "model_id:='$1'"
else
  python3 -m orchestrator.orchestrator_node "$@"
fi
