#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

source /opt/ros/humble/setup.bash
source "$DIR/.venv/bin/activate"

rm -rf "$DIR/build" "$DIR/install" "$DIR/log"
colcon build --symlink-install

# Fix hardcoded shebangs — replace system python path with env python
sed -i 's|#!/usr/bin/python3|#!/usr/bin/env python3|g' \
  "$DIR/install/yolo26_ros2/lib/yolo26_ros2/yolo_node" \
  "$DIR/install/camera_nodes/lib/camera_nodes/usb_camera" \
  "$DIR/install/camera_nodes/lib/camera_nodes/realsense_camera" 2>/dev/null || true

echo "Done. Run with:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source .venv/bin/activate"
echo "  source install/setup.bash"
echo "  ros2 run yolo26_ros2 yolo_node"
