# Object Detection 
Simple Ros2 project to detect and track objects and humans video feed of 2D RGB WebCam or Intel RealSense DepthCam
Real-Time Detection and Tracking based on YOLO11 and ByteTrack

## Installation 
```bash
git clone git@github.com:Nils-Treuheit/sVObjTrack.git
cd sVObjTrack
sr2  # System ROS 2 source
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install pyrealsense2
uv pip install ultralytics opencv-python
uv pip install 'numpy<2'
# on my system this was required because of a systemwide installed matplotlib
uv pip uninstall matplotlib
pip uninstall matplotlib
colcon build --cmake-args -DPYTHON_EXECUTABLE=$(which python) --symlink-install
# sometimes you have to use the following instead
uv pip install "setuptools<64.0.0"
rm -rf build/ install/ log/
python3 -m colcon build --symlink-install
```

## How to Use
Start terminals with:
```bash
sr2  # System ROS 2 source
source .venv/bin/activate
source install/setup.bash
```

First Terminal:
```bash
ros2 run yolo11_ros2 yolo_node
```

Second Terminal:
```bash
ros2 run camera_nodes usb_camera
```
or
```bash
ros2 run camera_nodes realsense_camera
```

Third Terminal (watch 2D object tracking in real-time):
```bash
ros2 run rqt_image_view rqt_image_view
```

Fourth Terminal (inspect topics, output 3D detection stream):
```bash
ros2 topic list
ros2 topic echo /yolo/detections_3d
```