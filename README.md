# Object Detection 
Simple Ros2 project to detect and track objects and humans video feed of 2D RGB WebCam or Intel RealSense DepthCam
Real-Time Detection and Tracking based on YOLO and ByteTrack

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
cbp
# sometimes you have to use the following instead
uv pip install "setuptools<64.0.0"
cbc # rm -rf build/ install/ log/
python3 -m colcon build --symlink-install
```

## How to Use
Start terminals with:
```bash
sr2      # source System ROS 2 
act_venv # source venv
sis      # source ROS 2 module install
```

First Terminal:
```bash
ros2 run yolo_ros2 yolo_node
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