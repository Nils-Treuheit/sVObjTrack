# sVObjTrack — ROS2 Object Detection, Tracking & Visual Grounding

Real-time 2D/3D detection and tracking using YOLO (11/26) and LocateAnything
with TensorRT-accelerated vision encoder.

## Quick Start

```bash
# Terminal 1: Camera
ros2 run camera_nodes usb_camera          # USB webcam
# or
ros2 run camera_nodes realsense_camera    # Intel RealSense
# or launch with arg (0=USB, 1=RealSense):
ros2 launch camera_nodes cameras.launch.py 0

# Terminal 2: YOLO detection (multi-object tracking)
./run_yolo_node.sh

# Terminal 3: LocateAnything visual grounding (TRT-accelerated)
./run_la_node.sh

# Terminal 4: Watch detections
ros2 run rqt_image_view rqt_image_view
```

## Prerequisites

- ROS2 Humble
- CUDA 12.8 + TensorRT 10.16
- NVIDIA GPU with SM120+ (RTX 5090)

## Installation

```bash
git clone git@github.com:Nils-Treuheit/sVObjTrack.git
cd sVObjTrack
sr2                           # source ROS2
uv venv .venv                 # create venv
source .venv/bin/activate
uv pip install -r requirements.txt
cbp                           # colcon build (alias: python3 -m colcon build --symlink-install)
```

## Launcher Scripts (use `python3 -m` instead of `ros2 run`)

Launchers activate the sVObjTrack venv first to ensure numpy 1.x (required
for cv_bridge ABI compatibility) and to set up the TRT library path.

### `run_la_node.sh`
Activates venv + ROS2 + sets `LD_LIBRARY_PATH` for TRT. Runs
LocateAnything-3B with vision encoder via TensorRT EP (~10ms/frame).
Subscribes to `/camera/image_raw`, publishes to `/la/detections_2d`
and `/la/debug_image`.

```
./run_la_node.sh
```

### `run_yolo_node.sh`
Same environment for YOLO detection (11/26). Multi-object tracking via
ByteTrack at the YOLO level + centroid tracking.

```
./run_yolo_node.sh           # default: YOLO26 AABB
./run_yolo_node.sh yolo11    # YOLO11
./run_yolo_node.sh cubified  # Oriented bounding boxes
```

### Launch file
```
ros2 launch camera_nodes cameras.launch.py 0   # USB webcam
ros2 launch camera_nodes cameras.launch.py 1   # Intel RealSense
```

## Models

- YOLO11:  `models/yolo11m.pt`
- YOLO26:  `models/yolo26m.pt`
- Cubified: `models/yolo_cubified.pt`
- LocateAnything-3B: loaded from `/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/locate_anything/`
- TRT engines: cached in that project's `model/tensorRT/engines/` (fp16: 12ms, fp32: 50ms)

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera frames |
| `/la/debug_image` | `sensor_msgs/Image` | Annotated output frames |
| `/la/detections_2d` | `vision_msgs/Detection2DArray` | Bounding boxes |
| `/la/grounding_text` | `std_msgs/String` | LA grounding results |
| `/la/grounding_query` | `std_msgs/String` | Ad-hoc grounding query |
| `/yolo/detections_2d` | `vision_msgs/Detection2DArray` | YOLO boxes |
| `/yolo/debug_image` | `sensor_msgs/Image` | YOLO annotated frames |
| `/yolo/detections_3d` | `vision_msgs/Detection3DArray` | 3D detections (with depth) |

## Parameters

### la_node
- `detect_queries` (default: `["person","car","dog","chair"]`) — round-robin queries
- `interval_frames` (default: 30) — frames between LA inferences
- `conf_threshold` (default: 0.3)
- `max_new_tokens` (default: 32)
- `debug` (default: true)

### yolo_node
- `model_id` — path or `yolo11`/`yolo26`/`cubified` or `[yolo26, cubified]` for fusion
- `model_type` — `AABB` or `OBB` or `[AABB, OBB]` for fusion
- `bb_tracker` — tracker config (default: `bytetrack.yaml`)
- `conf_threshold` (default: 0.4)

## Notes

- **Depth camera topics** (`/camera/depth/image_raw`, `/camera/camera_info`)
  only appear when the camera source is `realsense`. USB cameras don't
  advertise depth topics.
- **Multi-object tracking**: YOLO returns all detections above threshold;
  each gets a persistent track ID via centroid nearest-neighbor matching.
- **TRT engines** are SM120-specific (RTX 5090). Rebuild for other GPUs:
  `cd /path/to/locate_anything && python model/tensorRT/convert_onnx_to_trt.py`
- **sVObjTrack venv** has numpy 1.x installed for cv_bridge ABI
  compatibility (ROS2 Humble's cv_bridge was compiled against numpy 1.x).
- **Performance**: system/user-local torch (~5s import) vs venv torch
  (~90s import from HDD). The launcher activates the venv for
  numpy/cv_bridge/TRT; torch imports from the venv (slower start, but
  fine for long-running nodes).
