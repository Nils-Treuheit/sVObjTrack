# sVObjTrack — ROS2 Object Detection, Tracking & Visual Grounding

Real-time 2D/3D detection and tracking using YOLO (11/26) and LocateAnything
with TensorRT-accelerated vision encoder. Supports AABB, OBB, pose keypoint
estimation, multi-object tracking, and visual grounding — with optional depth
fusion for 3D when using Intel RealSense.

## Quick Start

```bash
# Terminal 1: Camera
ros2 run camera_nodes usb_camera          # USB webcam
# or
ros2 run camera_nodes realsense_camera    # Intel RealSense (publishes depth + camera_info)
# or launch with arg (0=USB, 1=RealSense):
ros2 launch camera_nodes cameras.launch.py 0

# Terminal 2: Orchestrator — manages YOLO + LA internally
./run_orchestrator_node.sh                          # default: yolo26 + yolo26-obb + yolo26-pose
./run_orchestrator_node.sh yolo11                   # single model
./run_orchestrator_node.sh '[yolo26, cubified]'     # custom fusion

# Terminal 3: Watch detections
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
Standalone LA node (for debugging). Normally managed by the orchestrator
in query-only mode. Activates venv + ROS2 + sets `LD_LIBRARY_PATH` for TRT.
Runs LocateAnything-3B with vision encoder via TensorRT EP (~10ms/frame).
Subscribes to `/camera/image_raw`, cycles through object classes from
`config/la_objects.json` (one class per `interval_frames`), drawing
bounding boxes with class labels on `/la/debug_image` and publishing
structured detections to `/la/detections_2d` and `/la/detections_3d`
(with depth). Free-text grounding results are published to
`/la/grounding_text` only when triggered via `/la/grounding_query`.

```
./run_la_node.sh
```

To run in query-only mode (no detection cycle):
```bash
./run_la_node.sh --ros-args -p query_only:=True
```

To use a custom object list:
```bash
./run_la_node.sh --ros-args -p objects_file:=my_objects.json
```

### `run_la_grounding.sh`
One-shot ad-hoc visual grounding query against a running LA node.
Pass the object(s) you want to find — the query is sent to
`/la/grounding_query` and the response is printed (text on
`/la/grounding_text`, boxes drawn on `/la/debug_image`).

Find objects:
```
./run_la_grounding.sh "human plant chair"          # find all humans, plants, chairs
./run_la_grounding.sh "car"                        # find all cars
```

Referring / relational (attribute-based selection):
```
./run_la_grounding.sh "select the cube with a 2"             # object with specific attribute
./run_la_grounding.sh "find the person wearing a red hat"    # relational
./run_la_grounding.sh "the cube that has the letter A on it" # attribute-based
./run_la_grounding.sh "plant closest to the person"          # spatial relation
```

OCR (read text from objects):
```
./run_la_grounding.sh "read the text on the cube"
./run_la_grounding.sh "find all text in this scene"
./run_la_grounding.sh "read what it says"
```

GUI-style (pointing / clicking):
```
./run_la_grounding.sh "select the submit button"
./run_la_grounding.sh "click the red icon in the top right"
./run_la_grounding.sh "point to the settings menu"
```

### `run_orchestrator_node.sh`
Manages both YOLO detection and LocateAnything as internal subprocesses.
YOLO runs continuous online detection (default fusion: `[yolo26, yolo26-obb,
yolo26-pose]`). LA runs in **query-only mode** (no detection cycle) — it
only responds to text queries on `/la/grounding_query`. Listens for queries
on `/orchestrator/query`.

Query with the Python wrapper:

```bash
# Terminal 1: start orchestrator (manages YOLO + LA internally)
./run_orchestrator_node.sh                          # default fusion
./run_orchestrator_node.sh yolo11                   # single model override
./run_orchestrator_node.sh '[yolo26, cubified]'     # custom fusion override

# Terminal 2: query it
./run_orchestrator_query.sh "how many people are in the scene?"
./run_orchestrator_query.sh "find the red cup"
./run_orchestrator_query.sh "read the text on the cube"
./run_orchestrator_query.sh "select the cube with a 2"
./run_orchestrator_query.sh "plant closest to the person"
```

The orchestrator forwards the query to `/la/grounding_query`, waits for LA's
response on `/la/grounding_text`, and publishes the enriched result (LA answer
+ YOLO scene count) to `/orchestrator/response`.

Model selection works the same as `run_yolo_node.sh` (single shortcut or
`[model_a, model_b, ...]` fusion list). Both subprocesses are killed on
orchestrator shutdown.

See `run_la_grounding.sh` for more query patterns (referring, relational,
OCR, GUI-style).

### `run_yolo_node.sh`
Same environment for YOLO detection (11/26, pose). Multi-object tracking via
ByteTrack at the YOLO level + centroid nearest-neighbor matching. Pose
models draw COCO 17-keypoint skeleton on the debug image.

Single model:
```
./run_yolo_node.sh                # default: YOLO26 AABB
./run_yolo_node.sh yolo11         # YOLO11
./run_yolo_node.sh cubified       # Oriented bounding boxes
./run_yolo_node.sh yolo11-pose    # YOLO11 pose (keypoints)
./run_yolo_node.sh yolo26-pose    # YOLO26 pose (keypoints)
```

Model fusion (pass as comma-separated list inside brackets):
```
./run_yolo_node.sh '[yolo26, cubified]'                   # AABB + OBB fusion
./run_yolo_node.sh '[yolo26, yolo11-pose]'                # AABB + pose fusion
./run_yolo_node.sh '[yolo26, cubified, yolo11-pose]'      # AABB + OBB + pose fusion
```

Fusion runs all models on every frame and applies cross-model NMS:
- OBB preferred over AABB when they hit the same object
- Most confident detection kept within each type (AABB, OBB)
- Pose keypoints merged into the winning box from the best pose detection

### Launch file
```
ros2 launch camera_nodes cameras.launch.py 0   # USB webcam
ros2 launch camera_nodes cameras.launch.py 1   # Intel RealSense
```

## Models

Detection:
- YOLO11 AABB: `models/yolo11m.pt`
- YOLO26 AABB: `models/yolo26m.pt`
- YOLO Cubified OBB: `models/yolo_cubified.pt`

Pose (download separately from ultralytics):
- YOLO11 Pose: `models/yolo11m-pose.pt`
- YOLO26 Pose: `models/yolo26m-pose.pt`

VLM:
- LocateAnything-3B: loaded from `/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/locate_anything/`
- TRT engines: cached in that project's `model/tensorRT/engines/` (fp16: 12ms, fp32: 50ms)

## Topics

| Topic | Type | Description | Published By |
|-------|------|-------------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera frames | camera_nodes |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth image (RealSense only) | realsense_camera |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics (RealSense only) | realsense_camera |
| `/camera/source_info` | `std_msgs/String` | Camera source identifier | camera_nodes |
| `/yolo/debug_image` | `sensor_msgs/Image` | YOLO annotated frames (boxes + keypoints) | yolo_node |
| `/yolo/detections_2d` | `vision_msgs/Detection2DArray` | YOLO 2D bounding boxes | yolo_node |
| `/yolo/detections_3d` | `vision_msgs/Detection3DArray` | YOLO 3D detections (with depth) | yolo_node |
| `/yolo/object_point` | `geometry_msgs/PointStamped` | YOLO 3D center point | yolo_node |
| `/la/debug_image` | `sensor_msgs/Image` | LA annotated output frames | la_node |
| `/la/detections_2d` | `vision_msgs/Detection2DArray` | LA 2D bounding boxes | la_node |
| `/la/detections_3d` | `vision_msgs/Detection3DArray` | LA 3D detections (with depth) | la_node |
| `/la/object_point` | `geometry_msgs/PointStamped` | LA 3D center point | la_node |
| `/la/grounding_text` | `std_msgs/String` | LA visual grounding results | la_node |
| `/la/grounding_query` | `std_msgs/String` | Ad-hoc grounding query (input) | any node |
| `/orchestrator/query` | `std_msgs/String` | Text query to orchestrator (input) | any node |
| `/orchestrator/response` | `std_msgs/String` | Enriched LA query response | orchestrator_node |

Each topic only exists when its publishing node is loaded:
- YOLO topics are dead when no `yolo_node` is running
- LA topics are dead when no `la_node` is running
- 3D topics are dead with USB camera (no depth source)
- Orchestrator topics are dead when no `orchestrator_node` is running

## Parameters

### la_node
- `objects_file` (default: `config/la_objects.json`) — JSON file with objects to detect (`{"objects": ["person", ...], "ocr": ["A", "B", ...]}`). Detection uses the full list in a single query — one inference per interval detects all listed object types at once. The `ocr` list is for reference only (used in grounding queries, not detection).
- `interval_frames` (default: `15`) — frames between LA inferences
- `conf_threshold` (default: `0.3`)
- `max_new_tokens` (default: `128`) — token limit for detection queries (output is parsed for `<box>` tags only)
- `grounding_max_new_tokens` (default: `1024`) — longer token limit for ad-hoc grounding queries (free-text response published to `/la/grounding_text`)
- `debug` (default: `true`)

**Detection** uses one comprehensive query listing all objects:  
`"Detect all of the following in this scene: person, monitor, pc, robot, ... For each one output <box>x1,y1,x2,y2</box> class_name."`

**OCR** (reading text on cubes) is done via the grounding query — not part of the detection cycle:  
`./run_la_grounding.sh "read the text on the cube"`

Ad-hoc grounding is done via topic:
```bash
ros2 topic pub -1 /la/grounding_query std_msgs/msg/String "data: 'human plant chair'"
# or use the wrapper:
./run_la_grounding.sh "human plant chair"
```

### yolo_node
- `model_id` — path or shortcut: `yolo11`/`yolo26`/`cubified`/`yolo11-pose`/`yolo26-pose`
  - Fusion: `[yolo26, cubified]` / `[yolo26, yolo11-pose]` / `[yolo26, cubified, yolo11-pose]`
- `model_type` — `AABB` or `OBB` or `pose` or `[AABB, OBB, pose]` for fusion
- `bb_tracker` — tracker config per model (default: `bytetrack.yaml`), e.g. `[bytetrack.yaml, bytetrack.yaml]` for fusion
- `conf_threshold` (default: 0.4)

## Tracker Selection

The YOLO node supports six tracking backends. Pick one by setting `bb_tracker`:

```bash
./run_yolo_node.sh --ros-args -p bb_tracker:=bytetrack.yaml
./run_yolo_node.sh --ros-args -p bb_tracker:=botsort.yaml
./run_yolo_node.sh --ros-args -p bb_tracker:=ocsort.yaml
```

| Scenario | Recommended Tracker |
|----------|-------------------|
| Fastest, simplest baseline | `bytetrack.yaml` |
| Moving/handheld camera | `botsort.yaml` (default) |
| Non-linear motion (sports, abrupt turns) | `ocsort.yaml` |
| Crowded moving-camera, ID swaps are main issue | `deepocsort.yaml` or `tracktrack.yaml` |
| Frequent partial overlap, no ReID budget | `fasttrack.yaml` |

For full parameter descriptions, tuning tips, and ReID setup see
[`src/yolo_ros2/tracking_options.md`](src/yolo_ros2/tracking_options.md).

In fusion mode, each model can use a different tracker:
```bash
./run_yolo_node.sh '[yolo26, cubified]' \
  --ros-args -p bb_tracker:="[bytetrack.yaml, botsort.yaml]"
```

## Notes

- **Depth camera topics** (`/camera/depth/image_raw`, `/camera/camera_info`)
  only appear when the camera source is `realsense`. USB cameras don't
  advertise depth topics.
- **Multi-object tracking**: YOLO returns all detections above threshold;
  each gets a persistent track ID via centroid nearest-neighbor matching.
- **Architecture**: YOLO nodes handle real-time online detection (AABB, OBB, pose). The LA node serves as a **query engine** for grounding, counting, OCR, and text-driven object selection — running inference only when queried via `/la/grounding_query`. The orchestrator node bridges the two: it subscribes to YOLO detections for scene context, forwards text queries to LA, and publishes enriched responses.
- **OCR on cubes**: Not part of the detection cycle. Run via the grounding module:
  ```bash
  ./run_la_grounding.sh "read the text on the cube"
  ./run_la_grounding.sh "find all text in this scene"
  ./run_la_grounding.sh "read what it says on the cube"
  ```
  LA returns `<ref>text</ref><box>x1,y1,x2,y2</box>` for each text region — the recognized text becomes the detection label, and boxes are drawn on the debug image with center points. The `ocr` list in `config/la_objects.json` documents which characters to look for.
- **Pose estimation**: YOLO pose models draw COCO 17-keypoint skeleton with
  colored joints and limb connections on the debug image.
- **3D projection**: YOLO and LA both project bounding box centers to 3D
  using RealSense depth. Published as `Detection3DArray` and `PointStamped`.
- **TRT engines** are SM120-specific (RTX 5090). Rebuild for other GPUs:
  `cd /path/to/locate_anything && python model/tensorRT/convert_onnx_to_trt.py`
- **sVObjTrack venv** has numpy 1.x installed for cv_bridge ABI
  compatibility (ROS2 Humble's cv_bridge was compiled against numpy 1.x).
- **Performance**: system/user-local torch (~5s import) vs venv torch
  (~90s import from HDD). The launcher activates the venv for
  numpy/cv_bridge/TRT; torch imports from the venv (slower start, but
  fine for long-running nodes).
