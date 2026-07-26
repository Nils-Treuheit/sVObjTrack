# 2D to 3D — DONE
use MetaDepth/HYDEN-MoGe_v2 ( hugging_face: [https://huggingface.co/facebook/hyden-da2-relative-depth, https://huggingface.co/facebook/hyden-mogev2-metric-point, https://huggingface.co/facebook/hyden-mogev2-surface-normal], github: https://github.com/facebookresearch/metadepth, paper: https://openreview.net/pdf?id=2eL6yXLCh8 ) to create 3D object detections out of 2D object detections when only supplied with the webcam image stream

Implemented in `depth_ros2` package. Publishes `/camera/depth/image_raw` and `/camera/camera_info` for use by YOLO node's existing 3D projection pipeline.

# Zero-Shot Segment Anything — DONE
use SAM 3.1 (hugging_face: https://huggingface.co/facebook/sam3.1, github: https://github.com/facebookresearch/sam3, paper: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) to do live segmentation of almost everything in the scene with a huge dictonary of objects

Implemented in `sam3_ros2` package. Uses uv venv with Python 3.12 at `/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/sam3/.venv`. Subscribes to `/camera/image_raw` and `/yolo/detections_2d`, publishes `/sam3/debug_image` and `/sam3/masks`.

# Zero-Shot Object Detection
use YOLO-World with a huge dictonary of objects to detect almost everything in the scene

