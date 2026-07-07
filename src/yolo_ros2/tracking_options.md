# Multi-Object Tracking Options for YOLO ROS2 Node

The YOLO node supports all Ultralytics YOLO tracking backends. Pass the tracker
config filename via the `bb_tracker` parameter.

## Supported Trackers

| Tracker | Config | Motion Model | Appearance / ReID | Camera Motion Comp. | Occlusion Handling |
|---------|--------|-------------|-------------------|---------------------|--------------------|
| **BoT-SORT** | `botsort.yaml` | Linear Kalman | Optional (`with_reid`) | Yes (sparseOptFlow / ECC) | Track buffer + ReID rebinding |
| **ByteTrack** | `bytetrack.yaml` | Linear Kalman | None | No | Two-stage low-conf rescue |
| **OC-SORT** | `ocsort.yaml` | Observation-centric Kalman | None | No | ORU, OCM, OCR |
| **Deep OC-SORT** | `deepocsort.yaml` | Observation-centric Kalman | Optional (`with_reid`) | Optional (`gmc_method`) | OC-SORT + adaptive appearance EMA |
| **FastTracker** | `fasttrack.yaml` | Linear Kalman + rollback | None | No | Kalman rollback + bbox enlargement |
| **TrackTrack** | `tracktrack.yaml` | Linear Kalman (NSA) | Optional (HMIoU fallback) | Yes (sparseOptFlow / ECC) | Iterative multi-cue + TAI |

### Which Tracker Should I Use?

1. **Need the fastest, simplest baseline?** → **ByteTrack** (no ReID, no camera-motion compensation, minimum overhead).
2. **Handheld, drone, or moving-camera footage?** → **BoT-SORT** (default; adds camera-motion compensation and optional ReID).
3. **Non-linear motion (sports, dancing, abrupt turns) and no ReID?** → **OC-SORT** (observation-centric corrections without appearance cost).
4. **Crowded moving-camera scenes where ID swaps are the main problem?** → **Deep OC-SORT** or **TrackTrack** (both add adaptive appearance fusion; TrackTrack also adds multi-cue association and duplicate-ID suppression).
5. **Frequent partial overlap in real-time, no ReID budget?** → **FastTracker** (occlusion-aware ByteTrack variant with Kalman rollback).

---

## Shared Tracker Arguments

Parameters common to most tracker YAML files.

| Parameter | Range | Description |
|-----------|-------|-------------|
| `tracker_type` | `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack`, `tracktrack` | Tracker backend |
| `track_high_thresh` | `0.0–1.0` | Threshold for first association |
| `track_low_thresh` | `0.0–1.0` | Threshold for second association (low-conf detections) |
| `new_track_thresh` | `0.0–1.0` | Threshold to initialize a new track |
| `track_buffer` | `≥0` | Frames lost tracks are kept alive before removal |
| `match_thresh` | `0.0–1.0` | Threshold for matching tracks |
| `fuse_score` | `True`, `False` | Fuse confidence scores with IoU distances |
| `gmc_method` | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Global motion compensation |
| `proximity_thresh` | `0.0–1.0` | Min IoU for valid ReID match |
| `appearance_thresh` | `0.0–1.0` | Min appearance similarity for ReID |
| `with_reid` | `True`, `False` | Enable appearance-based matching |
| `model` | `auto` or path to `.onnx`/`.engine` | ReID model |

---

## BoT-SORT

Default tracker. Extends ByteTrack with camera-motion compensation and optional ReID.

**Best for:** general-purpose tracking, especially moving cameras.

**Specific parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `gmc_method` | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Camera-motion-compensation backend (`sparseOptFlow` default) |
| `with_reid` | `True`, `False` | Enable appearance-based matching (off by default) |
| `model` | `auto` or path | ReID model |
| `proximity_thresh` | `0.0–1.0` | Min IoU before appearance features are considered |
| `appearance_thresh` | `0.0–1.0` | Min cosine similarity for ReID match |

**Tuning tips:**
- Static camera: set `gmc_method: none` to save a few ms/frame.
- Heavy camera motion: keep `sparseOptFlow`; `ecc` is more accurate but slower.
- Look-alike crowds: turn on `with_reid: True` and raise `appearance_thresh` (e.g. `0.85+`).

---

## ByteTrack

Lightweight baseline. Linear Kalman + IoU with two-stage association.

**Best for:** static or near-static cameras where you want minimum tracker overhead.

**Specific parameters:** None beyond shared arguments.

**Tuning tips:**
- Noisy detector: lower `track_low_thresh` so stage 2 has more candidates.
- High-recall detector: raise `track_high_thresh` to reduce fragmented IDs.
- Frequent ID flicker: raise `track_buffer` so briefly-missed tracks survive.

---

## OC-SORT

Observation-centric SORT. Adds ORU, OCM, OCR corrections.

**Best for:** non-linear motion without the cost of a ReID model.

**Specific parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `delta_t` | `≥1` | Temporal window for velocity-direction in OCM |
| `inertia` | `0.0–1.0` | Weight of velocity-consistency cost |
| `use_byte` | `True`, `False` | Enable ByteTrack-style second association |

**Tuning tips:**
- Non-linear motion: raise `inertia` (e.g. `0.3–0.4`).
- Sparse detections: enable `use_byte: True`.
- Long occlusions: raise `track_buffer`.

---

## Deep OC-SORT

OC-SORT + appearance + camera-motion compensation.

**Best for:** crowded or moving-camera scenes where ID swaps between visually similar objects are common.

**Specific parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `with_reid` | `True`, `False` | Enable appearance-based matching (off by default) |
| `model` | `auto` or path | ReID model |
| `proximity_thresh` | `0.0–1.0` | Min IoU before appearance features are considered |
| `appearance_thresh` | `0.0–1.0` | Min cosine similarity for ReID match |
| `alpha_fixed_emb` | `0.0–1.0` | Base EMA factor for track-embedding updates |
| `gmc_method` | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Global motion compensation |
| `delta_t` | `≥1` | Temporal window for OCM (inherited from OC-SORT) |
| `inertia` | `0.0–1.0` | Velocity-consistency cost weight (inherited) |
| `use_byte` | `True`, `False` | ByteTrack-style second association (inherited) |

**Tuning tips:**
- ID swaps in crowds: raise `appearance_thresh` (e.g. `0.92–0.95`), lower `alpha_fixed_emb`.
- Moving camera: set `gmc_method: sparseOptFlow` (defaults to `none`).
- Lower latency: keep `with_reid: False` (default); enable only when ID swaps dominate.

---

## FastTracker

Occlusion-aware ByteTrack variant. No appearance model.

**Best for:** real-time detection-only pipelines with frequent target-on-target overlap (crowds, queues, sports).

**Specific parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `reset_velocity_offset_occ` | `≥0` | History frames to restore Kalman velocity on occlusion |
| `reset_pos_offset_occ` | `≥0` | History frames to restore Kalman position on occlusion |
| `enlarge_bbox_occ` | `≥1.0` | Height scaling for predicted bbox while occluded |
| `dampen_motion_occ` | `0.0–1.0` | Velocity multiplier while occluded |
| `active_occ_to_lost_thresh` | `≥1` | Max consecutive occluded frames before moving to lost |
| `occ_cover_thresh` | `0.0–1.0` | Fraction of track area covered by another track to declare occlusion |
| `occ_reappear_window` | `≥0` | Frames a recently-occluded lost track stays re-findable |
| `init_iou_suppress` | `0.0–1.0` | Suppress new-track init if IoU with active track exceeds this |

**Tuning tips:**
- Frequent partial occlusions: lower `occ_cover_thresh` (e.g. `0.5–0.6`).
- Duplicate IDs around overlap: lower `init_iou_suppress` (e.g. `0.5`).
- Long occlusions: raise `occ_reappear_window` and `track_buffer` together.
- Fast-moving targets: raise `dampen_motion_occ` (closer to `1.0`) and lower `enlarge_bbox_occ`.

---

## TrackTrack

Track-perspective-based association with multi-cue iterative matching.

**Best for:** crowded scenes with frequent occlusion where duplicate IDs are a problem.

**Specific parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `iou_weight` | `0.0–1.0` | Weight of HMIoU distance in cost matrix |
| `reid_weight` | `0.0–1.0` | Weight of cosine ReID distance |
| `conf_weight` | `0.0–1.0` | Weight of confidence-projection distance |
| `angle_weight` | `0.0–1.0` | Weight of corner-angle distance |
| `penalty_p` | `0.0–1.0` | Cost penalty for low-confidence detections |
| `penalty_q` | `0.0–1.0` | Cost penalty for detections recovered by secondary NMS |
| `reduce_step` | `0.0–1.0` | Match-threshold relaxation per iteration |
| `tai_thr` | `0.0–1.0` | IoU threshold for Track-Aware Initialization NMS |
| `min_track_len` | `≥0` | Min successful updates before track is confirmed |
| `lost_match_thr` | `0.0–1.0` | Looser cost gate for relaxed lost-rebind pass |
| `with_reid` | `True`, `False` | Enable cosine-ReID appearance matching |
| `model` | `auto` or path | ReID model |
| `gmc_method` | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Global motion compensation |

**Tuning tips:**
- Crowded pedestrians: lower `tai_thr` (e.g. `0.45`); raise `track_buffer`.
- Fast camera motion: keep `gmc_method: sparseOptFlow`.
- Small/fast objects: raise `angle_weight`, lower `min_track_len`.
- Enable ReID only when needed (adds inference cost).

---

## Enabling Re-Identification (ReID)

Set `with_reid: True` in a tracker config. ReID model options:

- **`model: auto`** — uses native YOLO detector features (minimal overhead).
- **Exported model** — point `model:` at `.onnx` / `.engine` / `.torchscript` for stronger embeddings.

Ready-to-use ONNX encoders (auto-download on first use):

| Model | size (pixels) | params (M) | FLOPs (B) |
|-------|--------------|------------|-----------|
| `yolo26n-reid.onnx` | 448 | 2.8 | 2.0 |
| `yolo26s-reid.onnx` | 448 | 7.5 | 6.6 |
| `yolo26m-reid.onnx` | 448 | 12.4 | 20.1 |
| `yolo26l-reid.onnx` | 448 | 15.3 | 25.2 |
| `yolo26x-reid.onnx` | 448 | 32.7 | 55.9 |

---

## Config Files

The tracker YAML files are bundled with ultralytics:
- [`botsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- [`bytetrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)
- [`ocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/ocsort.yaml)
- [`deepocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/deepocsort.yaml)
- [`fasttrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/fasttrack.yaml)
- [`tracktrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/tracktrack.yaml)

Copy one of these to a custom name, edit parameters (except `tracker_type`), then
pass the custom yaml path to `bb_tracker` in the ROS2 node.
