"""
LocateAnything ROS2 node with TensorRT-accelerated vision encoder.

Provides visual grounding, object detection, pose estimation, and
multi-object tracking using NVIDIA's LocateAnything-3B VLM.

Topics:
  Subscribed:
    /camera/image_raw       (sensor_msgs/Image)   - Input camera frames
    /camera/source_info     (std_msgs/String)     - Camera source id
    /camera/depth/image_raw (sensor_msgs/Image)   - Depth image (optional)
    /camera/camera_info     (sensor_msgs/CameraInfo) - Camera intrinsics
    /la/grounding_query     (std_msgs/String)     - Ad-hoc visual grounding query

  Published:
    /la/debug_image         (sensor_msgs/Image)   - Annotated frames
    /la/detections_2d       (vision_msgs/Detection2DArray) - Bounding boxes
    /la/detections_3d       (vision_msgs/Detection3DArray) - 3D detections (with depth)
    /la/object_point        (geometry_msgs/PointStamped)   - 3D center point
    /la/grounding_text      (std_msgs/String)     - Visual grounding results

Detection queries run in a round-robin cycle. Each query runs LA on one frame,
results persist for subsequent frames until the next cycle updates them.
"""
import sys
import os
import time
import math
import re
import threading
from pathlib import Path
from collections import OrderedDict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, Detection3DArray, Detection3D
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
import json
from PIL import Image as PILImage

# cv_bridge may fail under numpy 2.x; guard both import and instantiation
_CV_BRIDGE_OK = False
try:
    from cv_bridge import CvBridge
    _CV_BRIDGE_OK = True
except Exception:
    pass

# LocateAnything imports — use absolute path (ROS2 workspace is separate from LA project)
LA_PROJECT = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/locate_anything")
sys.path.insert(0, str(LA_PROJECT))

try:
    from unified_engine import LocateAnythingEngine
    _LA_AVAILABLE = True
except Exception as e:
    print(f"[LA Node] Cannot import LocateAnythingEngine: {e}", file=sys.stderr)
    _LA_AVAILABLE = False

# Try TRT vision encoder — add TRT venv SP to END of sys.path (fallback)
# to avoid slow sympy/mpmath cascade from global PYTHONPATH
TRT_DIR = LA_PROJECT / "model" / "tensorRT"
TRT_VENV_SP = TRT_DIR / ".venv" / "lib" / "python3.10" / "site-packages"
if TRT_DIR.exists():
    sys.path.insert(0, str(TRT_DIR))
    if TRT_VENV_SP.exists():
        sys.path.append(str(TRT_VENV_SP))
    try:
        from trt_vision_encoder import TrtVisionEncoder
        _TRT_AVAILABLE = True
    except Exception:
        _TRT_AVAILABLE = False
else:
    _TRT_AVAILABLE = False


def parse_detections(text: str) -> list:
    """Extract (label, [x1,y1,x2,y2]) pairs from LA <box> output.

    Handles formats like:
      "<ref>label</ref><box>x1,y1,x2,y2</box>"   (OCR/grounding)
      "<box>x1,y1,x2,y2</box> person"             (detection)
      "a person <box>x1,y1,x2,y2</box>"
      "<box>x1,y1,x2,y2</box>"

    Returns list of (class_name, [x1, y1, x2, y2]  or  [x, y]).
    """
    results = []
    for match in re.finditer(r'<box>(.+?)</box>', text):
        coords_str = match.group(1)
        coords = [float(p) for p in re.findall(r'[\d.]+', coords_str)]
        if len(coords) not in (2, 4):
            continue

        # Try <ref> tag before the box (OCR/grounding format)
        before = text[:match.start()]
        ref_matches = re.findall(r'<ref>(.+?)</ref>', before)
        if ref_matches:
            label = ref_matches[-1].strip()
        else:
            # Fallback: label = text right after the closing tag (first ~3 words)
            end = match.end()
            context = text[end:end+60].strip().rstrip(',').rstrip('.').strip()
            label = ' '.join(context.split()[:3]) if context else "object"

        results.append((label, coords))
    return results


def classify_query(query: str) -> str:
    """Classify a query into a task type.

    Returns 'detect', 'pose', 'ground', 'count', or 'scene'.
    """
    ql = query.lower().strip()
    pose_keywords = ['standing', 'sitting', 'walking', 'running', 'bending',
                     'raising', 'lying', 'kneeling', 'jumping', 'pose', 'posture']
    count_keywords = ['count', 'how many', 'number of']
    scene_keywords = ['describe', 'scene', 'what', 'weather', 'setting', 'indoor', 'outdoor']
    rel_keywords = ['next to', 'beside', 'behind', 'in front', 'on top',
                    'left of', 'right of', 'holding', 'wearing']

    if any(k in ql for k in count_keywords):
        return 'count'
    first_word = ql.split()[0] if ql.split() else ''
    if first_word in scene_keywords or any(k in ql for k in scene_keywords):
        return 'scene'
    if any(k in ql for k in pose_keywords):
        return 'pose'
    if any(k in ql for k in rel_keywords):
        return 'ground'
    return 'detect'


def load_objects(filepath: str) -> list:
    """Load object class list from a JSON file.
    
    Expected format:
      {"objects": ["person", "monitor", ...], "ocr": ["A", "B", ...]}
    Only the 'objects' list is used for detection. OCR is done via
    grounding query ("read the text on the cube") through the ad-hoc
    /la/grounding_query topic.
    Falls back to a default list if the file cannot be read.
    """
    default = ["person", "monitor", "pc", "robot", "plant", "cube", "cup", "book", "laptop", "cell phone",
               "purse", "backpack", "bottle", "mug", "tv", "ball", "painting", "pen", "keys", "glasses", "hat"]
    p = Path(filepath)
    if not p.exists():
        # Try relative to workspace root
        p = Path(__file__).resolve().parent.parent.parent.parent / filepath
    if not p.exists():
        print(f"[LA Node] Objects file not found: {filepath}, using defaults")
        return default
    try:
        with open(p) as f:
            data = json.load(f)
        objs = data.get("objects", default)
        if not objs:
            return default
        ocr = data.get("ocr", [])
        print(f"[LA Node] Loaded {len(objs)} objects (+ {len(ocr)} OCR chars for grounding queries) from {p}")
        return objs
    except Exception as e:
        print(f"[LA Node] Error loading objects file: {e}, using defaults")
        return default


class LocateAnythingROS2(Node):
    """ROS2 node for LocateAnything visual grounding with TRT acceleration."""

    def __init__(self):
        super().__init__('la_node')

        # Parameters
        self.declare_parameter('objects_file', 'config/la_objects.json')
        self.declare_parameter('interval_frames', 15)
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('max_new_tokens', 128)
        self.declare_parameter('grounding_max_new_tokens', 1024)
        self.declare_parameter('debug', True)

        objects_file = self.get_parameter('objects_file').value
        self.detect_objects = load_objects(objects_file)
        self.interval_frames = self.get_parameter('interval_frames').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.grounding_max_new_tokens = self.get_parameter('grounding_max_new_tokens').value
        self.debug_mode = self.get_parameter('debug').value

        if _CV_BRIDGE_OK:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().warning(
                'cv_bridge unavailable (numpy ABI mismatch); '
                'debug image publishing disabled')
        self.camera_source = "unknown"
        self.frame_count = 0
        self.query_idx = 0
        self.lock = threading.Lock()

        self.depth_image = None
        self.camera_info = None

        # Detection cache: persists between LA inference runs
        self.detections = []
        self.track_history = OrderedDict()
        self._next_track_id = 1
        self.last_inference_time = 0.0
        self.last_query_type = 'detect'
        self.last_query_label = ''

        # LA model (lazy-loaded)
        self.la = None
        self.loading = False

        # Current frame (for async inference)
        self.latest_frame = None
        self.latest_msg_header = None

        # Subscriptions
        self.source_sub = self.create_subscription(
            String, '/camera/source_info', self.source_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.grounding_sub = self.create_subscription(
            String, '/la/grounding_query', self.grounding_callback, 10)
        self._depth_sub = None
        self._camera_info_sub = None

        # Publishers
        self.image_pub = self.create_publisher(Image, '/la/debug_image', 10)
        self.det2d_pub = self.create_publisher(Detection2DArray, '/la/detections_2d', 10)
        self.det3d_pub = self.create_publisher(Detection3DArray, '/la/detections_3d', 10)
        self.point_pub = self.create_publisher(PointStamped, '/la/object_point', 10)
        self.grounding_pub = self.create_publisher(String, '/la/grounding_text', 10)

        # Load model in background
        self._load_model_async()

        self.get_logger().info(
            f'LA Node started. Objects ({len(self.detect_objects)}): {self.detect_objects}, '
            f'interval: {self.interval_frames} frames. '
            f'OCR via grounding query: "read the text on the cube"'
        )

    def _load_model(self):
        """Load the LA model (blocking)."""
        if not _LA_AVAILABLE:
            self.get_logger().error('LocateAnythingEngine not available')
            return
        try:
            self.get_logger().info('Loading LA model (may take 10-20s)...')
            t0 = time.time()
            self.la = LocateAnythingEngine(compile_llm=False)
            # Warmup
            warmup = PILImage.new('RGB', (224, 224), 'gray')
            self.la.ground(warmup, 'find the object', max_new_tokens=8, temperature=0)
            elapsed = time.time() - t0
            self.get_logger().info(f'LA model loaded in {elapsed:.1f}s')
        except Exception as e:
            self.get_logger().error(f'Failed to load LA model: {e}')
            self.la = None

    def _load_model_async(self):
        """Start model loading in a background thread."""
        if self.loading:
            return
        self.loading = True
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()

    def source_callback(self, msg):
        new_source = msg.data
        if new_source == self.camera_source:
            return
        self.camera_source = new_source
        if new_source == "realsense":
            if self._depth_sub is None:
                self._depth_sub = self.create_subscription(
                    Image, "/camera/depth/image_raw", self._depth_callback, 10)
            if self._camera_info_sub is None:
                self._camera_info_sub = self.create_subscription(
                    CameraInfo, "/camera/camera_info", self._camera_info_callback, 10)
        else:
            if self._depth_sub is not None:
                self.destroy_subscription(self._depth_sub)
                self._depth_sub = None
                self.depth_image = None
            if self._camera_info_sub is not None:
                self.destroy_subscription(self._camera_info_sub)
                self._camera_info_sub = None
                self.camera_info = None

    def _depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")

    def _camera_info_callback(self, msg):
        self.camera_info = msg

    def project_3d(self, cx, cy):
        if self.depth_image is None or self.camera_info is None:
            return None
        h, w = self.depth_image.shape[:2]
        px, py = int(cx), int(cy)
        if px < 0 or px >= w or py < 0 or py >= h:
            return None
        depth = self.depth_image[py, px]
        if self.depth_image.dtype == np.uint16:
            depth_m = float(depth) / 1000.0
        else:
            depth_m = float(depth)
        if np.isnan(depth_m) or np.isinf(depth_m) or depth_m <= 0:
            return None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx0 = self.camera_info.k[2]
        cy0 = self.camera_info.k[5]
        X = (cx - cx0) * depth_m / fx
        Y = (cy - cy0) * depth_m / fy
        Z = depth_m
        return (X, Y, Z)

    def grounding_callback(self, msg):
        """Handle ad-hoc visual grounding query via topic."""
        query = msg.data
        if self.la is None or self.latest_frame is None:
            self.get_logger().warning('Model not ready or no frame yet')
            return
        self._run_grounding(query)

    def _run_grounding(self, query):
        """Run LA for a visual grounding query and publish text result."""
        if self.la is None:
            return
        try:
            pil_img = PILImage.fromarray(
                cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB))
            t0 = time.time()
            text = self.la.ground(pil_img, query,
                                  max_new_tokens=self.grounding_max_new_tokens, temperature=0)
            elapsed = (time.time() - t0) * 1000
            dets = parse_detections(text)
            task_type = classify_query(query)

            result = f'[LA Grounding] Query: "{query}"\n{text}\n'
            if dets:
                dets_str = ', '.join(f'{l} {c}' for l, c in dets)
                result += f'Detections: {dets_str}\n'
            result += f'Time: {elapsed:.0f}ms'

            self.get_logger().info(result)
            msg = String()
            msg.data = result
            self.grounding_pub.publish(msg)

            if dets:
                h, w = self.latest_frame.shape[:2]
                with self.lock:
                    self.detections = []
                    for label, b in dets:
                        self.detections.append({
                            'source': 'grounding',
                            'class_name': label,
                            'conf': 1.0,
                            'cx': ((b[0] + b[2]) / 1000.0) * w / 2 if len(b) == 4 else (b[0] / 1000.0) * w,
                            'cy': ((b[1] + b[3]) / 1000.0) * h / 2 if len(b) == 4 else (b[1] / 1000.0) * h,
                            'x1': (b[0] / 1000.0) * w if len(b) == 4 else (b[0] / 1000.0) * w,
                            'y1': (b[1] / 1000.0) * h if len(b) == 4 else (b[1] / 1000.0) * h,
                            'x2': (b[2] / 1000.0) * w if len(b) == 4 else (b[0] / 1000.0) * w,
                            'y2': (b[3] / 1000.0) * h if len(b) == 4 else (b[1] / 1000.0) * h,
                            'w': ((b[2] - b[0]) / 1000.0) * w if len(b) == 4 else 0,
                            'h': ((b[3] - b[1]) / 1000.0) * h if len(b) == 4 else 0,
                            'angle_rad': None,
                            'color': (0, 255, 255),
                        })
                    self.last_query_label = query
                    self.last_query_type = task_type
        except Exception as e:
            self.get_logger().error(f'Grounding failed: {e}')

    def _build_detection_query(self) -> str:
        """Build single comprehensive query with previous-frame context."""
        items = ", ".join(self.detect_objects)
        query = f"Detect all of the following in this scene: {items}. For each one output <box>x1,y1,x2,y2</box> class_name."

        # Prepend previous frame's detections as context (from 2nd query onward)
        with self.lock:
            prev = list(self.detections)
        if not prev:
            return query

        ctx_parts = []
        h, w = 480, 640  # fallback — will be updated if we have a frame
        if self.latest_frame is not None:
            h, w = self.latest_frame.shape[:2]
        for d in prev[:20]:  # limit to 20 objects
            tid = d.get('track_id')
            cn = d['class_name']
            # Convert pixel coords back to LA token format [0,1000] for the prompt
            nx1 = int((d['x1'] / w) * 1000) if w else 0
            ny1 = int((d['y1'] / h) * 1000) if h else 0
            nx2 = int((d['x2'] / w) * 1000) if w else 0
            ny2 = int((d['y2'] / h) * 1000) if h else 0
            box = f"{nx1},{ny1},{nx2},{ny2}"
            ctx_parts.append(f"<box>{box}</box> {cn} ID:{tid}" if tid else f"<box>{box}</box> {cn}")
        ctx = ", ".join(ctx_parts)
        return f"Previously detected: {ctx}. {query}"

    def _run_detection(self, frame, header):
        """Run LA detection — one query lists all objects, replaces previous detections."""
        if self.la is None:
            return

        query = self._build_detection_query()
        self.last_query_label = query[:80]

        try:
            pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.time()
            text = self.la.ground(pil_img, query,
                                  max_new_tokens=self.max_new_tokens, temperature=0)
            elapsed = (time.time() - t0) * 1000
            self.last_inference_time = elapsed
            dets = parse_detections(text)

            new_dets = []
            h, w = frame.shape[:2]
            for label, b in dets:
                if len(b) == 4:
                    x1, y1, x2, y2 = b
                    x1 = (x1 / 1000.0) * w
                    y1 = (y1 / 1000.0) * h
                    x2 = (x2 / 1000.0) * w
                    y2 = (y2 / 1000.0) * h
                elif len(b) == 2:
                    cx = (b[0] / 1000.0) * w
                    cy = (b[1] / 1000.0) * h
                    x1, y1, x2, y2 = cx - 10, cy - 10, cx + 10, cy + 10
                else:
                    continue
                new_dets.append({
                    'source': 'detect',
                    'class_name': label,
                    'conf': 1.0,
                    'cx': (x1 + x2) / 2.0,
                    'cy': (y1 + y2) / 2.0,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'w': x2 - x1,
                    'h': y2 - y1,
                    'angle_rad': None,
                    'color': (0, 255, 0),
                })

            now = time.time()
            for det in new_dets:
                det['track_id'] = self._assign_track_id(det, now)

            with self.lock:
                self.detections = new_dets

            if self.debug_mode:
                self.get_logger().info(
                    f'[LA] detection: {len(new_dets)} boxes in {elapsed:.0f}ms, '
                    f'frame {self.frame_count}')

        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')

    def _assign_track_id(self, det, now):
        """Simple nearest-neighbor track ID assignment with stale cleanup."""
        cx, cy = det['cx'], det['cy']
        best_id = None
        best_dist = 200.0
        stale_cutoff = now - 5.0
        stale_ids = []
        for tid, (tcx, tcy, tt) in list(self.track_history.items()):
            if tt < stale_cutoff:
                stale_ids.append(tid)
                continue
            d = math.hypot(cx - tcx, cy - tcy)
            if d < best_dist:
                best_dist = d
                best_id = tid
        for sid in stale_ids:
            del self.track_history[sid]
        if best_id is not None:
            self.track_history[best_id] = (cx, cy, now)
            return best_id
        else:
            new_id = self._next_track_id
            self._next_track_id += 1
            self.track_history[new_id] = (cx, cy, now)
            return new_id

    def _draw_detections(self, annotated, detections):
        """Draw bounding boxes and YOLO-style labels on the frame."""
        h_img, w_img = annotated.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = map(int, [det['x1'], det['y1'], det['x2'], det['y2']])
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y2 = max(0, min(y2, h_img - 1))
            color = det.get('color', (0, 255, 0))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Center point (LA point-mode style)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

            class_name = det['class_name']
            conf = det.get('conf', 1.0)
            track_id = det.get('track_id')
            w = x2 - x1
            h = y2 - y1

            # Match YOLO label format: class conf [TYPE] ID:id Z:zm
            label = f'{class_name} {conf:.2f} [AABB]'
            if track_id is not None:
                label += f' ID:{track_id}'

            # Add Z distance when depth available
            if self.camera_source == "realsense" and self.depth_image is not None and self.camera_info is not None:
                pt3d = self.project_3d(cx, cy)
                if pt3d:
                    label += f' Z:{pt3d[2]:.2f}m'

            # Top-center label position (matches YOLO)
            lx = int(cx - w / 2)
            ly = max(20, int(cy - h / 2 - 5))
            cv2.putText(annotated, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _publish_detections(self, detections, header):
        """Publish detection results as vision_msgs messages."""
        det2d_msg = Detection2DArray()
        det2d_msg.header = header
        det3d_msg = Detection3DArray()
        det3d_msg.header = header

        for det in detections:
            d2d = Detection2D()
            d2d.bbox.center.position.x = float(det['cx'])
            d2d.bbox.center.position.y = float(det['cy'])
            d2d.bbox.size_x = float(det['w'])
            d2d.bbox.size_y = float(det['h'])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det['class_name']
            hyp.hypothesis.score = det.get('conf', 1.0)
            d2d.results.append(hyp)
            det2d_msg.detections.append(d2d)

            # 3D projection when depth available
            point_3d = None
            if (self.camera_source == "realsense"
                    and self.depth_image is not None
                    and self.camera_info is not None):
                point_3d = self.project_3d(det['cx'], det['cy'])

            if point_3d:
                pt = PointStamped()
                pt.header = header
                pt.point.x, pt.point.y, pt.point.z = point_3d
                self.point_pub.publish(pt)

                d3d = Detection3D()
                d3d.bbox.center.position.x = point_3d[0]
                d3d.bbox.center.position.y = point_3d[1]
                d3d.bbox.center.position.z = point_3d[2]

                d3d_hyp = ObjectHypothesisWithPose()
                d3d_hyp.hypothesis.class_id = det['class_name']
                d3d_hyp.hypothesis.score = det.get('conf', 1.0)
                d3d.results.append(d3d_hyp)
                det3d_msg.detections.append(d3d)

        self.det2d_pub.publish(det2d_msg)
        if len(det3d_msg.detections) > 0:
            self.det3d_pub.publish(det3d_msg)

    def image_callback(self, msg: Image):
        """Main image processing callback."""
        if self.bridge is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_frame = frame.copy()
        self.latest_msg_header = msg.header
        self.frame_count += 1

        # Run detection every N frames
        if (self.frame_count % self.interval_frames == 0
                and self.la is not None):
            self._run_detection(frame.copy(), msg.header)

        # Build annotated frame
        annotated = frame.copy()
        with self.lock:
            active_dets = list(self.detections)
        if active_dets:
            self._draw_detections(annotated, active_dets)
            self._publish_detections(active_dets, msg.header)


        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocateAnythingROS2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
