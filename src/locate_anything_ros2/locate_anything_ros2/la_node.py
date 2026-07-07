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
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
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


def parse_boxes(text: str) -> list:
    """Extract bounding boxes from LA <box> tag output.

    Returns list of [x1, y1, x2, y2] or [x, y] with normalized coords (0-1).
    """
    boxes = []
    for match in re.finditer(r'<box>(.+?)</box>', text):
        content = match.group(1).strip()
        coords = [float(p) for p in re.findall(r'[\d.]+', content)]
        if len(coords) in (2, 4):
            boxes.append(coords)
    return boxes


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


class LocateAnythingROS2(Node):
    """ROS2 node for LocateAnything visual grounding with TRT acceleration."""

    def __init__(self):
        super().__init__('la_node')

        # Parameters
        self.declare_parameter('detect_queries', '["person", "car", "dog", "chair"]')
        self.declare_parameter('interval_frames', 30)
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('max_new_tokens', 32)
        self.declare_parameter('debug', True)

        detect_queries_str = self.get_parameter('detect_queries').value
        self.detect_queries = eval(detect_queries_str)
        self.interval_frames = self.get_parameter('interval_frames').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.debug_mode = self.get_parameter('debug').value

        if not isinstance(self.detect_queries, list):
            self.detect_queries = [self.detect_queries]

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

        # Detection cache: persists between LA inference runs
        self.detections = []
        self.track_history = OrderedDict()
        self._next_track_id = 1
        self.last_inference_time = 0.0

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

        # Publishers
        self.image_pub = self.create_publisher(Image, '/la/debug_image', 10)
        self.det2d_pub = self.create_publisher(Detection2DArray, '/la/detections_2d', 10)
        self.grounding_pub = self.create_publisher(String, '/la/grounding_text', 10)

        # Load model in background
        self._load_model_async()

        self.get_logger().info(
            f'LA Node started. Queries: {self.detect_queries}, '
            f'interval: {self.interval_frames} frames, '
            f'TRT: {_TRT_AVAILABLE}'
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
        self.camera_source = msg.data

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
                                  max_new_tokens=self.max_new_tokens, temperature=0)
            elapsed = (time.time() - t0) * 1000
            boxes = parse_boxes(text)

            result = f'[LA Grounding] Query: "{query}"\n{text}\n'
            if boxes:
                result += f'Boxes: {boxes}\n'
            result += f'Time: {elapsed:.0f}ms'

            self.get_logger().info(result)
            msg = String()
            msg.data = result
            self.grounding_pub.publish(msg)

            # Also draw on debug frame if available
            if boxes:
                h, w = self.latest_frame.shape[:2]
                with self.lock:
                    self.detections = [{
                        'source': 'grounding',
                        'class_name': query,
                        'conf': 1.0,
                        'cx': (b[0] + b[2]) * w / 2 if len(b) == 4 else b[0] * w,
                        'cy': (b[1] + b[3]) * h / 2 if len(b) == 4 else b[1] * h,
                        'x1': b[0] * w if len(b) == 4 else b[0] * w,
                        'y1': b[1] * h if len(b) == 4 else b[1] * h,
                        'x2': b[2] * w if len(b) == 4 else b[0] * w,
                        'y2': b[3] * h if len(b) == 4 else b[1] * h,
                        'w': (b[2] - b[0]) * w if len(b) == 4 else 0,
                        'h': (b[3] - b[1]) * h if len(b) == 4 else 0,
                        'angle_rad': None,
                        'color': (0, 255, 255),
                    } for b in boxes]
        except Exception as e:
            self.get_logger().error(f'Grounding failed: {e}')

    def _run_detection(self, frame, header):
        """Run LA detection query for current frame."""
        if self.la is None:
            return

        # Pick the next query in round-robin
        query = self.detect_queries[self.query_idx % len(self.detect_queries)]
        self.query_idx += 1

        try:
            pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.time()
            text = self.la.ground(pil_img, query,
                                  max_new_tokens=self.max_new_tokens, temperature=0)
            elapsed = (time.time() - t0) * 1000
            self.last_inference_time = elapsed
            boxes = parse_boxes(text)

            detections = []
            h, w = frame.shape[:2]
            for b in boxes:
                if len(b) == 4:
                    x1, y1, x2, y2 = b
                    # Normalized coords -> pixel
                    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                elif len(b) == 2:
                    cx, cy = b[0] * w, b[1] * h
                    x1, y1, x2, y2 = cx - 10, cy - 10, cx + 10, cy + 10
                else:
                    continue
                detections.append({
                    'source': query,
                    'class_name': query,
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

            # Assign track IDs
            now = time.time()
            for det in detections:
                det['track_id'] = self._assign_track_id(det, now)

            with self.lock:
                self.detections = detections

            if self.debug_mode:
                self.get_logger().info(
                    f'[LA] "{query}": {len(boxes)} boxes in {elapsed:.0f}ms, '
                    f'frame {self.frame_count}')

        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')

    def _assign_track_id(self, det, now):
        """Simple nearest-neighbor track ID assignment."""
        cx, cy = det['cx'], det['cy']
        best_id = None
        best_dist = 100.0
        for tid, (tcx, tcy, tt) in list(self.track_history.items()):
            d = math.hypot(cx - tcx, cy - tcy)
            if d < best_dist:
                best_dist = d
                best_id = tid
        if best_id is not None:
            self.track_history[best_id] = (cx, cy, now)
            return best_id
        else:
            new_id = self._next_track_id
            self._next_track_id += 1
            self.track_history[new_id] = (cx, cy, now)
            return new_id

    def _draw_detections(self, annotated, detections):
        """Draw bounding boxes and labels on the frame."""
        for det in detections:
            x1, y1, x2, y2 = map(int, [det['x1'], det['y1'], det['x2'], det['y2']])
            color = det.get('color', (0, 255, 0))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = det['class_name']
            track_id = det.get('track_id')
            if track_id is not None:
                label += f' ID:{track_id}'

            lx, ly = x1, max(20, y1 - 5)
            cv2.putText(annotated, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _publish_detections(self, detections, header):
        """Publish detection results as vision_msgs/Detection2DArray."""
        det2d_msg = Detection2DArray()
        det2d_msg.header = header

        for det in detections:
            d2d = Detection2D()
            d2d.bbox.center.position.x = float(det['cx'])
            d2d.bbox.center.position.y = float(det['cy'])
            d2d.bbox.size_x = float(det['w'])
            d2d.bbox.size_y = float(det['h'])
            det2d_msg.detections.append(d2d)

        self.det2d_pub.publish(det2d_msg)

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

        # Publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)

        # Show inference overlay
        if self.last_inference_time > 0:
            status = f'LA: {self.last_inference_time:.0f}ms | {len(active_dets)} tracked'
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
