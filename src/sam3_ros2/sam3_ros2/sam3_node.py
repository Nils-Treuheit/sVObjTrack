"""SAM 3.1 zero-shot segmentation ROS2 node.

Subscribes to /camera/image_raw and text prompts or 2D detection boxes.
Runs SAM 3.1 inference and publishes:
  /sam3/debug_image    — annotated image with mask overlays
  /sam3/masks          — binary mask images

SAM 3.1 lives in a separate uv venv discovered via model_registry.
Checkpoint is downloaded from HuggingFace Hub on first use.
"""
import os
import sys
import time
import threading

from model_registry import sam3_venv, sam3_checkpoint

# SAM3 venv — add site-packages to sys.path before any heavy imports
_sam3_sp = sam3_venv() / 'lib' / 'python3.12' / 'site-packages'
if _sam3_sp.is_dir() and str(_sam3_sp) not in sys.path:
    sys.path.insert(0, str(_sam3_sp))

# SAM3 repo source is inside the venv's parent (editable install)
_sam3_repo = sam3_venv().parent
if _sam3_repo.is_dir() and str(_sam3_repo) not in sys.path:
    sys.path.insert(0, str(_sam3_repo))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np

MASK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


class SAM3Node(Node):
    def __init__(self):
        super().__init__('sam3_node')

        self.declare_parameter('text_prompt', 'object')
        self.declare_parameter('max_fps', 5.0)
        self.declare_parameter('use_boxes', True)
        self.declare_parameter('conf_threshold', 0.3)

        self.text_prompt = self.get_parameter('text_prompt').value
        self.max_fps = self.get_parameter('max_fps').value
        self.use_boxes = self.get_parameter('use_boxes').value
        self.conf_threshold = self.get_parameter('conf_threshold').value

        self.bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._latest_boxes = None
        self._boxes_lock = threading.Lock()
        self._frame_count = 0

        self._load_model()

        self._frame_sub = self.create_subscription(
            Image, '/camera/image_raw', self._frame_cb, 10)
        self._det_sub = self.create_subscription(
            Detection2DArray, '/yolo/detections_2d', self._det_cb, 10)
        self._text_sub = self.create_subscription(
            String, '/sam3/text_prompt', self._text_cb, 10)

        self._debug_pub = self.create_publisher(Image, '/sam3/debug_image', 10)
        self._mask_pub = self.create_publisher(Image, '/sam3/masks', 10)

        self._inference_timer = self.create_timer(1.0 / self.max_fps, self._inference_tick)

        self.get_logger().info(
            f'SAM 3.1 node ready (prompt="{self.text_prompt}", '
            f'max_fps={self.max_fps}, use_boxes={self.use_boxes})')

    def _load_model(self):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        t0 = time.time()
        self.get_logger().info('Loading SAM 3.1 model...')

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        ckpt = str(sam3_checkpoint())
        self.get_logger().info(f'SAM 3.1 checkpoint: {ckpt}')

        self.model = build_sam3_image_model(
            checkpoint_path=ckpt,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(self.model)

        self.get_logger().info(f'SAM 3.1 loaded in {time.time() - t0:.1f}s')

    def _frame_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'Frame conversion failed: {e}')
            return
        with self._frame_lock:
            self._latest_frame = (frame, msg.header)

    def _det_cb(self, msg):
        boxes = []
        for det in msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx, sy = det.bbox.size_x, det.bbox.size_y
            x1 = max(0, cx - sx / 2)
            y1 = max(0, cy - sy / 2)
            x2 = cx + sx / 2
            y2 = cy + sy / 2
            hyp = det.results[0].hypothesis if det.results else None
            cls = hyp.class_id if hyp else 'object'
            score = hyp.score if hyp else 0.0
            if score >= self.conf_threshold:
                boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': cls})
        with self._boxes_lock:
            self._latest_boxes = boxes

    def _text_cb(self, msg):
        new_prompt = msg.data.strip()
        if new_prompt and new_prompt != self.text_prompt:
            self.text_prompt = new_prompt
            self.get_logger().info(f'Text prompt changed: "{self.text_prompt}"')

    def _inference_tick(self):
        import torch

        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame, header = self._latest_frame
            self._latest_frame = None

        with self._boxes_lock:
            boxes = self._latest_boxes

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()

        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.inference_mode():
                state = self.processor.set_image(rgb)

                if self.use_boxes and boxes:
                    for box in boxes:
                        bbox = [box['x1'], box['y1'], box['x2'], box['y2']]
                        state = self.processor.add_geometric_prompt(bbox, True, state)
                    state = self.processor.set_text_prompt(self.text_prompt, state)
                else:
                    state = self.processor.set_text_prompt(self.text_prompt, state)

        dt_ms = (time.time() - t0) * 1000

        masks = state.get('masks', [])
        scores = state.get('scores', [])
        pred_boxes = state.get('boxes', [])

        annotated = frame.copy()
        all_masks = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            color = MASK_COLORS[i % len(MASK_COLORS)]

            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            if mask_np.dtype != bool:
                mask_np = mask_np > 0.5

            overlay = annotated.copy()
            overlay[mask_np] = color
            annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

            contours, _ = cv2.findContours(
                mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

            if i < len(pred_boxes):
                bx = pred_boxes[i]
                if hasattr(bx, 'tolist'):
                    bx = bx.tolist()
                x1, y1, x2, y2 = map(int, bx[:4])
                label = f'{self.text_prompt} {score:.2f}'
                cv2.putText(annotated, label, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            all_masks.append(mask_np)

        if all_masks:
            combined_mask = np.zeros_like(all_masks[0], dtype=np.uint8)
            for i, m in enumerate(all_masks):
                combined_mask[m] = i + 1
            mask_msg = self.bridge.cv2_to_imgmsg(
                combined_mask.astype(np.uint8), encoding='mono8')
            mask_msg.header = header
            self._mask_pub.publish(mask_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'SAM3: {len(masks)} masks, {dt_ms:.1f}ms',
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = SAM3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
