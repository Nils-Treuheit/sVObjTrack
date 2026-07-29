"""RF-DETR object detection ROS2 node.

Runs RF-DETR (Nano/Small/Medium/Large) with optimize_for_inference()
for maximum FPS.  Publishes Detection2DArray and annotated debug image.

The RF-DETR venv is discovered via model_registry.
"""
import os
import sys
import time

from model_registry import rfdetr_venv

# RF-DETR venv site-packages
_rf_sp = rfdetr_venv() / 'lib' / 'python3.10' / 'site-packages'
if _rf_sp.is_dir() and str(_rf_sp) not in sys.path:
    sys.path.insert(0, str(_rf_sp))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

import torch

RFDETR_MODELS = {
    'nano':   ('RFDETRNano',   384),
    'small':  ('RFDETRSmall',  512),
    'medium': ('RFDETRMedium', 576),
    'large':  ('RFDETRLarge',  704),
}

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


class RFDETRNode(Node):
    def __init__(self):
        super().__init__('rfdetr_node')

        self.declare_parameter('model_size', 'nano')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('device', '')

        model_size = self.get_parameter('model_size').value.lower()
        self.conf_threshold = self.get_parameter('conf_threshold').value
        device_param = self.get_parameter('device').value

        if model_size not in RFDETR_MODELS:
            self.get_logger().warning(f'Unknown model_size "{model_size}", falling back to nano')
            model_size = 'nano'

        class_name, resolution = RFDETR_MODELS[model_size]
        self.get_logger().info(f'Loading RF-DETR {model_size} ({class_name}) at {resolution}x{resolution}')

        # Lazy import so module-level import stays fast
        rfdetr_mod = __import__('rfdetr')
        model_cls = getattr(rfdetr_mod, class_name)

        t0 = time.time()
        self.model = model_cls(resolution=resolution)
        self.get_logger().info(f'RF-DETR loaded in {time.time()-t0:.1f}s')
        self.model.optimize_for_inference()
        self.get_logger().info(f'RF-DETR optimized in {time.time()-t0:.1f}s')

        self.bridge = CvBridge()
        self.source = 'unknown'
        self.frame_w, self.frame_h = 640, 480

        self.create_subscription(String, '/camera/source_info', self._src_cb, 10)
        self.create_subscription(Image, '/camera/image_raw', self._img_cb, 10)

        self.det_pub = self.create_publisher(Detection2DArray, '/rfdetr/detections_2d', 10)
        self.img_pub = self.create_publisher(Image, '/rfdetr/debug_image', 10)

        self.get_logger().info(f'RF-DETR node ready (model={model_size})')

    def _src_cb(self, msg):
        self.source = msg.data

    def _img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        self.frame_w, self.frame_h = w, h
        now = self.get_clock().now().to_msg()

        # RF-DETR expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        dets = self.model.predict(rgb, threshold=self.conf_threshold)
        dt_ms = (time.time() - t0) * 1000

        det2d_msg = Detection2DArray()
        det2d_msg.header = msg.header
        annotated = frame.copy()

        n = len(dets.xyxy)
        for i in range(n):
            x1, y1, x2, y2 = dets.xyxy[i].tolist()
            conf = float(dets.confidence[i])
            cls_id = int(dets.class_id[i])
            cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)

            det = Detection2D()
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            det2d_msg.detections.append(det)

            color = (0, 255, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'{cls_name} {conf:.2f}'
            cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.det_pub.publish(det2d_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self.img_pub.publish(debug_msg)

        if n > 0:
            self.get_logger().debug(f'RF-DETR: {n} detections, {dt_ms:.1f}ms', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = RFDETRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
