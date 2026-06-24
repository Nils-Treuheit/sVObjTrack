import sys
print(f"CURRENT PYTHON INTERPRETER: {sys.executable}")

import rclpy
from rclpy.node import Node

import time
import math
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo

from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    Detection3DArray,
    Detection3D
)

from cv_bridge import CvBridge
import cv2

import torch
from ultralytics import YOLO
from geometry_msgs.msg import PointStamped


class YOLO26Node(Node):
    def __init__(self):
        super().__init__('yolo26_node')

        self.declare_parameter("model_type", "cubified")
        self.declare_parameter("conf_threshold", 0.4)

        model_type = self.get_parameter("model_type").value
        self.conf_threshold = self.get_parameter("conf_threshold").value

        if model_type == "yolo11":
            model_path = "yolo11n.pt"
        elif model_type == "yolo26":
            model_path = "yolo11n.pt"
        elif model_type == "cubified":
            model_path = "yolo.pt"
        else:
            self.get_logger().error(f"Unknown model_type: {model_type}, falling back to yolo11")
            model_path = "yolo11n.pt"
            model_type = "yolo11"

        self.model_type = model_type
        self.is_obb = (model_type == "cubified")

        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded model_type={model_type} path={model_path} obb={self.is_obb}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.get_logger().info(f"YOLO model moved to {device}")

        self.bridge = CvBridge()

        self.camera_source = "unknown"
        self.depth_image = None
        self.camera_info = None
        self.track_history = {}

        self.source_sub = self.create_subscription(
            String, "/camera/source_info", self.source_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback, 10)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self.camera_info_callback, 10)

        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10)

        self.det2d_pub = self.create_publisher(
            Detection2DArray, "/yolo/detections_2d", 10)

        self.det3d_pub = self.create_publisher(
            Detection3DArray, "/yolo/detections_3d", 10)

        self.point_pub = self.create_publisher(
            PointStamped, "/yolo/object_point", 10)

        self.image_pub = self.create_publisher(
            Image, "/yolo/debug_image", 10)

    def source_callback(self, msg):
        self.camera_source = msg.data

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")

    def camera_info_callback(self, msg):
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

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        now = time.time()

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            verbose=False
        )[0]

        det2d_msg = Detection2DArray()
        det2d_msg.header = msg.header

        det3d_msg = Detection3DArray()
        det3d_msg.header = msg.header

        annotated = frame.copy()

        if self.is_obb and hasattr(results, 'obb') and results.obb is not None:
            self._process_obb(results.obb, annotated, det2d_msg, det3d_msg, msg, now)
        elif results.boxes is not None:
            self._process_boxes(results.boxes, annotated, det2d_msg, det3d_msg, msg, now)

        self.det2d_pub.publish(det2d_msg)
        if len(det3d_msg.detections) > 0:
            self.det3d_pub.publish(det3d_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)

    def _process_boxes(self, boxes, annotated, det2d_msg, det3d_msg, img_msg, now):
        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_id = int(box.id[0]) if box.id is not None else None

            vx, vy = 0.0, 0.0
            if track_id is not None and track_id in self.track_history:
                px, py, pt = self.track_history[track_id]
                dt = now - pt
                if dt > 0:
                    vx = (cx - px) / dt
                    vy = (cy - py) / dt

            self.track_history[track_id] = (cx, cy, now)

            point_3d = None
            if (self.camera_source == "realsense"
                    and self.depth_image is not None
                    and self.camera_info is not None):
                point_3d = self.project_3d(cx, cy)

            label = f"{class_name} {conf:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            if point_3d:
                label += f" Z:{point_3d[2]:.2f}m"

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            det2d = Detection2D()
            det2d.bbox.center.position.x = float(cx)
            det2d.bbox.center.position.y = float(cy)
            det2d.bbox.size_x = float(x2 - x1)
            det2d.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = conf

            det2d.results.append(hyp)
            det2d_msg.detections.append(det2d)

            if point_3d:
                pt = PointStamped()
                pt.header = img_msg.header
                pt.point.x, pt.point.y, pt.point.z = point_3d
                self.point_pub.publish(pt)

                det3d = Detection3D()
                det3d.bbox.center.position.x = point_3d[0]
                det3d.bbox.center.position.y = point_3d[1]
                det3d.bbox.center.position.z = point_3d[2]
                det3d.results.append(hyp)
                det3d_msg.detections.append(det3d)

    def _process_obb(self, obb_data, annotated, det2d_msg, det3d_msg, img_msg, now):
        for obb in obb_data:
            conf = float(obb.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(obb.cls[0])
            class_name = self.model.names[cls_id]

            xywhr = obb.xywhr[0].cpu().numpy()
            cx, cy, w, h, angle_rad = xywhr

            track_id = int(obb.id[0]) if obb.id is not None else None

            vx, vy = 0.0, 0.0
            if track_id is not None and track_id in self.track_history:
                px, py, pt = self.track_history[track_id]
                dt = now - pt
                if dt > 0:
                    vx = (cx - px) / dt
                    vy = (cy - py) / dt

            self.track_history[track_id] = (cx, cy, now)

            point_3d = None
            if (self.camera_source == "realsense"
                    and self.depth_image is not None
                    and self.camera_info is not None):
                point_3d = self.project_3d(cx, cy)

            angle_deg = math.degrees(angle_rad)
            rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int32(box_pts)
            cv2.polylines(annotated, [box_pts], True, (255, 0, 0), 2)

            label = f"{class_name} {conf:.2f}"
            if track_id is not None:
                label += f" ID:{track_id}"
            if point_3d:
                label += f" Z:{point_3d[2]:.2f}m"

            label_x = int(cx - w / 2)
            label_y = max(20, int(cy - h / 2 - 5))
            cv2.putText(
                annotated, label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            det2d = Detection2D()
            det2d.bbox.center.position.x = float(cx)
            det2d.bbox.center.position.y = float(cy)
            det2d.bbox.size_x = float(w)
            det2d.bbox.size_y = float(h)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = conf
            hyp.pose.pose.orientation.z = math.sin(angle_rad / 2.0)
            hyp.pose.pose.orientation.w = math.cos(angle_rad / 2.0)

            det2d.results.append(hyp)
            det2d_msg.detections.append(det2d)

            if point_3d:
                pt = PointStamped()
                pt.header = img_msg.header
                pt.point.x, pt.point.y, pt.point.z = point_3d
                self.point_pub.publish(pt)

                det3d = Detection3D()
                det3d.bbox.center.position.x = point_3d[0]
                det3d.bbox.center.position.y = point_3d[1]
                det3d.bbox.center.position.z = point_3d[2]
                det3d.results.append(hyp)
                det3d_msg.detections.append(det3d)


def main(args=None):
    rclpy.init(args=args)
    node = YOLO26Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# ros2 run yolo26_ros2 yolo_node --ros-args -p model_type:=cubified
# ros2 run yolo26_ros2 yolo_node --ros-args -p model_type:=yolo11
