import sys
print(f"CURRENT PYTHON INTERPRETER: {sys.executable}")

import rclpy
from rclpy.node import Node

import time
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


class YOLO11Node(Node):
    def __init__(self):
        super().__init__('yolo11_node')

        # Parameters
        self.declare_parameter("model_path", "yolo11n.pt")
        self.declare_parameter("conf_threshold", 0.4)

        model_path = self.get_parameter("model_path").value
        self.conf_threshold = self.get_parameter("conf_threshold").value

        # Load YOLO11 model
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded YOLO model: {model_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.get_logger().info(f"YOLO model moved to {device}")

        self.bridge = CvBridge()

        # Camera Source State
        self.camera_source = "unknown"

        self.depth_image = None
        self.camera_info = None

        # Tracking State
        self.track_history = {}

        # --------------------------
        # Subscribers and Publishers
        # --------------------------

        # Source Subscriber
        self.source_sub = self.create_subscription(
            String,
            "/camera/source_info",
            self.source_callback,
            10
        )

        # Depth Subscriber
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            10
        )
        
        # Camera Info Subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            10
        )

        # Image Subscriber
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        # 2D Detection Publisher
        self.det2d_pub = self.create_publisher(
            Detection2DArray,
            "/yolo/detections_2d",
            10
        )

        # 3D Detection Publisher
        self.det3d_pub = self.create_publisher(
            Detection3DArray,
            "/yolo/detections_3d",
            10
        )

        # Point Stamped Publisher
        self.point_pub = self.create_publisher(
            PointStamped,
            "/yolo/object_point",
            10
        )

        # Debug Image Publisher
        self.image_pub = self.create_publisher(
            Image,
            "/yolo/debug_image",
            10
        )
    
    def source_callback(self, msg):
        self.camera_source = msg.data


    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough"
        )

    def camera_info_callback(self, msg):
        self.camera_info = msg


    # =========================================================
    # DEPTH -> 3D PROJECTION
    # =========================================================
    def project_3d(self, cx, cy):
        if self.depth_image is None or self.camera_info is None: return None
    
        h, w = self.depth_image.shape[:2]
    
        px = int(cx)
        py = int(cy)
    
        if px < 0 or px >= w: return None
        if py < 0 or py >= h: return None

        depth = self.depth_image[py, px]
        
        # depth image typically arrives in mm otherwise in meters
        if self.depth_image.dtype == np.uint16:
            # Intel RealSense driver [encoding = 16UC1]
            # units = millimeters
            depth_m = float(depth) / 1000.0
        else:
            # Intel RealSense driver [encoding = 32FC1]
            # units = meters
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

    # =========================================================
    # MAIN CALLBACK
    # =========================================================
    def image_callback(self, msg: Image):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        now = time.time()

        # -------------------------
        # YOLO TRACKING
        # -------------------------
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

        if results.boxes is not None:

            for box in results.boxes:

                conf = float(box.conf[0])
                if conf < self.conf_threshold: continue

                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                track_id = int(box.id[0]) if box.id is not None else None

                # -------------------------
                # VELOCITY (2D fallback)
                # -------------------------
                vx, vy = 0.0, 0.0
                if track_id is not None and track_id in self.track_history:
                    px, py, pt = self.track_history[track_id]
                    dt = now - pt
                    if dt > 0:
                        vx = (cx - px) / dt
                        vy = (cy - py) / dt

                self.track_history[track_id] = (cx, cy, now)

                # -------------------------
                # 3D PROJECTION (IF REALSENSE)
                # -------------------------
                point_3d = None

                if (
                    self.camera_source == "realsense"
                    and self.depth_image is not None
                    and self.camera_info is not None
                ): point_3d = self.project_3d(cx, cy)

                # -------------------------
                # DRAW BOX
                # -------------------------
                label = f"{class_name} {conf:.2f}"
                if track_id is not None: label += f" ID:{track_id}"
                if point_3d: label += f" Z:{point_3d[2]:.2f}m"

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # -------------------------
                # 2D DETECTION MSG
                # -------------------------
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

                # -------------------------
                # 3D OUTPUT (IF AVAILABLE)
                # -------------------------
                if point_3d:
                    pt = PointStamped()
                    pt.header = msg.header
                    pt.point.x, pt.point.y, pt.point.z = point_3d
                    self.point_pub.publish(pt)

                    det3d = Detection3D()

                    det3d.bbox.center.position.x = point_3d[0]
                    det3d.bbox.center.position.y = point_3d[1]
                    det3d.bbox.center.position.z = point_3d[2]

                    det3d.results.append(hyp)

                    det3d_msg.detections.append(det3d)

        # -------------------------
        # PUBLISH
        # -------------------------
        self.det2d_pub.publish(det2d_msg)
        if len(det3d_msg.detections) > 0: self.det3d_pub.publish(det3d_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLO11Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# ros2 run yolo11_ros2 yolo_node