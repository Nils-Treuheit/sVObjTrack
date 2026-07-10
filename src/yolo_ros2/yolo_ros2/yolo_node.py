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

from geometry_msgs.msg import PointStamped
from typing import Tuple,List,Union
from os.path import basename
from random import randint


# Default Fallback Models
MODEL_YOLO11 = "models/yolo11m.pt"
MODEL_YOLO26 = "models/yolo26m.pt"
MODEL_CUBIFIED = "models/yolo_cubified.pt"
MODEL_YOLO11_OBB = "models/yolo11s-obb.pt"
MODEL_YOLO26_OBB = "models/yolo26s-obb.pt"
MODEL_YOLO11_POSE = "models/yolo11s-pose.pt"
MODEL_YOLO26_POSE = "models/yolo26s-pose.pt"
MODEL_UNIFIED = "models/yolo26-obb_cubified_v2.pt"
NC_CUBE = 66
NC_NORMAL = 95

# COCO keypoint skeleton (17 keypoints)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
KEYPOINT_COLORS = [
    (255, 0, 0), (255, 0, 255), (0, 0, 255), (255, 0, 255), (0, 0, 255),
    (255, 165, 0), (0, 255, 0), (255, 165, 0), (0, 255, 0),
    (255, 165, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 255)
]
SKELETON_COLORS = [
    (255, 0, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255),
    (255, 165, 0), (255, 165, 0), (255, 165, 0), (0, 255, 0), (0, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 0, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255)
]


class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        def translate_model_id(model_id, model_t):
            if model_id == "yolo26":
                path = MODEL_YOLO26
                model_type = "AABB"
            elif model_id == "yolo11":
                path = MODEL_YOLO11
                model_type = "AABB"
            elif model_id == "cubified":
                path = MODEL_CUBIFIED
                model_type = "OBB"
            elif model_id == "yolo26-obb":
                path = MODEL_YOLO26_OBB
                model_type = "OBB"
            elif model_id == "yolo11-obb":
                path = MODEL_YOLO11_OBB
                model_type = "OBB"
            elif model_id == "yolo11-pose":
                path = MODEL_YOLO11_POSE
                model_type = "pose"
            elif model_id == "yolo26-pose":
                path = MODEL_YOLO26_POSE
                model_type = "pose"
            elif model_id in ("unified", "yolo26-cubified-v2"):
                path = MODEL_UNIFIED
                model_type = "OBB"
            elif model_id=="":
                self.get_logger().warning(f"No model provided using YOLO26!")
                path = MODEL_YOLO26
                model_type = "AABB"
            else:
                self.get_logger().warning(f"Assuming model to be YOLO26-like, if no type provided and type not in model name assume AABB!")
                path = model_id
                mtype = basename(model_id).strip().split(".")[0].split('-')[-1]
                model_type = model_t if model_t else (mtype if mtype else "AABB")
            return path, model_type

        self.declare_parameter("model_id", "")
        self.declare_parameter("model_type", "")
        self.declare_parameter("bb_tracker", "")
        self.declare_parameter("conf_threshold", 0.4)

        model_id:str = self.get_parameter("model_id").value
        model_type:str = self.get_parameter("model_type").value
        bb_tracker:str = self.get_parameter("bb_tracker").value
        self.conf_threshold = self.get_parameter("conf_threshold").value

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models:Union[List,object]

        if model_id.startswith("["):
            self.model_ids = [m_id.strip() for m_id in model_id.strip('[]').split(',')]

            raw_types = [t.strip() for t in model_type.strip('[]').split(',') if t.strip()]
            model_types = raw_types if raw_types else []
            if not model_types:
                model_types = ['AABB']
            model_types.extend([model_types[-1] for _ in range(len(self.model_ids) - len(model_types))])

            raw_trackers = [t.strip() for t in bb_tracker.strip('[]').split(',') if t.strip()]
            self.bb_trackers = raw_trackers if raw_trackers else ['bytetrack.yaml']
            self.bb_trackers.extend(['bytetrack.yaml' for _ in range(len(self.model_ids) - len(self.bb_trackers))])

            def _build_model(path, m_id):
                if "MoE" in m_id or m_id == "cubified":
                    from ultralytics import YOLO
                    from moe_yolo.moe_model import MoEYOLO
                    return MoEYOLO(YOLO(path), NC_NORMAL, NC_CUBE)
                if path == MODEL_UNIFIED or m_id in ("unified", "yolo26-cubified-v2") or basename(path).startswith("yolo26-obb_cubified_v2"):
                    from unified_yolo.unified_model import UnifiedYOLO
                    return UnifiedYOLO(path, device)
                from ultralytics import YOLO
                return YOLO(path)

            self.models = list()
            self.model_types = list() 
            for m_type, m_id in zip(model_types, self.model_ids):
                path, m_t = translate_model_id(m_id, m_type) 
                self.model_types.append(m_t) 
                self.models.append(_build_model(path, m_id))

            self.get_logger().info(f"Fusion mode: combine YOLOs on {device}\n\t> (Models,Trackers):{list(zip(self.model_ids,self.bb_trackers))}")
        else:
            self.model_ids = model_id
            path, m_type = translate_model_id(model_id, model_type)
            self.model_types = m_type
            self.bb_trackers = bb_tracker if bb_tracker else 'bytetrack.yaml'
            if "MoE" in model_id or model_id == "cubified":
                from ultralytics import YOLO
                from moe_yolo.moe_model import MoEYOLO
                self.models = MoEYOLO(YOLO(path), NC_NORMAL, NC_CUBE)
            elif path == MODEL_UNIFIED or model_id in ("unified", "yolo26-cubified-v2") or basename(path).startswith("yolo26-obb_cubified_v2"):
                from unified_yolo.unified_model import UnifiedYOLO
                self.models = UnifiedYOLO(path, device)
            else:
                from ultralytics import YOLO
                self.models = YOLO(path)
                self.models.to(device)
            self.get_logger().info(f"Single model: {basename(model_id).split('.')[0]} on {device}")
        
        self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)] 
        if not isinstance(self.models, list): self.colors = self.colors[0]
        elif len(self.colors)>len(self.models): self.colors = self.colors[:len(self.models)]
        else: 
            self.colors = set(self.colors) 
            while len(self.colors)<len(self.models): self.colors.add((randint(0,255),randint(0,255),randint(0,255)))
            self.colors = list(self.colors)

        if isinstance(self.model_types, list):
            self.is_obb = any(t == "OBB" for t in self.model_types)
            self.is_pose = any(t == "pose" for t in self.model_types)
        else:
            self.is_obb = (self.model_types == "OBB")
            self.is_pose = (self.model_types == "pose")

        self.bridge = CvBridge()

        self.camera_source = "unknown"
        self.depth_image = None
        self.camera_info = None
        self.track_history = {}
        self._next_track_id = 1

        self.source_sub = self.create_subscription(String, "/camera/source_info", self.source_callback, 10)
        self._depth_sub = None
        self._camera_info_sub = None
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        self.det2d_pub = self.create_publisher(Detection2DArray, "/yolo/detections_2d", 10)
        self.det3d_pub = self.create_publisher(Detection3DArray, "/yolo/detections_3d", 10)
        self.point_pub = self.create_publisher(PointStamped, "/yolo/object_point", 10)
        self.image_pub = self.create_publisher(Image, "/yolo/debug_image", 10)

    def source_callback(self, msg):
        new_source = msg.data
        if new_source == self.camera_source:
            return
        self.camera_source = new_source
        # Enable depth/CameraInfo only for realsense (has depth sensor)
        if new_source == "realsense":
            if self._depth_sub is None:
                self._depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self._depth_callback, 10)
            if self._camera_info_sub is None:
                self._camera_info_sub = self.create_subscription(CameraInfo, "/camera/camera_info", self._camera_info_callback, 10)
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
    
    # --- draw keypoints on debug frame ---
    def _draw_keypoints(self, annotated, keypoints, kpt_conf=None):
        h, w = annotated.shape[:2]
        vis_kpts = []
        for i, (x, y) in enumerate(keypoints):
            conf = kpt_conf[i] if kpt_conf is not None else 1.0
            if conf < self.conf_threshold:
                vis_kpts.append((None, None))
                continue
            px, py = int(x), int(y)
            if px < 0 or px >= w or py < 0 or py >= h:
                vis_kpts.append((None, None))
                continue
            color = KEYPOINT_COLORS[i] if i < len(KEYPOINT_COLORS) else (0, 255, 0)
            cv2.circle(annotated, (px, py), 4, color, -1)
            vis_kpts.append((px, py))

        for si, (i, j) in enumerate(SKELETON):
            if i >= len(vis_kpts) or j >= len(vis_kpts):
                continue
            p1 = vis_kpts[i]
            p2 = vis_kpts[j]
            if p1[0] is None or p2[0] is None:
                continue
            color = SKELETON_COLORS[si] if si < len(SKELETON_COLORS) else (0, 255, 0)
            cv2.line(annotated, p1, p2, color, 2)

    # --- retrieve box preds ---
    def _process_boxes(self, preds, m_id:str, model, color:Tuple[int,int,int]) -> list:
        detections = []
        has_pose = hasattr(preds, 'keypoints') and preds.keypoints is not None

        kpt_data = None
        if has_pose:
            kpt_xy = preds.keypoints.xy.cpu().numpy()  # (N, 17, 2)
            kpt_c = preds.keypoints.conf.cpu().numpy() if preds.keypoints.conf is not None else None
            kpt_data = (kpt_xy, kpt_c)

        if hasattr(preds, 'obb') and preds.obb is not None:
            for i, obb in enumerate(preds.obb):
                conf = float(obb.conf[0])
                if conf < self.conf_threshold: continue
                xywhr = obb.xywhr[0].cpu().numpy()
                cx, cy, w, h, angle_rad = xywhr
                kpts = kpt_data[0][i] if kpt_data is not None else None
                kpt_conf_i = kpt_data[1][i] if kpt_data is not None and kpt_data[1] is not None else None
                detections.append({
                    'source': m_id,
                    'color': color,
                    'conf': conf,
                    'cls_id': int(obb.cls[0]),
                    'class_name': model.names[int(obb.cls[0])],
                    'cx': cx, 'cy': cy,
                    'x1': cx - w / 2, 'y1': cy - h / 2,
                    'x2': cx + w / 2, 'y2': cy + h / 2,
                    'w': w, 'h': h,
                    'angle_rad': angle_rad,
                    'track_id': int(obb.id[0]) if obb.id is not None else None,
                    'keypoints': kpts,
                    'kpt_conf': kpt_conf_i,
                })
        elif preds.boxes is not None:
            for i, box in enumerate(preds.boxes):
                conf = float(box.conf[0])
                if conf < self.conf_threshold: continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                kpts = kpt_data[0][i] if kpt_data is not None else None
                kpt_conf_i = kpt_data[1][i] if kpt_data is not None and kpt_data[1] is not None else None
                detections.append({
                    'source': m_id,
                    'color': color,
                    'conf': conf,
                    'cls_id': int(box.cls[0]),
                    'class_name': model.names[int(box.cls[0])],
                    'cx': (x1 + x2) / 2.0,
                    'cy': (y1 + y2) / 2.0,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'w': x2 - x1, 'h': y2 - y1,
                    'angle_rad': None,
                    'track_id': int(box.id[0]) if box.id is not None else None,
                    'keypoints': kpts,
                    'kpt_conf': kpt_conf_i,
                })
        return detections
    
    # ----- fuse detections (multi-model NMS) -----
    @staticmethod
    def _overlap(a, b, iou_thresh=0.5):
        inter_x1 = max(a['x1'], b['x1'])
        inter_y1 = max(a['y1'], b['y1'])
        inter_x2 = min(a['x2'], b['x2'])
        inter_y2 = min(a['y2'], b['y2'])
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return False
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        a_area = a['w'] * a['h']
        b_area = b['w'] * b['h']
        union = a_area + b_area - inter_area
        if union <= 0:
            return False
        return (inter_area / union) >= iou_thresh

    def _fuse_candidates(self, candidates):
        kept = []
        assigned = [False] * len(candidates)

        for i, det in enumerate(candidates):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, len(candidates)):
                if assigned[j]:
                    continue
                if det['cls_id'] != candidates[j]['cls_id']:
                    continue
                if self._overlap(det, candidates[j], iou_thresh=0.5):
                    group.append(j)
                    assigned[j] = True

            group_dets = [candidates[idx] for idx in group]

            obb_dets = [d for d in group_dets if d['angle_rad'] is not None]
            aabb_dets = [d for d in group_dets if d['angle_rad'] is None and d.get('keypoints') is None]
            pose_dets = [d for d in group_dets if d.get('keypoints') is not None]

            # Box priority: OBB > AABB > pose-provided box
            if obb_dets:
                best = max(obb_dets, key=lambda d: d['conf'])
            elif aabb_dets:
                best = max(aabb_dets, key=lambda d: d['conf'])
            elif pose_dets:
                best = max(pose_dets, key=lambda d: d['conf'])
            else:
                continue

            # Merge keypoints from the best pose detection (if any) into the kept box
            if pose_dets:
                best_pose = max(pose_dets, key=lambda d: d['conf'])
                if best.get('keypoints') is None:
                    best['keypoints'] = best_pose['keypoints']
                    best['kpt_conf'] = best_pose.get('kpt_conf')

            kept.append(best)

        return kept
    
    # ----- assign track ids -----
    def _assign_track_id(self, det, now):
        cx, cy = det['cx'], det['cy']
        cls_id = det['cls_id']

        best_id = None
        best_dist = 60.0
        for tid, (tcx, tcy, tcls, tt) in self.track_history.items():
            if tcls != cls_id:
                continue
            d = math.hypot(cx - tcx, cy - tcy)
            if d < best_dist:
                best_dist = d
                best_id = tid

        if best_id is not None:
            self.track_history[best_id] = (cx, cy, cls_id, now)
            return best_id
        else:
            new_id = self._next_track_id
            self._next_track_id += 1
            self.track_history[new_id] = (cx, cy, cls_id, now)
            return new_id

    # ----- publish detections -----
    def _publish(self, det, annotated, det2d_msg, det3d_msg, img_msg, now):
        cx, cy = det['cx'], det['cy']
        conf = det['conf']
        class_name = det['class_name']

        model_id = det['source']
        color = det['color'] # TODO: change color to super category color (in total 20 for COCO + DOTAv1 class categories + 1 called Cube for Cube Faces, read super category from some translation map)
        is_obb_result = (det['angle_rad'] is not None)
        is_pose_result = (det.get('keypoints') is not None)
        src_tag = "OBB" if is_obb_result else ("POSE" if is_pose_result else "AABB")
        track_id = self._assign_track_id(det, now)

        # Draw keypoints if available
        if is_pose_result and det['keypoints'] is not None:
            kpts = det['keypoints']
            kpt_conf = det.get('kpt_conf')
            self._draw_keypoints(annotated, kpts, kpt_conf)

        point_3d = None
        if (self.camera_source == "realsense"
                and self.depth_image is not None
                and self.camera_info is not None):
            point_3d = self.project_3d(cx, cy)

        if is_obb_result:
            angle_rad = det['angle_rad']
            angle_deg = math.degrees(angle_rad)
            rect = ((float(cx), float(cy)),
                    (float(det['w']), float(det['h'])), float(angle_deg))
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int32(box_pts)
            cv2.polylines(annotated, [box_pts], True, color, 2)
        else:
            x1, y1, x2, y2 = map(int, [det['x1'], det['y1'],
                                        det['x2'], det['y2']])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f} [{src_tag}]"
        if track_id is not None:
            label += f" ID:{track_id}"
        if point_3d:
            label += f" Z:{point_3d[2]:.2f}m"

        lx = int(cx - det['w'] / 2)
        ly = max(20, int(cy - det['h'] / 2 - 5))
        cv2.putText(annotated, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        det2d = Detection2D()
        det2d.bbox.center.position.x = float(cx)
        det2d.bbox.center.position.y = float(cy)
        det2d.bbox.size_x = float(det['w'])
        det2d.bbox.size_y = float(det['h'])

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = class_name
        hyp.hypothesis.score = conf
        if is_obb_result:
            hyp.pose.pose.orientation.z = math.sin(det['angle_rad'] / 2.0)
            hyp.pose.pose.orientation.w = math.cos(det['angle_rad'] / 2.0)

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

    # ----- image callback -----
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        now = time.time()

        det2d_msg = Detection2DArray()
        det2d_msg.header = msg.header
        det3d_msg = Detection3DArray()
        det3d_msg.header = msg.header
        annotated = frame.copy()

        if isinstance(self.models,list):
            predictions = [model.track(frame,persist=True, tracker=bb_tracker, conf=self.conf_threshold, verbose=False)[0] for model,bb_tracker in zip(self.models,self.bb_trackers)]
            candidates = []
            for preds,m_id,model,color in zip(predictions, self.model_ids, self.models, self.colors):
                candidates.extend(self._process_boxes(preds, m_id, model, color))
            fused = self._fuse_candidates(candidates)
            for det in fused: self._publish(det, annotated, det2d_msg, det3d_msg, msg, now)
        else:
            preds = self.models.track(
                frame, persist=True, tracker=self.bb_trackers,
                conf=self.conf_threshold, verbose=False)[0]
            for det in self._process_boxes(preds, self.model_ids, self.models, self.colors):
                self._publish(det, annotated, det2d_msg, det3d_msg, msg, now)

        self.det2d_pub.publish(det2d_msg)
        if len(det3d_msg.detections) > 0:
            self.det3d_pub.publish(det3d_msg)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# ros2 run yolo_ros2 yolo_node --ros-args -p model_type:=cubified
# ros2 run yolo_ros2 yolo_node --ros-args -p model_type:=yolo11
# ros2 run yolo_ros2 yolo_node --ros-args -p model_id:=yolo11-pose
# ros2 run yolo_ros2 yolo_node --ros-args -p model_id:=yolo26-pose
# Fusion: AABB + OBB
#   ros2 run yolo_ros2 yolo_node --ros-args -p model_id:="[yolo26, cubified]" -p model_type:="[AABB, OBB]"
# Fusion: AABB + OBB + Pose
#   ros2 run yolo_ros2 yolo_node --ros-args -p model_id:="[yolo26, cubified, yolo11-pose]" -p model_type:="[AABB, OBB, pose]"
