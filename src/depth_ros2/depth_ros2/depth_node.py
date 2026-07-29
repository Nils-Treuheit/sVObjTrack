"""Monocular depth estimation node using HYDEN-MoGe_v2 (metric point model).

Subscribes to /camera/image_raw and publishes:
  /camera/depth/image_raw  (sensor_msgs/Image, 32FC1, metric meters)
  /camera/camera_info       (sensor_msgs/CameraInfo with estimated intrinsics)

The metric point model outputs (1,H,W,3) XYZ coordinates per pixel.
Depth (Z channel) is extracted and published as the depth map.

Model lives in a separate venv discovered via model_registry.
Checkpoint is downloaded from HuggingFace Hub on first use.
"""
import os
import sys
import time
import threading

from model_registry import metadepth_venv, metadepth_checkpoint

# Metadepth venv — add to sys.path before any heavy imports
_md_sp = metadepth_venv() / 'lib' / 'python3.10' / 'site-packages'
if _md_sp.is_dir() and str(_md_sp) not in sys.path:
    sys.path.insert(0, str(_md_sp))

# Metadepth repo source
_md_repo = metadepth_venv().parent / 'repo'
if _md_repo.is_dir() and str(_md_repo) not in sys.path:
    sys.path.insert(0, str(_md_repo))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import cv2
import numpy as np


class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')

        self.declare_parameter('max_fps', 10.0)
        self.declare_parameter('device', '')
        self.declare_parameter('input_width', 518)
        self.declare_parameter('input_height', 518)

        self.max_fps = self.get_parameter('max_fps').value
        self.input_w = self.get_parameter('input_width').value
        self.input_h = self.get_parameter('input_height').value

        device_param = self.get_parameter('device').value

        self.bridge = CvBridge()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0

        import torch
        self.device = device_param if device_param else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')

        self._load_model()

        self._frame_sub = self.create_subscription(Image, '/camera/image_raw', self._frame_cb, 10)
        self._source_sub = self.create_subscription(String, '/camera/source_info', self._source_cb, 10)
        self._depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self._info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        self._inference_timer = self.create_timer(1.0 / self.max_fps, self._inference_tick)

        self.get_logger().info(
            f'Depth node ready (max_fps={self.max_fps}, '
            f'input={self.input_w}x{self.input_h})')

    def _load_model(self):
        import torch
        from mogev2 import HyDenMoGe, MODEL_CONFIGS

        t0 = time.time()
        self.get_logger().info('Loading HYDEN-MoGe_v2 metric point model...')

        ckpt = str(metadepth_checkpoint())
        self.get_logger().info(f'Checkpoint: {ckpt}')

        self.model = HyDenMoGe(**MODEL_CONFIGS['vitl_dinov2'])
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self.get_logger().info(f'Model loaded in {time.time() - t0:.1f}s')

        self._transform = __import__('torchvision').transforms.Compose([
            __import__('torchvision').transforms.Resize((self.input_h, self.input_w)),
            __import__('torchvision').transforms.ToTensor(),
        ])

    def _frame_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warning(f'Frame conversion failed: {e}')
            return
        with self._frame_lock:
            self._latest_frame = (frame, msg.header)

    def _source_cb(self, msg):
        self.get_logger().info(f'Camera source: {msg.data}', throttle_duration_sec=5.0)

    @torch.no_grad()
    def _inference_tick(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame, header = self._latest_frame
            self._latest_frame = None

        import torch
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)

        t0 = time.time()
        output = self.model(tensor)
        dt_ms = (time.time() - t0) * 1000

        points = output['points']  # (1, H, W, 3) — metric XYZ
        depth_map = points[0, :, :, 2].cpu().numpy()  # (H, W) Z channel in meters

        orig_h, orig_w = frame.shape[:2]
        if depth_map.shape[0] != orig_h or depth_map.shape[1] != orig_w:
            depth_map = cv2.resize(depth_map, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST)

        depth_msg = self.bridge.cv2_to_imgmsg(
            depth_map.astype(np.float32), encoding='32FC1')
        depth_msg.header = header
        self._depth_pub.publish(depth_msg)

        self._publish_camera_info(header, orig_w, orig_h)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'Depth: {dt_ms:.1f}ms, '
                f'depth range [{np.nanmin(depth_map):.2f}, {np.nanmax(depth_map):.2f}]m',
                throttle_duration_sec=2.0)

    def _publish_camera_info(self, header, width, height):
        msg = CameraInfo()
        msg.header = header
        msg.width = width
        msg.height = height

        fx = float(width) * 0.8
        fy = float(height) * 0.8
        cx = float(width) / 2.0
        cy = float(height) / 2.0

        msg.k = [fx, 0, cx,
                 0, fy, cy,
                 0, 0, 1]
        msg.p = [fx, 0, cx, 0,
                 0, fy, cy, 0,
                 0, 0, 1, 0]
        msg.distortion_model = 'plumb_bob'
        msg.d = [0, 0, 0, 0, 0]

        self._info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
