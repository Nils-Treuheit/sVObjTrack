import rclpy
from rclpy.node import Node

import numpy as np
import pyrealsense2 as rs

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String


class RealSenseNode(Node):
    def __init__(self):
        super().__init__("realsense_node")

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self.source_pub = self.create_publisher(String, "/camera/source_info", 10)

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_profile.get_intrinsics()

        self.camera_info = CameraInfo()
        self.camera_info.width = color_intr.width
        self.camera_info.height = color_intr.height
        self.camera_info.k = [
            color_intr.fx, 0.0, color_intr.ppx,
            0.0, color_intr.fy, color_intr.ppy,
            0.0, 0.0, 1.0
        ]
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info.p = [
            color_intr.fx, 0.0, color_intr.ppx, 0.0,
            0.0, color_intr.fy, color_intr.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        self.camera_info.distortion_model = "plumb_bob"

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.get_logger().info("RealSense camera started (depth + camera_info enabled)")

        self.timer = self.create_timer(1.0 / 30.0, self.loop)

    def loop(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        frame = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        stamp = self.get_clock().now().to_msg()

        color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = "realsense_camera"
        self.image_pub.publish(color_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="mono16")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "realsense_camera"
        self.depth_pub.publish(depth_msg)

        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = "realsense_camera"
        info_msg.width = self.camera_info.width
        info_msg.height = self.camera_info.height
        info_msg.k = self.camera_info.k
        info_msg.d = self.camera_info.d
        info_msg.r = self.camera_info.r
        info_msg.p = self.camera_info.p
        info_msg.distortion_model = self.camera_info.distortion_model
        self.info_pub.publish(info_msg)

        source = String()
        source.data = "realsense"
        self.source_pub.publish(source)


def main():
    rclpy.init()
    node = RealSenseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
