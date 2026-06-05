import rclpy
from rclpy.node import Node

import numpy as np
import pyrealsense2 as rs

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class RealSenseNode(Node):
    def __init__(self):
        super().__init__("realsense_node")

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.source_pub = self.create_publisher(String, "/camera/source_info", 10)

        # -------------------------
        # REALSENSE PIPELINE
        # -------------------------
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline.start(config)

        self.get_logger().info("RealSense camera started")

        self.timer = self.create_timer(1.0 / 30.0, self.loop)

    def loop(self):
        frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "realsense_camera"

        self.image_pub.publish(msg)

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

# ros2 run camera_nodes realsense_camera