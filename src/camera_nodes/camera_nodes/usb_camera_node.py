import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


class USBCameraNode(Node):
    def __init__(self):
        super().__init__("usb_camera_node")

        self.declare_parameter("device_id", 0)
        device_id = self.get_parameter("device_id").value

        self.cap = cv2.VideoCapture(device_id)

        if not self.cap.isOpened():
            self.get_logger().error("USB camera not found")
            raise RuntimeError("Camera not available")

        self.bridge = CvBridge()

        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.source_pub = self.create_publisher(String, "/camera/source_info", 10)

        self.timer = self.create_timer(1.0 / 30.0, self.loop)

        self.get_logger().info("USB camera node started (30 FPS)")

    def loop(self):
        ret, frame = self.cap.read()

        if not ret:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "usb_camera"

        self.pub.publish(msg)

        source = String()
        source.data = "usb"
        self.source_pub.publish(source)


def main():
    rclpy.init()
    node = USBCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

# ros2 run camera_nodes usb_camera