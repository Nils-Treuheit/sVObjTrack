"""
Orchestrator node — manages YOLO detection + LocateAnything subprocesses
and bridges text queries to LA responses.

By default starts YOLO fusion: yolo26 + yolo26-obb + yolo26-pose.
Pass a different model_id parameter to override (single model or fusion list).

Topics:
  Subscribed:
    /yolo/detections_2d    (vision_msgs/Detection2DArray)  — from managed YOLO
    /orchestrator/query    (std_msgs/String)               — text query from user
    /la/grounding_text     (std_msgs/String)               — LA query responses
    /camera/camera_info    (sensor_msgs/CameraInfo)        — image dimensions

  Published:
    /la/grounding_query    (std_msgs/String)               — forwarded to LA
    /orchestrator/response (std_msgs/String)               — query answer
"""
import os
import signal
import subprocess
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

LA_PROJECT = "/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/locate_anything"
LA_TRT = f"{LA_PROJECT}/model/tensorRT"

_TRT_LIBS = (
    f"{LA_TRT}/.venv/lib/python3.10/site-packages/tensorrt_libs:"
    f"{os.path.expanduser('~')}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:"
    "/usr/local/cuda-12.8/lib64"
)


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('model_id', '[yolo26, yolo26-obb, yolo26-pose]')
        model_id = self.get_parameter('model_id').value

        self._yolo_proc = None
        self._la_proc = None
        self.latest_yolo_dets = None
        self.img_w = 640
        self.img_h = 480
        self.query_pending = False

        # Subscribe to camera_info for image dimensions
        self._camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._camera_info_callback, 10)

        # Spawn subprocesses
        self._start_yolo(model_id)
        self._start_la()

        # Subscriptions
        self.create_subscription(Detection2DArray, '/yolo/detections_2d',
                                 self._yolo_callback, 10)
        self.create_subscription(String, '/orchestrator/query',
                                 self._query_in_callback, 10)
        self.create_subscription(String, '/la/grounding_text',
                                 self._la_response_callback, 10)

        # Publishers
        self._query_pub = self.create_publisher(String, '/la/grounding_query', 10)
        self._response_pub = self.create_publisher(String, '/orchestrator/response', 10)

        self.get_logger().info(
            f'[Orch] Started with model_id={model_id}. '
            f'Publish text queries to /orchestrator/query.')

    def _camera_info_callback(self, msg):
        self.img_w = msg.width
        self.img_h = msg.height

    def _start_yolo(self, model_id):
        python = sys.executable
        cmd = [
            python, '-m', 'yolo_ros2.yolo_node',
            '--ros-args', '-p', f"model_id:='{model_id}'",
        ]
        self.get_logger().info(f'[Orch] Spawning YOLO: {" ".join(cmd)}')
        try:
            self._yolo_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            self.get_logger().error(f'[Orch] Failed to start YOLO: {e}')

    def _start_la(self):
        python = sys.executable
        cmd = [python, '-m', 'locate_anything_ros2.la_node',
               '--ros-args', '-p', 'query_only:=True']
        env = os.environ.copy()
        existing = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f'{_TRT_LIBS}:{existing}' if existing else _TRT_LIBS
        self.get_logger().info(f'[Orch] Spawning LA: {" ".join(cmd)}')
        try:
            self._la_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            self.get_logger().error(f'[Orch] Failed to start LA: {e}')

    def _yolo_dets_to_context(self):
        """Convert latest YOLO detections to LA token-coordinate context string."""
        if self.latest_yolo_dets is None or not self.latest_yolo_dets.detections:
            return ''
        w, h = self.img_w, self.img_h
        if w == 0 or h == 0:
            return ''
        ctx_parts = []
        for det in self.latest_yolo_dets.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size_x
            sy = det.bbox.size_y
            x1 = int((max(0, cx - sx / 2) / w) * 1000)
            y1 = int((max(0, cy - sy / 2) / h) * 1000)
            x2 = int((min(w, cx + sx / 2) / w) * 1000)
            y2 = int((min(h, cy + sy / 2) / h) * 1000)
            cls = det.results[0].hypothesis.class_id if det.results else 'object'
            tid = det.id
            if tid:
                ctx_parts.append(f"<box>{x1},{y1},{x2},{y2}</box> {cls} ID:{tid}")
            else:
                ctx_parts.append(f"<box>{x1},{y1},{x2},{y2}</box> {cls}")
        if not ctx_parts:
            return ''
        ctx = ", ".join(ctx_parts)
        return f"Previously detected: {ctx}."

    def _yolo_callback(self, msg):
        self.latest_yolo_dets = msg

    def _query_in_callback(self, msg):
        query = msg.data.strip()
        if not query:
            return
        self.get_logger().info(f'[Orch] Query in: "{query}"')
        self.query_pending = True

        context = self._yolo_dets_to_context()
        if context:
            augmented = f"{context} {query}"
        else:
            augmented = query

        out = String()
        out.data = augmented
        self._query_pub.publish(out)

    def _la_response_callback(self, msg):
        if not self.query_pending:
            return
        self.query_pending = False

        response = msg.data
        self.get_logger().info(f'[Orch] LA response:\n{response}')
        yolo_count = len(self.latest_yolo_dets.detections) if self.latest_yolo_dets else 0

        enriched = f'{response}\n---\nYOLO detections in scene: {yolo_count}'

        out = String()
        out.data = enriched
        self._response_pub.publish(out)

    def _kill_proc(self, proc, name):
        if proc is None:
            return
        self.get_logger().info(f'[Orch] Shutting down {name}...')
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def destroy_node(self):
        self._kill_proc(self._yolo_proc, 'YOLO')
        self._kill_proc(self._la_proc, 'LA')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
