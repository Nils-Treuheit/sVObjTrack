"""
Orchestrator node — manages YOLO/LocateAnything/RF-DETR subprocesses
and bridges text queries to LA responses.

By default starts YOLO fusion: yolo26 + yolo26-obb + yolo26-pose.
Pass model_id='rfdetr' to use RF-DETR instead of YOLO.

Topics:
  Subscribed:
    /yolo/detections_2d    (vision_msgs/Detection2DArray)  — from managed YOLO
    /rfdetr/detections_2d  (vision_msgs/Detection2DArray)  — from managed RF-DETR
    /orchestrator/query    (std_msgs/String)               — text query from user
    /la/grounding_text     (std_msgs/String)               — LA query responses
    /camera/camera_info    (sensor_msgs/CameraInfo)        — image dimensions

  Published:
    /la/grounding_query    (std_msgs/String)               — forwarded to LA
    /orchestrator/response (std_msgs/String)               — query answer
"""
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

from model_registry import la_project, la_trt_dir, rfdetr_venv, sam3_venv

_TRT_LIBS = (
    f"{la_trt_dir()}/.venv/lib/python3.10/site-packages/tensorrt_libs:"
    f"{os.path.expanduser('~')}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:"
    "/usr/local/cuda-12.8/lib64"
)


def _find_venv_python():
    """Find the sVObjTrack venv Python, falling back to sys.executable."""
    # Walk up from this file to find .venv/bin/python3
    here = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = here / '.venv' / 'bin' / 'python3'
        if candidate.exists():
            return str(candidate)
        here = here.parent
    # Check workspace root
    ws = Path(__file__).resolve().parent.parent.parent.parent
    candidate = ws / '.venv' / 'bin' / 'python3'
    if candidate.exists():
        return str(candidate)
    return sys.executable


def parse_box_detections(text: str) -> list:
    """Extract (label, [x1,y1,x2,y2]) pairs from LA <box> output.

    Returns list of (class_name, [x1, y1, x2, y2]) in [0, 1000] token coords.
    """
    results = []
    for match in re.finditer(r'<box>(.+?)</box>', text):
        coords_str = match.group(1)
        coords = [float(p) for p in re.findall(r'[\d.]+', coords_str)]
        if len(coords) != 4:
            continue
        before = text[:match.start()]
        ref_matches = re.findall(r'<ref>(.+?)</ref>', before)
        if ref_matches:
            label = ref_matches[-1].strip()
        else:
            end = match.end()
            context = text[end:end + 60].strip().rstrip(',').rstrip('.').strip()
            label = ' '.join(context.split()[:3]) if context else "object"
        results.append((label, coords))
    return results


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('model_id', '[yolo26, yolo26-obb, yolo26-pose]')
        self.declare_parameter('use_depth', False)
        self.declare_parameter('use_sam3', False)
        model_id = self.get_parameter('model_id').value
        use_depth = self.get_parameter('use_depth').value
        use_sam3 = self.get_parameter('use_sam3').value

        self._yolo_proc = None
        self._la_proc = None
        self._rf_proc = None
        self._depth_proc = None
        self._sam3_proc = None
        self.latest_dets = None
        self.img_w = 640
        self.img_h = 480
        self.query_pending = False

        self._camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self._camera_info_callback, 10)

        # Determine which detection backend to use
        self.use_rfdetr = model_id.strip().lower() == 'rfdetr'
        if self.use_rfdetr:
            self._start_rfdetr()
            self.create_subscription(Detection2DArray, '/rfdetr/detections_2d',
                                     self._det_callback, 10)
        else:
            self._start_yolo(model_id)
            self.create_subscription(Detection2DArray, '/yolo/detections_2d',
                                     self._det_callback, 10)

        self._start_la()

        if use_depth:
            self._start_depth()

        if use_sam3:
            self._start_sam3()

        self.create_subscription(String, '/orchestrator/query',
                                 self._query_in_callback, 10)
        self.create_subscription(String, '/la/grounding_text',
                                 self._la_response_callback, 10)

        self._query_pub = self.create_publisher(String, '/la/grounding_query', 10)
        self._response_pub = self.create_publisher(String, '/orchestrator/response', 10)

        backend = 'RF-DETR' if self.use_rfdetr else f'YOLO({model_id})'
        self.get_logger().info(
            f'[Orch] Started with backend={backend}. '
            f'Publish text queries to /orchestrator/query.')

    def _camera_info_callback(self, msg):
        self.img_w = msg.width
        self.img_h = msg.height

    def _start_yolo(self, model_id):
        python = _find_venv_python()
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

    def _start_rfdetr(self):
        python = _find_venv_python()
        env = os.environ.copy()
        existing_pp = env.get('PYTHONPATH', '')
        rf_site = str(rfdetr_venv() / 'lib' / 'python3.10' / 'site-packages')
        env['PYTHONPATH'] = f'{rf_site}:{existing_pp}' if existing_pp else rf_site
        cmd = [python, '-m', 'rfdetr_ros2.rfdetr_node']
        self.get_logger().info(f'[Orch] Spawning RF-DETR: {" ".join(cmd)}')
        try:
            self._rf_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            self.get_logger().error(f'[Orch] Failed to start RF-DETR: {e}')

    def _start_la(self):
        python = _find_venv_python()
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

    def _start_depth(self):
        python = _find_venv_python()
        cmd = [python, '-m', 'depth_ros2.depth_node']
        self.get_logger().info(f'[Orch] Spawning depth node: {" ".join(cmd)}')
        try:
            self._depth_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            self.get_logger().error(f'[Orch] Failed to start depth node: {e}')

    def _start_sam3(self):
        sam3_python = str(sam3_venv() / 'bin' / 'python3')
        if not os.path.isfile(sam3_python):
            self.get_logger().error(f'[Orch] SAM3 venv Python not found: {sam3_python}')
            return
        cmd = [sam3_python, '-m', 'sam3_ros2.sam3_node']
        self.get_logger().info(f'[Orch] Spawning SAM 3.1: {" ".join(cmd)}')
        try:
            self._sam3_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            self.get_logger().error(f'[Orch] Failed to start SAM 3.1: {e}')

    def _yolo_dets_to_context(self):
        """Convert latest detections to LA token-coordinate context string."""
        if self.latest_dets is None or not self.latest_dets.detections:
            return ''
        w, h = self.img_w, self.img_h
        if w == 0 or h == 0:
            return ''
        ctx_parts = []
        for det in self.latest_dets.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size_x
            sy = det.bbox.size_y
            x1 = int((max(0, cx - sx / 2) / w) * 1000)
            y1 = int((max(0, cy - sy / 2) / h) * 1000)
            x2 = int((min(w, cx + sx / 2) / w) * 1000)
            y2 = int((min(h, cy + sy / 2) / h) * 1000)
            cls = det.results[0].hypothesis.class_id if det.results else 'object'
            ctx_parts.append(f"<box>{x1},{y1},{x2},{y2}</box> {cls}")
        if not ctx_parts:
            return ''
        ctx = ", ".join(ctx_parts)
        return f"[ Previous Predictions (Context): {ctx} ]"

    def _det_callback(self, msg):
        self.latest_dets = msg

    def _query_in_callback(self, msg):
        query = msg.data.strip()
        if not query:
            return
        self.get_logger().info(f'[Orch] Query in: "{query}"')
        self.query_pending = True

        context = self._yolo_dets_to_context()
        if context:
            augmented = f"Given the Context answer the Query; {context}; Query: {query}"
        else:
            augmented = f"Given the Context answer the Query; Query: {query}"

        out = String()
        out.data = augmented
        self._query_pub.publish(out)

    def _la_response_callback(self, msg):
        if not self.query_pending:
            return
        self.query_pending = False

        response = msg.data
        self.get_logger().info(f'[Orch] LA response:\n{response}')

        # Parse any <box> detections from LA response
        dets = parse_box_detections(response)
        w, h = self.img_w, self.img_h
        if dets and w > 0 and h > 0:
            lines = []
            for label, token_box in dets:
                x1 = int((token_box[0] / 1000.0) * w)
                y1 = int((token_box[1] / 1000.0) * h)
                x2 = int((token_box[2] / 1000.0) * w)
                y2 = int((token_box[3] / 1000.0) * h)
                lines.append(f"  {label} @ [{x1} {y1} {x2} {y2}]")
            det_str = '\n'.join(lines)
            self.get_logger().info(f'[Orch] Box detections:\n{det_str}')

        yolo_count = len(self.latest_dets.detections) if self.latest_dets else 0

        enriched = f'{response}\n---\nDetections in scene: {yolo_count}'

        out = String()
        out.data = enriched
        self._response_pub.publish(out)

    def _kill_proc(self, proc, name):
        if proc is None:
            return
        if proc.poll() is not None:
            self.get_logger().info(f'[Orch] {name} already exited (code={proc.returncode})')
            return
        self.get_logger().info(f'[Orch] Shutting down {name}...')
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.get_logger().warning(f'[Orch] {name} did not exit after SIGTERM, sending SIGKILL')
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.get_logger().error(f'[Orch] {name} survived SIGKILL')
        except ProcessLookupError:
            pass
        except Exception as e:
            self.get_logger().warning(f'[Orch] Error killing {name}: {e}')
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

    def destroy_node(self):
        self._kill_proc(self._yolo_proc, 'YOLO')
        self._kill_proc(self._rf_proc, 'RF-DETR')
        self._kill_proc(self._depth_proc, 'Depth')
        self._kill_proc(self._sam3_proc, 'SAM3')
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
