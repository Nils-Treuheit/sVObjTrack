"""tkinter viewer for sVObjTrack — displays camera feed with detections and VQA.

Layout:
  Left  (60%):  Camera feed (debug image with annotations)
                Status bar below: Connecting..., Connected, Failed: <Error>
  Right (40%):  Tracking log (formatted) + VQA query/response
"""
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection3DArray
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

try:
    from PIL import Image as PILImage, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

import tkinter as tk
from tkinter import ttk, scrolledtext


class ViewerNode(Node):
    def __init__(self):
        super().__init__('viewer_node')
        self.bridge = CvBridge()
        self._cv_image = None
        self._det2d = []
        self._det3d = []
        self._tk_img = None
        self._camera_connected = False
        self._model_running = False
        self._last_error = None

        self.create_subscription(Image, '/yolo/debug_image', self._cb_yolo_img, 10)
        self.create_subscription(Image, '/la/debug_image', self._cb_la_img, 10)
        self.create_subscription(Image, '/rfdetr/debug_image', self._cb_rfdetr_img, 10)
        self.create_subscription(Image, '/sam3/debug_image', self._cb_sam3_img, 10)
        self.create_subscription(Detection2DArray, '/yolo/detections_2d', self._cb_yolo_2d, 10)
        self.create_subscription(Detection2DArray, '/la/detections_2d', self._cb_la_2d, 10)
        self.create_subscription(Detection2DArray, '/rfdetr/detections_2d', self._cb_rfdetr_2d, 10)
        self.create_subscription(Detection3DArray, '/yolo/detections_3d', self._cb_yolo_3d, 10)
        self.create_subscription(Detection3DArray, '/la/detections_3d', self._cb_la_3d, 10)
        self.create_subscription(String, '/orchestrator/response', self._cb_vqa_resp, 10)
        self._vqa_pub = self.create_publisher(String, '/orchestrator/query', 10)

        self._build_gui()
        self._tick()

    # ------------------------------------------------------------------ GUI
    def _build_gui(self):
        self._root = tk.Tk()
        self._root.title('sVObjTrack Viewer')
        self._root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._root.configure(bg='#111111')
        self._root.geometry('1600x900')
        self._root.minsize(1000, 600)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabelframe', background='#2b2b2b', foreground='white')
        style.configure('TLabelframe.Label', background='#2b2b2b',
                        foreground='#8ab4f8', font=('Consolas', 11, 'bold'))
        style.configure('TButton', background='#3c3f41', foreground='white')
        style.configure('TEntry', fieldbackground='#313335',
                        foreground='white', insertcolor='white')
        style.configure('TFrame', background='#2b2b2b')

        # --- main horizontal split ---
        outer = ttk.Frame(self._root)
        outer.pack(fill=tk.BOTH, expand=True)

        # Left panel (60%)
        left = tk.Frame(outer, bg='#111111')
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera feed canvas
        self._canvas = tk.Canvas(left, bg='#111111', highlightthickness=0)
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._canvas.bind('<Configure>', self._on_canvas_resize)

        # Status bar below camera
        self._status_var = tk.StringVar(value='Connecting camera...')
        status_frame = tk.Frame(left, bg='#1a1a1a', height=28)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)

        self._status_dot = tk.Canvas(status_frame, width=14, height=14,
                                     bg='#1a1a1a', highlightthickness=0)
        self._status_dot.pack(side=tk.LEFT, padx=(8, 4), pady=7)
        self._status_label = tk.Label(status_frame, textvariable=self._status_var,
                                      bg='#1a1a1a', fg='#888888',
                                      font=('Consolas', 10), anchor='w')
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Vertical separator
        sep = tk.Frame(outer, width=3, bg='#444444')
        sep.pack(side=tk.LEFT, fill=tk.Y)

        # Right panel (40%)
        right = tk.Frame(outer, width=480, bg='#2b2b2b')
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # Tracking log
        track_frame = ttk.LabelFrame(right, text='Tracking Log')
        track_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))
        self._track_text = scrolledtext.ScrolledText(
            track_frame, bg='#1e1e1e', fg='#d4d4d4',
            font=('Consolas', 10), insertbackground='white', wrap=tk.NONE)
        self._track_text.pack(fill=tk.BOTH, expand=True)
        self._track_text.tag_configure('header', foreground='#8ab4f8', font=('Consolas', 10, 'bold'))
        self._track_text.tag_configure('separator', foreground='#555555')
        self._track_text.tag_configure('data', foreground='#d4d4d4')

        # VQA
        vqa_frame = ttk.LabelFrame(right, text='VQA')
        vqa_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        entry_frame = ttk.Frame(vqa_frame)
        entry_frame.pack(fill=tk.X, padx=4, pady=4)
        self._vqa_entry = ttk.Entry(entry_frame)
        self._vqa_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._vqa_entry.bind('<Return>', lambda _: self._send_query())
        ttk.Button(entry_frame, text='Send', command=self._send_query).pack(side=tk.RIGHT)

        self._vqa_resp = scrolledtext.ScrolledText(
            vqa_frame, height=5, bg='#1e1e1e', fg='#d4d4d4',
            font=('Consolas', 10), insertbackground='white', wrap=tk.WORD)
        self._vqa_resp.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

    def _on_canvas_resize(self, _event):
        self._render_image()

    # --------------------------------------------------------- VQA helpers
    def _send_query(self):
        q = self._vqa_entry.get().strip()
        if not q:
            return
        msg = String()
        msg.data = q
        self._vqa_pub.publish(msg)
        self._vqa_entry.delete(0, tk.END)
        self._vqa_resp.insert(tk.END, f'> {q}\n')
        self._vqa_resp.see(tk.END)

    # ------------------------------------------------------ ROS callbacks
    def _cb_yolo_img(self, msg):
        try:
            self._cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if not self._camera_connected:
                self._camera_connected = True
                self._last_error = None
        except Exception as e:
            self._last_error = str(e)

    def _cb_la_img(self, msg):
        try:
            self._cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if not self._camera_connected:
                self._camera_connected = True
                self._last_error = None
        except Exception as e:
            self._last_error = str(e)

    def _cb_rfdetr_img(self, msg):
        try:
            self._cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if not self._camera_connected:
                self._camera_connected = True
                self._last_error = None
        except Exception as e:
            self._last_error = str(e)

    def _cb_sam3_img(self, msg):
        try:
            self._cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if not self._camera_connected:
                self._camera_connected = True
                self._last_error = None
        except Exception as e:
            self._last_error = str(e)

    def _cb_yolo_2d(self, msg):
        self._det2d = list(msg.detections)
        if self._det2d and not self._model_running:
            self._model_running = True

    def _cb_la_2d(self, msg):
        self._det2d = list(msg.detections)
        if self._det2d and not self._model_running:
            self._model_running = True

    def _cb_rfdetr_2d(self, msg):
        self._det2d = list(msg.detections)
        if self._det2d and not self._model_running:
            self._model_running = True

    def _cb_yolo_3d(self, msg):
        self._det3d = list(msg.detections)

    def _cb_la_3d(self, msg):
        self._det3d = list(msg.detections)

    def _cb_vqa_resp(self, msg):
        self._vqa_resp.insert(tk.END, f'{msg.data}\n')
        self._vqa_resp.see(tk.END)

    # -------------------------------------------------------- periodic tick
    def _tick(self):
        rclpy.spin_once(self, timeout_sec=0)
        self._render_image()
        self._render_tracking()
        self._update_status()
        self._root.after(33, self._tick)

    def _update_status(self):
        if self._last_error:
            self._status_var.set(f'Failed: {self._last_error}')
            self._status_label.configure(fg='#f44336')
            self._draw_status_dot('#f44336')
        elif self._model_running:
            self._status_var.set('Camera connected & model running')
            self._status_label.configure(fg='#4caf50')
            self._draw_status_dot('#4caf50')
        elif self._camera_connected:
            self._status_var.set('Camera connected, loading model...')
            self._status_label.configure(fg='#ff9800')
            self._draw_status_dot('#ff9800')
        else:
            self._status_var.set('Connecting camera...')
            self._status_label.configure(fg='#888888')
            self._draw_status_dot('#888888')

    def _draw_status_dot(self, color):
        self._status_dot.delete('all')
        self._status_dot.create_oval(3, 3, 11, 11, fill=color, outline='')

    def _render_image(self):
        if self._cv_image is None or not _HAS_PIL:
            return
        rgb = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)

        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        iw, ih = pil.size
        scale = min(cw / iw, ch / ih)
        new_w = max(int(iw * scale), 1)
        new_h = max(int(ih * scale), 1)
        pil = pil.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(pil)
        self._canvas.delete('all')
        x0 = (cw - new_w) // 2
        y0 = (ch - new_h) // 2
        self._canvas.create_image(x0, y0, anchor=tk.NW, image=self._tk_img)

    def _render_tracking(self):
        self._track_text.configure(state=tk.NORMAL)
        self._track_text.delete('1.0', tk.END)

        if not self._det2d and not self._det3d:
            self._track_text.insert(tk.END, 'Waiting for detections...\n')
            self._track_text.configure(state=tk.DISABLED)
            return

        # Merge 2D and 3D detections by class
        has_3d = bool(self._det3d)

        header = f'{"#":>3}  {"ID":>4}  {"Class":<16} {"BBox [x,y,z w*h*d angle]":<30} {"Conf":>5}'
        self._track_text.insert(tk.END, header + '\n', 'header')
        self._track_text.insert(tk.END, '─' * 64 + '\n', 'separator')

        # Track 3D detection index
        d3_idx = 0

        for i, det in enumerate(self._det2d):
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx, sy = det.bbox.size_x, det.bbox.size_y
            hyp = det.results[0].hypothesis if det.results else None
            cls = hyp.class_id if hyp else '?'
            sc = hyp.score if hyp else 0.0

            # Try to get orientation from 2D results
            angle = 0.0
            if det.results and det.results[0].pose.pose.orientation.w != 0:
                import math
                qz = det.results[0].pose.pose.orientation.z
                qw = det.results[0].pose.pose.orientation.w
                angle = math.degrees(2 * math.atan2(qz, qw))

            # Get 3D position if available
            bbox_str = f'[{cx:.0f},{cy:.0f} {sx:.0f}x{sy:.0f}'
            if d3_idx < len(self._det3d):
                p = self._det3d[d3_idx].bbox.center.position
                bbox_str = f'[{p.x:.2f},{p.y:.2f},{p.z:.2f} {sx:.0f}x{sy:.0f}'
                if angle != 0:
                    bbox_str += f' {angle:.0f}\u00b0'
                bbox_str += ']'
                d3_idx += 1
            else:
                if angle != 0:
                    bbox_str += f' {angle:.0f}\u00b0'
                bbox_str += ']'

            # Get track_id if available (from OBBox or custom field)
            track_id = getattr(det, 'track_id', None) if hasattr(det, 'track_id') else None
            id_str = f'{track_id}' if track_id is not None else '---'

            line = f'{i+1:>3}  {id_str:>4}  {cls:<16} {bbox_str:<30} {sc:>5.2f}'
            self._track_text.insert(tk.END, line + '\n', 'data')

        self._track_text.configure(state=tk.DISABLED)

    # ---------------------------------------------------------- lifecycle
    def _on_close(self):
        self._root.destroy()

    def spin(self):
        self._root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
