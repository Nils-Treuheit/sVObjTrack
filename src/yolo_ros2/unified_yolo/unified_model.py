"""
Unified YOLO26 OBB model (161-class: COCO + DOTA + cubes).

Supports two architectures trained via ``train.py``:
  - ``--arch unified``  → SingleHeadOBB (single backbone + OBB26 head)
  - ``--arch fused``    → DualBackbone (COCO + DOTA backbones + fusion + OBB26 head)

Both expose the same ultralytics-compatible inference interface so they plug
into the existing yolo_node.py fusion pipeline.

All heavy imports (torch, ultralytics) are lazy inside constructors
so module-level import of ``unified_yolo`` is fast.
``import torch`` and ``import torch.nn`` remain here because the class
definitions inherit from ``nn.Module`` — this is deferred to the point
where ``UnifiedYOLO`` is first imported (inside ``yolo_node.__init__``).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model_registry import MODELS_DIR

# Default base weights — resolved via MODELS_DIR at import time
BASE_WEIGHTS = str(MODELS_DIR / "yolo26m.pt")

# Default base weights for fused (dual-backbone) architecture
BASE_WEIGHTS_COCO = str(MODELS_DIR / "yolo26s.pt")
BASE_WEIGHTS_DOTA = str(MODELS_DIR / "yolo26s-obb.pt")

DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

CUBE_CLASS_NAMES = [
    "0_B", "0_W", "1_B", "1_W", "2_B", "2_W", "3_B", "3_W", "4_B", "4_W",
    "5_B", "5_W", "6_B", "6_W", "7_B", "7_W", "8_B", "8_W", "9_B", "9_W",
    "A_B", "A_W", "B_B", "B_W", "C_B", "C_W", "D_B", "D_W", "E_B", "E_W",
    "F_B", "F_W", "G_B", "G_W", "H_B", "H_W", "I_B", "I_W", "J_B", "J_W",
    "K_B", "K_W", "L_B", "L_W", "M_B", "M_W", "N_B", "N_W", "O_B", "O_W",
    "P_B", "P_W", "Q_B", "Q_W", "R_B", "R_W", "S_B", "S_W", "T_B", "T_W",
    "U_B", "U_W", "V_B", "V_W", "W_B", "W_W",
]

ALL_CLASS_NAMES = COCO_CLASSES + DOTA_CLASSES + CUBE_CLASS_NAMES  # 161
NAMES_DICT = {i: name for i, name in enumerate(ALL_CLASS_NAMES)}


class _OBBItem:
    """Lightweight obb-like item mimicking ultralytics' obb result element."""
    def __init__(self, xywhr, conf, cls_id, track_id=None):
        self._xywhr = torch.tensor([xywhr], dtype=torch.float32)
        self._conf = torch.tensor([conf], dtype=torch.float32)
        self._cls = torch.tensor([cls_id], dtype=torch.float32)
        self._id = torch.tensor([track_id], dtype=torch.float32) if track_id is not None else None

    @property
    def xywhr(self):
        return self._xywhr

    @property
    def conf(self):
        return self._conf

    @property
    def cls(self):
        return self._cls

    @property
    def id(self):
        return self._id


class _UnifiedResults:
    """Mimics ultralytics Results for _process_boxes compatibility."""
    boxes = None          # no AABB; _process_boxes falls through to .obb

    def __init__(self, boxes_xywhr, scores, cls_ids):
        self._obbs = []
        for xywhr, score, cid in zip(boxes_xywhr, scores, cls_ids):
            self._obbs.append(_OBBItem(xywhr, score, cid))

    @property
    def obb(self):
        return self._obbs if self._obbs else None


# ── helpers for feature channel detection ──

def _detect_feature_channels(backbone, head_input_idx):
    """Detect output channels of the layers feeding the head."""
    chs = []
    for idx in head_input_idx:
        layer = backbone[idx]
        if hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv'):
            chs.append(layer.cv1.conv.out_channels)
        elif hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
            chs.append(layer.cv2.conv.out_channels)
        elif hasattr(layer, 'conv'):
            chs.append(layer.conv.out_channels)
        else:
            for m in layer.modules():
                if isinstance(m, torch.nn.Conv2d):
                    chs.append(m.out_channels)
                    break
    return chs


def _load_compatible(src_module, dst_module):
    """Copy weights where shapes match, skip incompatible keys."""
    src_sd = src_module.state_dict()
    dst_sd = dst_module.state_dict()
    compatible = {}
    for key in src_sd:
        if key in dst_sd and src_sd[key].shape == dst_sd[key].shape:
            compatible[key] = src_sd[key]
    if compatible:
        dst_module.load_state_dict(compatible, strict=False)


def _copy_class_slice_expand(src_module, dst_module, dst_offset, n_copy):
    """Copy class weights from a narrower-input conv to a wider-input conv."""
    src_last = src_module[-1]
    dst_last = dst_module[-1]
    if not isinstance(src_last, nn.Conv2d) or not isinstance(dst_last, nn.Conv2d):
        return
    n = min(n_copy, src_last.out_channels, max(dst_last.out_channels - dst_offset, 0))
    if n <= 0:
        return
    src_in = src_last.in_channels
    dst_in = dst_last.in_channels
    with torch.no_grad():
        if src_in >= dst_in:
            dst_last.weight[dst_offset:dst_offset + n] = src_last.weight[:n, :dst_in]
        else:
            padded = dst_last.weight.new_zeros(n, dst_in, 1, 1)
            padded[:, :src_in] = src_last.weight[:n]
            dst_last.weight[dst_offset:dst_offset + n] = padded
        if dst_last.bias is not None and src_last.bias is not None:
            dst_last.bias[dst_offset:dst_offset + n] = src_last.bias[:n].clone()


def _mixed_init_head(head, coco_head, dota_obb_head):
    """Initialise unified OBB26 head from COCO Detect + DOTA OBB pretrained heads."""
    for i in range(len(dota_obb_head.cv2)):
        _load_compatible(dota_obb_head.cv2[i], head.cv2[i])
        if hasattr(dota_obb_head, "one2one_cv2"):
            _load_compatible(dota_obb_head.one2one_cv2[i], head.one2one_cv2[i])
    if hasattr(dota_obb_head, "cv4") and hasattr(head, "cv4"):
        for i in range(len(dota_obb_head.cv4)):
            _load_compatible(dota_obb_head.cv4[i], head.cv4[i])
            if hasattr(dota_obb_head, "one2one_cv4"):
                _load_compatible(dota_obb_head.one2one_cv4[i], head.one2one_cv4[i])
    for i in range(len(dota_obb_head.cv3)):
        _load_compatible(dota_obb_head.cv3[i], head.cv3[i])
        if hasattr(dota_obb_head, "one2one_cv3"):
            _load_compatible(dota_obb_head.one2one_cv3[i], head.one2one_cv3[i])
    for i in range(len(dota_obb_head.cv3)):
        _copy_class_slice_expand(coco_head.cv3[i], head.cv3[i], 0, 80)
        _copy_class_slice_expand(dota_obb_head.cv3[i], head.cv3[i], 80, 15)
        if hasattr(coco_head, "one2one_cv3") and hasattr(head, "one2one_cv3"):
            _copy_class_slice_expand(coco_head.one2one_cv3[i], head.one2one_cv3[i], 0, 80)
            _copy_class_slice_expand(dota_obb_head.one2one_cv3[i], head.one2one_cv3[i], 80, 15)


def _forward_backbone(model_list, x, save_indices, head_input_idx):
    """Run forward pass through a YOLO backbone+neck and collect feature maps."""
    y = [None] * len(model_list)
    out = []
    for m in model_list:
        if m.f != -1:
            x_in = [y[j] if isinstance(j, int) and j != -1 else x if j == -1 else x for j in m.f]
            if isinstance(x_in, list):
                x_in = x_in if len(x_in) > 1 else x_in[0]
            else:
                x_in = x_in
        else:
            x_in = x
        x = m(x_in)
        y[m.i] = x if m.i in save_indices else None
        if m.i in head_input_idx:
            out.append(x)
    return out


# ── model wrappers ──

class SingleHeadOBB(nn.Module):
    """Single-head OBB model — identical architecture to training.

    Backbone from yolo26m.pt, head is a fresh OBB26(161) with weights
    loaded from a unified training checkpoint.
    """

    def __init__(self, base_model, nc=161, args=None):
        super().__init__()
        from ultralytics.nn.modules import OBB26

        self.model = base_model.model
        self.save = base_model.save
        self.stride = base_model.stride
        self.nc = nc
        self.args = args

        orig_head = self.model[-1]
        self.head_input_idx = orig_head.f

        ch = [cv[0].conv.in_channels for cv in orig_head.cv2]

        self.head = OBB26(
            nc=nc, ne=1, reg_max=orig_head.reg_max,
            end2end=True, ch=ch,
        )
        stride = self.stride
        self.head.stride = stride.clone() if hasattr(stride, 'clone') else stride

    def forward_features(self, x):
        y, out = [], []
        for m in self.model:
            if m.f != -1:
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in self.head_input_idx:
                out.append(x)
        return out

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)


class DualBackboneOBB(nn.Module):
    """Dual-backbone fused model — identical architecture to training.

    Two YOLO backbones (COCO + DOTA) → feature concat → fusion convs → OBB26 head.
    """

    def __init__(self, coco_weights, dota_weights, nc=161, args=None):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.nn.modules import OBB26

        coco_base = YOLO(coco_weights)
        self.backbone_coco = coco_base.model.model

        dota_base = YOLO(dota_weights)
        self.backbone_dota = dota_base.model.model

        self.stride = dota_base.model.stride
        self.save = dota_base.model.save
        self.head_input_idx = dota_base.model.model[-1].f

        self.nc = nc
        self.args = args

        coco_chs = _detect_feature_channels(self.backbone_coco, self.head_input_idx)
        dota_chs = _detect_feature_channels(self.backbone_dota, self.head_input_idx)
        fused_chs = [min(256, c + d) for c, d in zip(coco_chs, dota_chs)]

        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + d, f, 3, padding=1, bias=False),
                nn.BatchNorm2d(f),
                nn.SiLU(inplace=True),
            )
            for c, d, f in zip(coco_chs, dota_chs, fused_chs)
        ])

        self.head = OBB26(
            nc=nc, ne=1, reg_max=1,
            end2end=True, ch=fused_chs,
        )
        self.head.stride = self.stride.clone() if hasattr(self.stride, "clone") else self.stride

        coco_head = coco_base.model.model[-1]
        dota_obb_head = dota_base.model.model[-1]
        _mixed_init_head(self.head, coco_head, dota_obb_head)

    def forward_features(self, x):
        """Run both backbones, fuse features, return head input list."""
        save_set = set(self.save)
        feats_coco = _forward_backbone(
            self.backbone_coco, x, save_set, self.head_input_idx)
        feats_dota = _forward_backbone(
            self.backbone_dota, x, save_set, self.head_input_idx)
        fused = []
        for fc, fd, conv in zip(feats_coco, feats_dota, self.fusion):
            fused.append(conv(torch.cat([fc, fd], dim=1)))
        return fused

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)


# ── public interface ──

class UnifiedYOLO(nn.Module):
    """Unified YOLO26 wrapper for ROS2 inference.

    Automatically detects architecture from checkpoint:
      - ``arch=fused``    → DualBackboneOBB (COCO s + DOTA s-obb + fusion)
      - ``arch=unified``  → SingleHeadOBB (COCO m + OBB26 head)

    Usage:
        model = UnifiedYOLO("runs/moe/fused_v2/weights/last.pt")
        results = model.track(frame, ...)
        for obb in results[0].obb:
            print(obb.xywhr, obb.conf, obb.cls)
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.checkpoint_path = str(checkpoint_path)
        self._build_model()

    def _build_model(self):
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils import IterableSimpleNamespace

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        arch = ckpt.get("arch", "unified")
        sd = ckpt["model_state_dict"]

        if arch == "fused":
            self._build_fused(sd, ckpt)
        else:
            self._build_single(sd, ckpt)

        self.inner.eval()
        self.names = {**NAMES_DICT}
        print(f"[UnifiedYOLO] Loaded {self.checkpoint_path} "
              f"(arch={arch}, epoch {ckpt.get('epoch', '?')}, "
              f"{len(NAMES_DICT)} classes)")

    def _build_single(self, sd, ckpt):
        """Build SingleHeadOBB from yolo26m.pt base."""
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils import IterableSimpleNamespace

        base = YOLO(BASE_WEIGHTS)
        base_model = base.model

        args_dict = dict(base_model.args) if base_model.args else {}
        hyp = {**DEFAULT_CFG_DICT, "box": 7.5, "cls": 0.5, "dfl": 1.5,
               "angle": 7.5, **args_dict}
        base_model.args = IterableSimpleNamespace(**hyp)

        self.inner = SingleHeadOBB(base_model, nc=161, args=base_model.args)
        self.inner.to(self.device)

        missing, unexpected = self.inner.load_state_dict(sd, strict=False)
        if missing:
            print(f"[UnifiedYOLO] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[UnifiedYOLO] Unexpected keys: {len(unexpected)}")

    def _build_fused(self, sd, ckpt):
        """Build DualBackboneOBB from yolo26s.pt + yolo26s-obb.pt bases."""
        from model_registry import resolve_checkpoint, MODELS_DIR
        try:
            coco_weights = str(resolve_checkpoint('yolo26s.pt'))
            dota_weights = str(resolve_checkpoint('yolo26s-obb.pt'))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Fused architecture requires yolo26s.pt and yolo26s-obb.pt "
                f"in {MODELS_DIR}: {e}")

        print(f"[UnifiedYOLO] Fused arch: COCO={coco_weights}, DOTA={dota_weights}")

        self.inner = DualBackboneOBB(coco_weights, dota_weights, nc=161)
        self.inner.to(self.device)

        missing, unexpected = self.inner.load_state_dict(sd, strict=False)
        if missing:
            print(f"[UnifiedYOLO] Missing keys (expected for architecture differences): "
                  f"{len(missing)}")
        if unexpected:
            print(f"[UnifiedYOLO] Unexpected keys: {len(unexpected)}")

    @torch.no_grad()
    def _decode_head(self, preds_dict, conf_thresh=0.25, max_det=300):
        """Decode one2one predictions into box lists.

        Returns per-image lists of (boxes_xywhr, scores, cls_ids).
        """
        from ultralytics.utils.loss import make_anchors
        from ultralytics.utils.tal import dist2rbox

        one2one = preds_dict["one2one"]
        box_dist = one2one["boxes"]
        scores = one2one["scores"]
        angle = one2one["angle"]
        feat_list = one2one["feats"]

        B = box_dist.shape[0]
        na = box_dist.shape[-1]
        reg_max = self.inner.head.reg_max

        box_dist = box_dist.permute(0, 2, 1).contiguous()
        box_dist = box_dist.view(B, na, 4, reg_max).softmax(-1)
        proj = torch.arange(reg_max, dtype=box_dist.dtype, device=box_dist.device)
        pred_dist = box_dist.matmul(proj)

        anchor_points, stride_tensor = make_anchors(feat_list, self.inner.head.stride, 0.5)
        pred_angle = angle.permute(0, 2, 1).contiguous()
        decoded = dist2rbox(pred_dist, pred_angle, anchor_points)
        decoded = decoded * stride_tensor
        scores_sig = scores.sigmoid().permute(0, 2, 1).contiguous()

        out_boxes, out_scores, out_classes = [], [], []
        for b in range(B):
            max_scores, max_cls = scores_sig[b].max(dim=-1)
            mask = max_scores > conf_thresh
            if not mask.any():
                out_boxes.append(np.empty((0, 5)))
                out_scores.append(np.empty((0,)))
                out_classes.append(np.empty((0,), dtype=int))
                continue
            sel_boxes = decoded[b, mask].cpu().numpy()
            sel_scores = max_scores[mask].cpu().numpy()
            sel_cls = max_cls[mask].cpu().numpy()
            order = np.argsort(-sel_scores)[:max_det]
            out_boxes.append(sel_boxes[order])
            out_scores.append(sel_scores[order])
            out_classes.append(sel_cls[order])
        return out_boxes, out_scores, out_classes

    @torch.no_grad()
    def forward(self, imgs):
        feats = self.inner.forward_features(imgs)
        head_out = self.inner.head(feats)
        preds_dict = head_out[1] if isinstance(head_out, (tuple, list)) else head_out
        return preds_dict

    def track(self, imgs, persist=False, tracker=None, conf=0.25, verbose=False, **kwargs):
        """Ultralytics-compatible track interface.

        Args:
            imgs: np.ndarray (H,W,3) BGR or batch (B,H,W,3)
            persist: unused (tracking handled externally)
            tracker: unused
            conf: confidence threshold
            verbose: unused

        Returns:
            list of _UnifiedResults, one per image.
        """
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[None]  # (1, H, W, 3)
            imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).to(self.device).float() / 255.0
        elif isinstance(imgs, torch.Tensor):
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(self.device).float() / 255.0
        else:
            raise TypeError(f"Unsupported input type: {type(imgs)}")

        preds_dict = self.forward(imgs)
        boxes_list, scores_list, cls_list = self._decode_head(
            preds_dict, conf_thresh=conf, max_det=300)

        results = []
        for boxes, scores, cls_ids in zip(boxes_list, scores_list, cls_list):
            results.append(_UnifiedResults(boxes, scores, cls_ids))
        return results

    def to(self, device):
        self.device = device
        self.inner.to(device)
        return super().to(device)
