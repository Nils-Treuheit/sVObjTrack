import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import OBB26

CUBE_OFFSET = 95


CUBE_FACE_NAMES = [
    f"{d}_{c}" for d in
    [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for c in ("B", "W")
]


class _MoEOBBItem:
    def __init__(self, xywhr, conf, cls_id, track_id=None):
        self._xywhr = torch.as_tensor([xywhr], dtype=torch.float32)
        self._conf = torch.as_tensor([conf], dtype=torch.float32)
        self._cls = torch.as_tensor([cls_id], dtype=torch.float32)
        self._id = torch.as_tensor([track_id], dtype=torch.float32) if track_id is not None else None

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


class _MoEResults:
    boxes = None

    def __init__(self, boxes_xywhr, scores, cls_ids):
        self._obbs = []
        for xywhr, score, cid in zip(boxes_xywhr, scores, cls_ids):
            self._obbs.append(_MoEOBBItem(xywhr, score, cid))

    @property
    def obb(self):
        return self._obbs if self._obbs else None


class LossModel:
    def __init__(self, head, args):
        self.model = [head]
        self.args = args

    def parameters(self):
        return self.model[0].parameters()


class MoEYOLO(nn.Module):
    def __init__(self, base_model, nc_normal, nc_cube, args=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_module = base_model.model  # OBBModel (BaseModel)
        self.model = base_module.model  # Sequential
        self.save = base_module.save
        self.stride = base_module.stride

        # Build names dict: 0-80 from original, 81-94 unknown, 95-160 cube faces
        base_names = getattr(base_model, "names", {})
        self.names = {}
        for i in range(CUBE_OFFSET + 66):
            if i < len(base_names):
                self.names[i] = base_names[i]
            elif i < CUBE_OFFSET:
                self.names[i] = f"unknown_{i}"
            else:
                idx = i - CUBE_OFFSET
                self.names[i] = CUBE_FACE_NAMES[idx] if idx < len(CUBE_FACE_NAMES) else f"cube_{idx}"

        orig_head = self.model[-1]
        self.head_input_idx = orig_head.f

        ch = [cv[0].conv.in_channels for cv in orig_head.cv2]

        self.gate_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            ) for c in ch
        ])

        self.normal_head = OBB26(
            nc=nc_normal, ne=1, reg_max=orig_head.reg_max,
            end2end=True, ch=ch,
        )
        self.normal_head.stride = self.stride.clone() if hasattr(self.stride, 'clone') else self.stride

        self.cube_head = OBB26(
            nc=nc_cube, ne=1, reg_max=orig_head.reg_max,
            end2end=True, ch=ch,
        )
        self.cube_head.stride = self.stride.clone() if hasattr(self.stride, 'clone') else self.stride

        self._init_heads(orig_head)

        self.args = args
        self.to(self.device)

    @staticmethod
    def _load_compatible(src_module, dst_module):
        src_sd = src_module.state_dict()
        dst_sd = dst_module.state_dict()
        compatible = {}
        for key in src_sd:
            if key in dst_sd and src_sd[key].shape == dst_sd[key].shape:
                compatible[key] = src_sd[key]
        if compatible:
            dst_module.load_state_dict(compatible, strict=False)

    def _init_heads(self, orig_head):
        for i in range(len(orig_head.cv2)):
            self._load_compatible(orig_head.cv2[i], self.normal_head.cv2[i])
            self._load_compatible(orig_head.cv2[i], self.cube_head.cv2[i])
            self._load_compatible(orig_head.cv4[i], self.normal_head.cv4[i])
            self._load_compatible(orig_head.cv4[i], self.cube_head.cv4[i])
            self._load_compatible(orig_head.cv3[i], self.normal_head.cv3[i])
            self._load_compatible(orig_head.cv3[i], self.cube_head.cv3[i])
            if hasattr(orig_head, "one2one_cv2"):
                self._load_compatible(orig_head.one2one_cv2[i], self.normal_head.one2one_cv2[i])
                self._load_compatible(orig_head.one2one_cv2[i], self.cube_head.one2one_cv2[i])
                self._load_compatible(orig_head.one2one_cv4[i], self.normal_head.one2one_cv4[i])
                self._load_compatible(orig_head.one2one_cv4[i], self.cube_head.one2one_cv4[i])
                self._load_compatible(orig_head.one2one_cv3[i], self.normal_head.one2one_cv3[i])
                self._load_compatible(orig_head.one2one_cv3[i], self.cube_head.one2one_cv3[i])

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
        gate_logits = [g(f).view(x.shape[0], 1) for g, f in zip(self.gate_convs, feats)]
        gate_logit = torch.stack(gate_logits, dim=1).mean(dim=1)

        normal_pred = self.normal_head(feats)
        cube_pred = self.cube_head(feats)

        return {
            "normal": normal_pred,
            "cube": cube_pred,
            "gate_logit": gate_logit,
        }

    @torch.no_grad()
    def track(self, imgs, persist=False, tracker=None, conf=0.25, verbose=False, **kwargs):
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[None]
            imgs = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).to(self.device).float() / 255.0
        elif isinstance(imgs, torch.Tensor):
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(self.device).float() / 255.0
        else:
            raise TypeError(f"Unsupported input type: {type(imgs)}")

        out = self.forward(imgs)
        gate_logit = out["gate_logit"]

        results = []
        for b in range(imgs.shape[0]):
            use_cube = gate_logit[b].item() > 0
            y, _ = out["cube"] if use_cube else out["normal"]
            dets = y[b]
            mask = dets[:, 4] > conf
            dets = dets[mask]

            if not len(dets):
                results.append(_MoEResults(
                    np.empty((0, 5)), np.empty((0,)), np.empty((0,), dtype=int)))
                continue

            boxes = dets[:, :5].cpu().numpy()
            scores = dets[:, 4].cpu().numpy()
            cls_ids = dets[:, 5].cpu().numpy().astype(int)
            if use_cube:
                cls_ids += CUBE_OFFSET

            results.append(_MoEResults(boxes, scores, cls_ids))

        return results

    def get_loss_model(self, head_key, args=None):
        head = self.normal_head if head_key == "normal" else self.cube_head
        return LossModel(head, args or self.args)

    def bias_init(self):
        for head in (self.normal_head, self.cube_head):
            head.bias_init()
