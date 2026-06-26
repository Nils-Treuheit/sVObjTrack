import torch
import torch.nn as nn
from ultralytics.nn.modules import OBB26


class LossModel:
    def __init__(self, head, args):
        self.model = [head]
        self.args = args

    def parameters(self):
        return self.model[0].parameters()


class MoEYOLO(nn.Module):
    def __init__(self, base_model, nc_normal, nc_cube, args=None):
        super().__init__()
        self.model = base_model.model
        self.save = base_model.save
        self.stride = base_model.stride

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

    @staticmethod
    def _load_compatible(src_module, dst_module):
        """Load state dict from src to dst, skipping mismatched keys."""
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

    def get_loss_model(self, head_key, args=None):
        head = self.normal_head if head_key == "normal" else self.cube_head
        return LossModel(head, args or self.args)

    def bias_init(self):
        for head in (self.normal_head, self.cube_head):
            head.bias_init()
