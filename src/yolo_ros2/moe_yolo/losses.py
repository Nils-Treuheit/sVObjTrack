import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional per-class alpha weighting.

    Drop-in replacement for ``nn.BCEWithLogitsLoss(reduction="none")``.

    Args:
        gamma: focusing parameter (>=0). 0 = plain BCE.
        alpha: per-class weight, shape ``(nc,)`` or scalar. ``None`` disables.
        reduction: ``'none'`` | ``'sum'`` | ``'mean'``.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (*, nc) — logits and soft labels (e.g. from TaskAlignedAssigner)."""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if self.gamma == 0.0:
            focal_weight = 1.0
        else:
            p = pred.sigmoid()
            pt = p * target + (1 - p) * (1 - target)
            focal_weight = (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        loss = focal_weight * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


class BalancedFocalOBBLoss:
    """``v8OBBLoss`` variant with focal classification loss + per-image instance balancing.

    Per-image balancing normalises the classification loss by the number of positive
    anchors in *each* image separately, so that a frame with 1 object contributes as
    much as a frame with 50 objects.

    Args:
        model: (mock) model with ``.model[-1]`` (OBB head) and ``.args`` (hyp).
        gamma: focal-loss gamma.
        alpha: per-class alpha weights, shape ``(nc,)`` or scalar.
        balance_per_image: if ``True``, normalise classification loss per-image
            instead of globally.  Default ``True``.
    """

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        *,
        balance_per_image: bool = True,
        **kwargs,
    ):
        from ultralytics.utils.loss import v8OBBLoss

        self._inner = v8OBBLoss(model, **kwargs)
        self._inner.bce = FocalLoss(gamma=gamma, alpha=alpha, reduction="none")
        self._balance = balance_per_image

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def bce(self):
        return self._inner.bce

    @bce.setter
    def bce(self, value):
        self._inner.bce = value

    # ------------------------------------------------------------------
    #  Override ``loss()`` — identical to ``v8OBBLoss.loss`` except for
    #  the classification-loss normalisation at ``# <<< CHANGE``.
    # ------------------------------------------------------------------
    def loss(self, preds, batch):
        from ultralytics.utils.loss import make_anchors

        nd = self._inner.nc  # number of classes
        device = self._inner.device

        loss = torch.zeros(4, device=device)  # box, cls, dfl, angle

        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self._inner.stride, 0.5)
        batch_size = pred_angle.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype) * self._inner.stride[0]

        # ---- targets ----
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
        targets = targets[(rw >= 2) & (rh >= 2)]
        targets = self._inner.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # ---- assigner ----
        pred_bboxes = self._inner.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self._inner.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ---- cls loss with *per-image* balancing ---------------------------------
        bce_loss = self._inner.bce(pred_scores, target_scores.to(dtype))
        if self._inner.class_weights is not None:
            bce_loss *= self._inner.class_weights

        if self._balance and batch_size > 1:
            # Per-image normalisation: each image contributes equally
            bce_per_img = bce_loss.reshape(batch_size, -1, nd).sum(dim=(1, 2))  # (bs,)
            img_sum = target_scores.reshape(batch_size, -1, nd).sum(dim=(1, 2)).clamp(min=1)  # (bs,)
            loss[1] = (bce_per_img / img_sum).mean()
        else:
            # Global normalisation (original v8OBBLoss behaviour)
            loss[1] = bce_loss.sum() / target_scores_sum
        # --------------------------------------------------------------------------

        # ---- bbox / dfl loss ----
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self._inner.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self._inner.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self._inner.hyp.box
        loss[1] *= self._inner.hyp.cls
        loss[2] *= self._inner.hyp.dfl
        loss[3] *= self._inner.hyp.angle

        return loss * batch_size, loss.detach()

    def __call__(self, preds, batch):
        return self.loss(preds, batch)
