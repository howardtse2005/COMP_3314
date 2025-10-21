import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, outputs, targets):
        """
        Forward pass for the loss calculation.
        Implement this method in subclasses to define specific loss functions.
        """
        pass

class PixelWiseCrossEntropyLoss(Loss):
    """
    Pixel-wise cross entropy loss:
    - If outputs are 1-channel (binary), use BCEWithLogitsLoss.
    - If outputs have C>=2 channels, use CrossEntropyLoss with class weights.
    """
    def __init__(self, weight=None, ignore_index: int = -100, reduction: str = 'mean', pos_weight=None):
        super().__init__(name="PixelWiseCrossEntropy")
        # class weights for CE (multi-class)
        if weight is not None:
            if isinstance(weight, (list, tuple)):
                self.ce_weight = torch.tensor(weight, dtype=torch.float32)
            elif isinstance(weight, torch.Tensor):
                self.ce_weight = weight.float()
            else:
                self.ce_weight = None
        else:
            self.ce_weight = None

        self.ignore_index = ignore_index
        self.reduction = reduction
        # positive class weight for BCE (binary)
        self.pos_weight = pos_weight if isinstance(pos_weight, torch.Tensor) or pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32)

        self._bce = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=self.pos_weight)
        # CE weight is set at forward time (moved to device)
        self._ce = None  # created lazily as needed

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        if outputs.dim() != 4:
            raise ValueError(f"Outputs must be 4D (B, C, H, W), got {outputs.dim()}D")

        b, c, h, w = outputs.shape

        # Binary case: (B, 1, H, W)
        if c == 1:
            # targets: (B, H, W) or (B, 1, H, W) with {0,1}
            if targets.dim() == 3:
                targets_b = targets.unsqueeze(1)
            elif targets.dim() == 4 and targets.shape[1] == 1:
                targets_b = targets
            else:
                raise ValueError(f"Binary case requires targets as (B,H,W) or (B,1,H,W), got {targets.shape}")

            if targets_b.max() > 1 or targets_b.min() < 0:
                raise ValueError("Targets must be binary (0 or 1).")
            targets_b = targets_b.float()

            # BCE with logits per pixel
            return self._bce(outputs, targets_b)

        # Multi-class case: (B, C>=2, H, W)
        # targets expected as class indices (B, H, W) in [0..C-1]
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets_mc = targets.squeeze(1)
        elif targets.dim() == 3:
            targets_mc = targets
        else:
            raise ValueError(f"Multi-class case requires targets as (B,H,W) or (B,1,H,W), got {targets.shape}")

        if targets_mc.dtype != torch.long:
            if (targets_mc < 0).any() or (targets_mc > c - 1).any():
                raise ValueError("Targets out of valid class index range.")
            targets_mc = targets_mc.long()

        # Build CE criterion lazily on correct device with weight
        if self._ce is None or (self._ce.weight is not None and self._ce.weight.device != outputs.device):
            ce_weight = self.ce_weight.to(outputs.device) if self.ce_weight is not None else None
            self._ce = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=self.ignore_index, reduction=self.reduction)

        return self._ce(outputs, targets_mc)