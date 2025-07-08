import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.resnet import resnet50, ResNet50_Weights

# 1. Load pretrained ResNet50
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

# 2. Extract ResNet50 with FPN
#    trainable_backbone_layers = 3 or 5 is common
backbone_with_fpn = _resnet_fpn_extractor(resnet, trainable_layers=3)
backbone_with_fpn.eval()

# 3. Wrap for ONNX export (outputs must be Tensor)
class FPNWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # nn.Module with FPN

    def forward(self, x):
        features = self.backbone(x)  # returns dict of features: {'0': P3, '1': P4, ...}
        return tuple(features.values())  # must return tuple/list of tensors for ONNX

# 4. Instantiate model
model = FPNWrapper(backbone_with_fpn)
model.eval()

# 5. Dummy input for tracing
dummy_input = torch.randn(1, 3, 800, 800)  # [B, C, H, W]

# 6. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet50_fpn.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["P3", "P4", "P5", "P6"],
    dynamic_axes=None,  # avoid for TensorRT
    verbose=True
)

