import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

class AnchorWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3
        )

    def forward(self, images, fpn_feature_maps):
        # Assumes all images in batch have the same size
        image_list = ImageList(images, image_sizes=[(images.shape[-2], images.shape[-1])] * images.shape[0])
        anchors = self.anchor_generator(image_list, fpn_feature_maps)
        
        # Stack anchors for export — returning per-image tensors
        boxes = torch.stack(anchors)                      # [B, N, 4]
        scores = torch.rand_like(boxes[..., 0])           # Dummy scores [B, N]
        labels = torch.zeros_like(boxes[..., 0].long())   # Dummy labels [B, N]

        return boxes, scores, labels


# Create dummy inputs
batch_size = 2
input_tensor = torch.randn(batch_size, 3, 800, 800)

# Simulated FPN feature maps (as would come from a real backbone)
fpn_maps = [
    torch.randn(batch_size, 256, 100, 100),  # P3
    torch.randn(batch_size, 256, 50, 50),    # P4
    torch.randn(batch_size, 256, 25, 25),    # P5
]

# Initialize the model
model = AnchorWrapper()
model.eval()

for i, a in enumerate(anchors):
    print(f"Image {i} has {a.shape[0]} anchors. Anchor shape: {a.shape}")


# Export to ONNX
output_name = "anchor_generator.onnx"
dummy_input = (input_tensor, fpn_maps)

torch.onnx.export(
    model,
    dummy_input,  # Tuple of inputs
    output_name,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input', 'fpn_map1', 'fpn_map2', 'fpn_map3'],
    output_names=['boxes', 'scores', 'labels'],
    verbose=True
)

