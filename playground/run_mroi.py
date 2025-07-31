import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms.functional import to_tensor, resize
from torchvision.ops import MultiScaleRoIAlign
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load and prepare the image
image = Image.open("../test_cases/demo.jpg").convert("RGB")
image = resize(to_tensor(image), size=[800, 800])  # Resize and normalize
image = image.unsqueeze(0)  # Add batch dimension

# 2. Load backbone with FPN (used in Faster R-CNN)
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
backbone.eval()

# 3. Extract feature maps from different pyramid levels
with torch.no_grad():
    features = backbone(image[0])  # List of {feat1, feat2, feat3, feat4}

# Check what keys are in FPN
print("FPN feature keys:", features.keys())
# Typically: ['0', '1', '2', '3'] corresponding to P2, P3, P4, P5

# 4. Create dummy RoI boxes (simulate object proposals)
# Format: [x1, y1, x2, y2]
boxes = torch.tensor([
    [50, 60, 200, 220],
    [300, 300, 450, 450],
    [100, 100, 300, 350]
], dtype=torch.float)

# 5. Define MultiScaleRoIAlign
roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

# 6. Perform RoIAlign
image_sizes = [(800, 800)]  # List of (height, width) per image in batch
rois = roi_align(features, [boxes], image_sizes)

print("RoIAlign output shape:", rois.shape)  # [num_boxes, channels, H, W]
print(rois.view(-1)[:5])
# drawn = draw_bounding_boxes((image[0] * 255).byte(), boxes, colors="red", width=2)
# plt.imshow(to_pil_image(drawn))
# plt.title("RoIs on Input Image")
# plt.axis("off")
# plt.show()

