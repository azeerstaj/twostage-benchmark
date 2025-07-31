import torch
import torch.nn.functional as F

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor, resize
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image

from PIL import Image
import matplotlib.pyplot as plt



def roi_align(
    features: torch.Tensor,
    rois: torch.Tensor,
    output_size: int,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch RoIAlign implementation.

    Args:
        features (Tensor[B, C, H, W]):
            feature map.
        rois (Tensor[K, 5]):
            each row is [batch_idx, x1, y1, x2, y2] in *input image* coords.
        output_size (int):
            height = width = output spatial size of the output.
        spatial_scale (float):
            scale box coords by this to map to feature-map coords.
        sampling_ratio (int):
            number of sampling points per bin (if <=0, adaptive: ceil(roi_bin_size)).
        aligned (bool):
            if True, use the half-pixel shift (as in torchvision) to more exactly align
            the extracted features with the input pixels.

    Returns:
        output (Tensor[K, C, output_size, output_size])
    """
    assert rois.dim() == 2 and rois.size(1) == 5
    K = rois.size(0)
    B, C, H, W = features.shape

    # Scale ROIs to feature-map coordinates
    rois_scaled = rois.clone()
    rois_scaled[:, 1:] *= spatial_scale

    if aligned:
        # align corners as in torchvision: shift by -0.5 after scaling
        rois_scaled[:, 1:] -= 0.5

    # Prepare output tensor
    output = torch.zeros((K, C, output_size, output_size), device=features.device, dtype=features.dtype)

    for i in range(K):
        batch_idx, x1, y1, x2, y2 = rois_scaled[i]
        # width/height of the ROI in the feature map
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        # bin size
        bin_w = w / output_size
        bin_h = h / output_size

        # determine number of sampling points
        if sampling_ratio > 0:
            roi_bin_grid_w = roi_bin_grid_h = sampling_ratio
        else:
            # adaptive: at least 1 point per bin
            roi_bin_grid_w = int(torch.ceil(w   / output_size).item())
            roi_bin_grid_h = int(torch.ceil(h   / output_size).item())

        # build sampling grid for this ROI
        # coords in feature space in range [-1, +1] for grid_sample
        grid_y = []
        grid_x = []
        for iy in range(output_size):
            # start of this bin in feature coords
            y_start = y1 + iy * bin_h
            for ix in range(output_size):
                x_start = x1 + ix * bin_w

                # generate grid of points within the bin
                ys = y_start + (torch.arange(roi_bin_grid_h, device=features.device) + 0.5) * (bin_h  / roi_bin_grid_h)
                xs = x_start + (torch.arange(roi_bin_grid_w, device=features.device) + 0.5) * (bin_w  / roi_bin_grid_w)
                grid_y.append(ys)
                grid_x.append(xs)

        # stack into (output_size*output_size*grid_h, grid_w)
        grid_y = torch.stack(grid_y).view(output_size, output_size, roi_bin_grid_h, 1).expand(-1, -1, -1, roi_bin_grid_w)
        grid_x = torch.stack(grid_x).view(output_size, output_size, 1, roi_bin_grid_w).expand(-1, -1, roi_bin_grid_h, -1)

        # normalize to [-1,1]
        grid_y = (grid_y / (H - 1)) * 2 - 1
        grid_x = (grid_x / (W - 1)) * 2 - 1

        # final grid is (1, out_h, out_w, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # shape: (out_h, out_w, grid_h, grid_w, 2)
        grid = grid.view(1, output_size * roi_bin_grid_h, output_size * roi_bin_grid_w, 2)

        # sample and then average pooling within each bin
        feat = features[int(batch_idx)].unsqueeze(0)  # [1, C, H, W]
        sampled = F.grid_sample(feat, grid, align_corners=True)  # [1, C, out_h*grid_h, out_w*grid_w]
        sampled = sampled.view(C,
                               output_size, roi_bin_grid_h,
                               output_size, roi_bin_grid_w)
        # average over the intra-bin samples
        output[i] = sampled.mean(dim=[2, 4])

    return output


def custom_multiscale_roi_align(features: dict, boxes: list[torch.Tensor], image_sizes: list[tuple[int, int]],
                                 output_size: int = 7, sampling_ratio: int = 2) -> torch.Tensor:
    """
    A simplified manual implementation of MultiScaleRoIAlign.

    Args:
        features: Dict of FPN features. Keys like ['0', '1', '2', '3'].
                  Each feature is a tensor of shape [B, C, H, W].
        boxes: List of boxes per image. Each item is a tensor of shape [num_boxes, 4].
        image_sizes: List of (H, W) for each image in the batch.
        output_size: Size of output features (H, W).
        sampling_ratio: Sampling ratio for roi_align.

    Returns:
        Tensor of shape [total_boxes, C, output_size, output_size]
    """
    assert len(features) > 0
    all_features = []
    all_levels = list(features.keys())
    all_boxes = boxes[0]  # batch size = 1 assumed

    # Compute scale of each box (sqrt(area))
    box_sizes = (all_boxes[:, 2:] - all_boxes[:, :2]).clamp(min=1e-6)
    box_scales = torch.sqrt(box_sizes[:, 0] * box_sizes[:, 1])

    # Map scale to level (similar to torchvision logic)
    levels = torch.floor(4 + torch.log2(box_scales / 224)).clamp(min=0, max=len(all_levels)-1).to(torch.int64)

    # Collect outputs from each level
    roi_outputs = []
    for level_idx, level_name in enumerate(all_levels):
        level_feature = features[level_name]  # [1, C, H, W]
        idx_in_level = torch.nonzero(levels == level_idx).squeeze(1)
        if idx_in_level.numel() == 0:
            continue

        boxes_in_level = all_boxes[idx_in_level]
        # Add batch index (assume batch=1)
        rois = torch.cat([torch.zeros((len(boxes_in_level), 1)), boxes_in_level], dim=1)  # [N, 5]
        output = roi_align(level_feature, rois, output_size=output_size,
                           spatial_scale=level_feature.shape[-1] / image_sizes[0][1],
                           sampling_ratio=sampling_ratio, aligned=True)
        roi_outputs.append((idx_in_level, output))

    # Stitch results back in correct order
    output = torch.zeros((len(all_boxes), features[all_levels[0]].shape[1], output_size, output_size),
                         dtype=output.dtype, device=output.device)

    for idxs, feat in roi_outputs:
        output[idxs] = feat

    return output

# Load and preprocess image
image = Image.open("../test_cases/demo.jpg").convert("RGB")
image = resize(to_tensor(image), size=[800, 800])
image = image.unsqueeze(0)

# Backbone with FPN
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
backbone.eval()
with torch.no_grad():
    features = backbone(image[0])  # features['0'] .. ['3']

# Dummy boxes
boxes = [torch.tensor([[50, 60, 200, 220], [300, 300, 450, 450], [100, 100, 300, 350]], dtype=torch.float)]

print("BoxesShape:", boxes[0].shape)

# Custom RoIAlign
rois = custom_multiscale_roi_align(features, boxes, image_sizes=[(800, 800)], output_size=7)
print("Custom RoIAlign Output Shape:", rois.shape)
print("rois[0][10][0][0]:", rois[0][10][0][0])

"""
print("RoIAlign output shape:", rois.shape)  # [num_boxes, channels, H, W]
drawn = draw_bounding_boxes((image[0] * 255).byte(), boxes[0], colors="red", width=2)
plt.imshow(to_pil_image(drawn))
plt.title("RoIs on Input Image")
plt.axis("off")
plt.show()
"""
