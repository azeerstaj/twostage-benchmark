import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

# Anchor generator parameters
SIZES = [(32,), (64,), (128,)]
ASPECT_RATIOS = [(0.5, 1.0, 2.0)] * len(SIZES)

import numpy as np

def anchor_forward_numpy(image_height, image_width, feature_map_shapes, sizes, aspect_ratios):
    anchors_all = []

    for (feat_h, feat_w), scale_set, ratio_set in zip(feature_map_shapes, sizes, aspect_ratios):
        # Compute stride (integer division)
        stride_h = image_height // feat_h
        stride_w = image_width // feat_w

        # === Base anchors (zero-centered) ===
        base_anchors = []
        for scale in scale_set:
            for ratio in ratio_set:
                h = scale * np.sqrt(ratio)
                w = scale / np.sqrt(ratio)
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                base_anchors.append([x1, y1, x2, y2])
        
        # Match PyTorch behavior: round and keep float32
        base_anchors = np.round(np.array(base_anchors, dtype=np.float32))  # shape: (A, 4)

        # === Generate grid shifts ===
        shift_x = np.arange(0, feat_w, dtype=np.float32) * stride_w  # shape: (W,)
        shift_y = np.arange(0, feat_h, dtype=np.float32) * stride_h  # shape: (H,)
        print("Shift X:", shift_x)
        print("Shift Y:", shift_y)

        shift_y, shift_x = np.meshgrid(shift_y, shift_x, indexing="ij")  # shape: (H, W)

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=1)  # shape: (H*W, 4)
        print("Shifts:", shifts[:10])

        # === Combine shifts and base anchors ===
        anchors = shifts[:, None, :] + base_anchors[None, :, :]  # (H*W, A, 4)
        anchors = anchors.reshape(-1, 4)  # shape: (H*W * A, 4)
        print("Anchors:", anchors[:10])
        # exit(0)

        anchors_all.append(anchors)

    return np.vstack(anchors_all).astype(np.float32)  # Final shape: (total_anchors, 4)


def anchor_forward(image_list: ImageList, feature_maps: list[torch.Tensor]) -> torch.Tensor:
    grid_sizes = [fm.shape[-2:] for fm in feature_maps]
    image_size = image_list.tensors.shape[-2:]
    dtype = feature_maps[0].dtype
    device = feature_maps[0].device

    # Compute strides
    strides = [
        [
            torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
            torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device),
        ]
        for g in grid_sizes
    ]

    # Generate zero-centered anchors
    cell_anchors = []
    for sizes, aspect_ratios in zip(SIZES, ASPECT_RATIOS):
        scales = torch.as_tensor(sizes, dtype=dtype, device=device)
        ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(ratios)
        w_ratios = 1.0 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        cell_anchors.append(base_anchors.round())

    anchors_all = []
    for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
        gh, gw = size
        sh, sw = stride
        shifts_x = torch.arange(0, gw, dtype=torch.int32, device=device) * sw
        shifts_y = torch.arange(0, gh, dtype=torch.int32, device=device) * sh
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        anchors = (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)
        anchors_all.append(anchors)

    # Return anchors for batch[0] only (ONNX doesn’t support loops over batch)
    return torch.cat(anchors_all, dim=0)

# === ONNX Export Wrapper ===
class AnchorONNXExport(torch.nn.Module):
    def forward(self, images, f1, f2, f3):
        image_list = ImageList(images, [(images.shape[-2], images.shape[-1])] * images.shape[0])
        return anchor_forward(image_list, [f1, f2, f3])

# === MAIN ===
if __name__ == "__main__":

    image_shape = (1, 3, 800, 800)
    dummy_images = ImageList(torch.randn(image_shape), image_shape)

    dummy_f1 = torch.randn(1, 256, 100, 100)
    dummy_f2 = torch.randn(1, 256, 50, 50)
    dummy_f3 = torch.randn(1, 256, 25, 25)
    dummy_f4 = torch.randn(1, 256, 10, 10)
    feature_maps = [dummy_f4]#dummy_f2, dummy_f3]

    output = anchor_forward(dummy_images,  feature_maps)

    # These should match what you're passing to the AnchorGenerator
    # SIZES = ((32,), (64,), (128,))
    SIZES = ((32,),)

    # ASPECT_RATIOS = ((0.5, 1.0, 2.0),)
    ASPECT_RATIOS = ((0.5,),)

    anchor_generator = AnchorGenerator(
        sizes=SIZES,
        aspect_ratios=ASPECT_RATIOS
    )

    # Step 5: Forward pass through AnchorGenerator (PyTorch)
    anchors_1 = anchor_generator(dummy_images, feature_maps)[0]

    # Prepare input for numpy version
    image_height, image_width = dummy_images.tensors.shape[-2:]
    feature_map_shapes = [tuple(fm.shape[-2:]) for fm in feature_maps]  # [(100, 100), (50, 50), (25, 25)]

    # Call the numpy anchor generator
    anchors_2_np = anchor_forward_numpy(
        image_height=image_height,
        image_width=image_width,
        feature_map_shapes=feature_map_shapes,
        sizes=SIZES,
        aspect_ratios=ASPECT_RATIOS
    )

    print(anchors_2_np)
    exit(0)

    # Convert numpy result to torch.Tensor for comparison
    anchors_2 = torch.tensor(anchors_2_np, dtype=anchors_1.dtype, device=anchors_1.device)

    anchors_ref = anchor_forward(dummy_images, feature_maps)

    # Compare results
    print("anchor_generator vs anchor_forward:", torch.allclose(anchors_1, anchors_ref))
    print("numpy vs anchor_forward:", torch.allclose(anchors_2, anchors_ref))

    # Print a few anchors
    print("Ref:")
    print(anchors_ref[:5], "\n")

    print("Raw Anchors:")
    print(anchors_1[:5], "\n")

    print("Numpy Anchors:")
    print(anchors_2[:5], "\n")

