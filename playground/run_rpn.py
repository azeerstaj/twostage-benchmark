import torch
from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.transform.image_list import ImageList
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork  # Replace with correct import path

torch.manual_seed(0)

# ---- Simulated inputs ----
image_tensor = torch.randn(1, 3, 800, 800)  # batch size 1
image_sizes = [(800, 800)]
images = ImageList(tensors=image_tensor, image_sizes=image_sizes)

# Simulated feature maps for one level
feature1 = torch.randn(1, 256, 50, 50)  # FPN level P3 for example
feature2 = torch.randn(1, 256, 25, 25)  # FPN level P3 for example
feature3 = torch.randn(1, 256, 13, 13)  # FPN level P3 for example
features = {"0": feature1}  # as dict[str, Tensor]

# ---- Anchor Generator ----
anchor_sizes = ((32,),)
aspect_ratios = ((0.5, 1.0, 2.0),)
anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

# ---- RPN Head ----
in_channels = 256
num_anchors = anchor_generator.num_anchors_per_location()[0]  # e.g., 3
head = RPNHead(in_channels, num_anchors)

# ---- RPN Module ----
rpn = RegionProposalNetwork(
    anchor_generator=anchor_generator,
    head=head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_top_n={"training": 2000, "testing": 1000},
    post_nms_top_n={"training": 2000, "testing": 1000},
    nms_thresh=0.7,
)

# Set to eval mode
rpn.eval()

# ---- Forward pass ----
with torch.no_grad():
    head_out = head([feature1, feature2, feature3])
    # proposals, losses = rpn(images, features)

print("Len Head Out:", len(head_out))
for i, out in enumerate(head_out):
    # print(f"{i}-) [0].shape:{out[0].shape}")
    print(f"{i}-) len:{len(out)}")

# print(head_out[0][0].shape)
# print(head_out[1][0].shape)

# print(head_out[0][0][1])
# print(head_out[1][0])

# print(head_out[1])

# ---- Output ----
# print("Number of proposals:", len(proposals))
# print("Proposals shape (per image):", proposals[0].shape)
# print("Losses:", losses)  # should be empty in eval mode

