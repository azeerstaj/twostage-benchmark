import torch
import torch.nn as nn
import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision.transforms import functional as F

def get_model_instance_segmentation():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model, backbone = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
    # print("Model Device:", model.device)
    return model.eval(), backbone.eval()
    # return model, backbone

if __name__ == "__main__":
    # Load an image
    image_path = "outimgs/demo.jpg"  # Update with the correct path if needed
    image = Image.open(image_path).convert("RGB")

    # Transform the image to a tensor
    x = F.to_tensor(image).unsqueeze(0)

    print("x.device:", x.device)
    net, backbone = get_model_instance_segmentation()
    y = net(x)  # Pass the image through the model
    fmaps = backbone(x)  # Forward pass through the backbone

    torch.save(x, f"tensor_image.pt")

    print("Image shape:", x.shape)
    for i, k in enumerate(fmaps.keys()):
        # print(f"Feature map {k}: {fmaps[k].view(-1)[:10]}")
        print(f"Feature map {k}: {fmaps[k].shape}")
        # torch.save(fmaps[k], f"tensor_fmap_{i}.pt")

    for k in y[0].keys():
        print(f"Output {k}: {y[0][k].view(-1)[:10]}")
    # torch.save(net.state_dict(), "fasterrcnn-coco-v1.pth")
    # print(net)
