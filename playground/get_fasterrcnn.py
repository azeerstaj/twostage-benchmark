import torch
import torch.nn as nn
import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def get_model_instance_segmentation():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False).eval()
    return model.eval()

if __name__ == "__main__":
    net = get_model_instance_segmentation()
    torch.save(net.state_dict(), "fasterrcnn-coco-v1.pth")
    print(net)
