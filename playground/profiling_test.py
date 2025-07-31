import torch
import torchvision.models as models

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()
model(inputs)