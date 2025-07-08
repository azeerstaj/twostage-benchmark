import onnx
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def print_all_modules(module, prefix=''):
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name, '->', submodule)
        print_all_modules(submodule, full_name)


def print_state_dict_info(module):
    print("\nState Dict:")
    for key, tensor in module.state_dict().items():
        print(f"{key}: shape={tuple(tensor.shape)}, num_elements={tensor.numel()}")

def print_onnx_nodes(model):
    for n in model.graph.node:
        print(n.name)

if __name__ == "__main__":

    # PyTorch Example.
    arch = "backbone.body."
    new_dict = dict()
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    """
    for key, tensor in module.state_dict().items():
        if arch in key:
            new_dict[key] = tensor
    """

    print_all_modules(model)
    print_state_dict_info(model)
    """
    # model_name = "fasterrcnn-12-official.onnx"
    model_name = "fasterrcnn-1_fixed.onnx"
    model = onnx.load_model(model_name)
    print_onnx_nodes(model)
    """
