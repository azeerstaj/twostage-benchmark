from PIL import Image
import onnx
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
from visualize import visualize_detections, visualize_boxes

torch.manual_seed(0)

# example_inputs= torch.randn([1, 3, 800, 800])
def preprocess_image(image):
    """
    Preprocesses an image for inference. See also
    https://github.com/onnx/models/tree/refs/heads/main/validated/vision/object_detection_segmentation/faster-rcnn#preprocessing-steps

    :param image: The image to preprocess.
    :return: The preprocessed image as a numpy array.
    """
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # RGB -> BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype("float32")

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    padded_h = int(np.ceil(image.shape[1] / 32) * 32)
    padded_w = int(np.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, : image.shape[1], : image.shape[2]] = image
    image = padded_image

    return image

# SOLUTION 2: Export with fixed input dimensions for TensorRT
def main():
    input_image_path = "demo.jpg"
    labels_path = "../labels_coco_80.txt"
    out_image_path = "pt_out_" + input_image_path
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False).eval()
    
    # Fixed input size for TensorRT compatibility
    fixed_input_size = (800, 800)  # or (640, 640) for faster inference
    
    # Create dummy input with fixed dimensions
    dummy_input = torch.randn(1, 3, fixed_input_size[0], fixed_input_size[1])
    
    # For actual inference, resize your image to match
    from torchvision.transforms import Resize
    resize_transform = Resize(fixed_input_size)
    actual_input = resize_transform(transforms(Image.open(input_image_path))).unsqueeze(0)
    
    print("Fixed Input Shape:", actual_input.shape)
    
    output_name = "fasterrcnn_fixed.onnx"
    model.eval()
    
    # Run inference
    with torch.no_grad():
        output = model(actual_input)
    
    # Visualize results
    visualize_boxes(input_image_path, out_image_path, output, labels_path)
    
    # Export with fixed dimensions (TensorRT friendly)
    torch.onnx.export(
        model,
        dummy_input,  # Use dummy input for consistent shape
        output_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'labels'],
        # No dynamic_axes for TensorRT compatibility
        verbose=True
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_name)
    onnx.checker.check_model(onnx_model)
    print("TensorRT-compatible ONNX export successful!")

def export_torchscript():
    """Alternative: Export to TorchScript for better TensorRT compatibility"""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False).eval()

    # Create example input
    example_input = torch.randn(1, 3, 800, 800)

    # Export to TorchScript
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("fasterrcnn_traced.pt")

    print("TorchScript export successful!")
    return traced_model

# SOLUTION 4: Post-process ONNX for TensorRT
def fix_onnx_for_tensorrt(onnx_path, output_path):
    """Fix ONNX model for TensorRT compatibility"""
    import onnx
    from onnx import shape_inference

    # Load model
    model = onnx.load(onnx_path)

    # Infer shapes
    model = shape_inference.infer_shapes(model)

    # Fix dynamic dimensions by setting specific values
    for input_tensor in model.graph.input:
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_param:  # Dynamic dimension
                if 'batch' in dim.dim_param.lower():
                    dim.dim_value = 1
                elif 'height' in dim.dim_param.lower():
                    dim.dim_value = 800
                elif 'width' in dim.dim_param.lower():
                    dim.dim_value = 800

    # Save fixed model
    onnx.save(model, output_path)
    print(f"Fixed ONNX model saved to {output_path}")

# Usage example
if __name__ == "__main__":
    # Try the fixed dimension approach first
    # main()
    
    # If you still have issues, try TorchScript
    # export_torchscript()
    
    # Or post-process the ONNX file
    fix_onnx_for_tensorrt("fasterrcnn-2.onnx", "fasterrcnn_fixed.onnx")
