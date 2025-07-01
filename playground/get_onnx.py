from PIL import Image
import onnx
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from visualize import visualize_detections

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

input_image_path = "demo.jpg"
labels_path = "../labels_coco_80.txt"
out_image_path = "pt_out_" + input_image_path
model = fasterrcnn_resnet50_fpn(pretrained=True)
# example_inputs= torch.randn([1, 3, 800, 800])
example_inputs = torch.tensor(preprocess_image(Image.open(input_image_path))).unsqueeze(0)
output_name = "fasterrcnn-1.onnx"

print(example_inputs.shape)
# exit(0)

model.eval()
output = model(example_inputs)
print(output)
exit(0)
visualize_detections(input_image_path, out_image_path, output, labels_path)

# Export the model
torch.onnx.export(model,               # model being run
                  example_inputs,                         # model input (or a tuple for multiple inputs)
                  output_name,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],) # the model's output names)


onnx_model = onnx.load(output_name)
onnx.checker.check_model(onnx_model)

print("Output:", output)