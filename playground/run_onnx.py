from PIL import Image
import torch
import onnx
import onnxruntime
from get_onnx import preprocess_image
from visualize import visualize_detections, visualize_boxes
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


# import numpy as np

# output_name = "fasterrcnn-1_fixed.onnx"
output_name = "fasterrcnn-2_fixed.onnx"
# output_name = "fasterrcnn-12-official.onnx"
labels_path = "../labels_coco_80.txt"
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

input_image_path = "demo.jpg"
onnx_model = onnx.load(output_name)
out_image_path = "onnx_out_3_" + input_image_path

example_inputs = transforms(Image.open(input_image_path)).unsqueeze(0).numpy()
# example_inputs = transforms(Image.open(input_image_path)).numpy() # official
print("Input Shape:", example_inputs.shape)

onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(
    output_name, providers=["CUDAExecutionProvider"]
)

# onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
output = ort_session.run(None, dict(input=example_inputs))
# print(output)
# exit(0)
# output = ort_session.run(None, dict(image=example_inputs)) # official
# print(onnxruntime_outputs)
output_dict = dict(boxes=output[0], labels=output[1], scores=output[2])
visualize_boxes(input_image_path, out_image_path, output_dict, labels_path)

