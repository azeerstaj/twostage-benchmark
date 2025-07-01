from PIL import Image
import torch
import onnx
import onnxruntime
from get_onnx import preprocess_image
# import numpy as np

output_name = "fasterrcnn-1.onnx"
input_image_path = "demo.jpg"
onnx_model = onnx.load(output_name)
example_inputs= torch.randn(1, 3, 800, 800).cpu().numpy()
example_inputs = torch.tensor(preprocess_image(Image.open(input_image_path))).unsqueeze(0)

onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(
    output_name, providers=["CUDAExecutionProvider"]
)

# onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, dict(input=example_inputs))
print(onnxruntime_outputs)