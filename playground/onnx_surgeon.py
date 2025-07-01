import onnx
import onnx_graphsurgeon as gs
import numpy as np

onnx_path = "fasterrcnn-1.onnx"
const_else = gs.Constant("const_else", np.array([0]).astype(np.bool_))

g = gs.import_onnx(onnx.load(onnx_path))

# Overwrite condition to a constant.
for n in g.nodes:
    if n.op == "If":
        n.inputs = [const_else]
        
# Export the modified graph to a new ONNX model.
onnx.save(gs.export_onnx(g), onnx_path.replace('.onnx', '_fixed.onnx'))
onnx_path = onnx_path.replace('.onnx', '_fixed.onnx')