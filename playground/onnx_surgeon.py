import onnx
import onnx_graphsurgeon as gs
import numpy as np

const_else = gs.Constant("const_else", np.array([0]).astype(np.bool_))

def modify_maskrcnn_opset12(path_to_model, output_path):
    graph = gs.import_onnx(onnx.load(path_to_model))
    """
        Step 1: Remove unnecessary UINT8 cast
            - Pattern match Cast[BOOL->UINT8] -> Cast[UINT8 -> BOOL]
            - Fixes node 2838 - casts bool to uint8 for slice / gather. Can keep all operations in bool.
    """
    for node in graph.nodes:
        if node.op == "Cast" and node.attrs["to"] == onnx.TensorProto.UINT8:
            node.attrs["to"] = onnx.TensorProto.BOOL
            node.outputs[0].dtype = np.bool_
            # Need to modify output_node output to be bool as well.
            for output_node in node.outputs[0].outputs:
                output_node.outputs[0].dtype = np.bool_
            print(f"Removed UINT8 casts in node {node.name}")
        if node.op == "If":
            node.inputs = [const_else]

    onnx.save(gs.export_onnx(graph.cleanup()), output_path)

if __name__ == "__main__":
    onnx_path = "fasterrcnn-12-official.onnx"
    modify_maskrcnn_opset12(onnx_path, onnx_path.replace('.onnx', '_fixed.onnx'))
