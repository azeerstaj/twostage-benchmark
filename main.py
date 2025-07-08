import os
import time
import numpy as np
import PIL.Image as Image
from infer.tensorrt_infer import TensorRTInfer
from infer.visualize import visualize_detections
import datetime

def main():

    labels_path = "labels_coco_80.txt"
    # engine_path = "weights/fasterrcnn-1_fixed.engine"
    engine_path = "playground/fasterrcnn-1_fixed.engine"
    
    test_cases_path = "test_cases/"
    # test_cases_path = None

    output_path = "output/"

    labels = []

    with open(labels_path) as f:
        for label in f:
            labels.append(label.strip())

    trt_infer = TensorRTInfer(engine_path, True, True)
    if test_cases_path:

        image_paths = os.listdir(test_cases_path)
        assert len(image_paths) != 0, f"{image_paths} dir is empty!"

        print(f"\nInferring data in {test_cases_path}")

        image_paths = [os.path.join(test_cases_path, p) for p in image_paths]
        for image_path in image_paths:
            image = Image.open(image_path)
            detections = trt_infer.process(image)
            if output_path:
                # Image Visualizations
                image_path_ = image_path.split('/')[-1].split('.')[0]
                ct = datetime.datetime.now()
                viz_output_path = os.path.join(output_path, f"{image_path_}_{ct}.png")
                visualize_detections(image_path, viz_output_path, detections, labels)

                # Text Results
                output_results = ""
                for d in detections:
                    line = [
                        d["xmin"],
                        d["ymin"],
                        d["xmax"],
                        d["ymax"],
                        d["score"],
                    ]
                    output_results += "\t".join([str(f) for f in line]) + "\n"
                with open(os.path.join(output_path, f"{image_path_}.txt"), "w") as f:
                    f.write(output_results)
    else:
        print("No input provided, running in benchmark mode")
        shape, dtype = trt_infer.input_spec()
        batch = 255 * np.random.rand(*shape).astype(dtype)
        trt_infer.context.set_input_shape("image", (batch.shape))
        iterations = 200
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            trt_infer.infer(batch)
            times.append(time.time() - start)
            print(f"Iteration {i+1} / {iterations}", end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print(f"Average Latency: {1000 * np.average(times):.3f} ms")

    print("\nFinished Processing")

if __name__ == "__main__":
    main()