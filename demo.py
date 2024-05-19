from src.pipe import Pipe
import cv2

pipe = Pipe(base_model_path="models/base.onnx", refiner_mode_path="models/refiner.onnx")
input_image_path = "demo.jpg"
img = pipe.visualize(input_image_path, instance_threshold=0.3, mask_threshold=0.6)
# cv2.imshow("demo", img)
cv2.imwrite("output.png", img)
