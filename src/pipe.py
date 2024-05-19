import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import cv2
from dataclasses import dataclass
from .base import BaseModel
from .refiner import RefinerModel
from IPython.display import Image


@dataclass
class Instance:
    bbox: any
    mask: any
    score: float


# Create an infinite color generator
def color_generator():
    colors = [
        (255, 16, 16),
        (16, 255, 16),
        (255, 240, 16),
        (16, 15, 255),
        (0, 24, 236),
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
        (44, 153, 168),
        (0, 194, 255),
        (52, 69, 147),
        (100, 115, 255),
        (0, 24, 236),
        (132, 56, 255),
        (82, 0, 133),
        (203, 56, 255),
        (255, 149, 200),
        (255, 55, 199),
    ]
    while True:
        for color in colors:
            yield color


class Pipe:
    def __init__(
        self,
        base_model_path: str,
        refiner_mode_path: str = None,
    ):
        self.base_model = BaseModel(base_model_path)
        self.refiner_model = None
        if refiner_mode_path:
            self.refiner_model = RefinerModel(refiner_mode_path)

    def infer(
        self,
        input_path,
        instance_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        refiner_batch_size: int = 4,
    ):
        img = cv2.imread(input_path)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return self._infer(img, instance_threshold, mask_threshold, refiner_batch_size)

    def _infer(
        self,
        img,
        instance_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        refiner_batch_size: int = 4,
    ):
        bboxes, scores, masks = self.base_model.infer(
            img, instance_threshold, mask_threshold
        )
        if self.refiner_model:
            masks = self.refiner_model.infer(
                img, masks, batch_size=refiner_batch_size, mask_threshold=mask_threshold
            )
        # masks = self.postprocess_masks(masks, bboxes)
        return list(zip(bboxes, scores, masks))

    def visualize(
        self,
        input_path: str,
        instance_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        refiner_batch_size: int = 4,
        visualize_bbox: bool = True,
        visualize_mask: bool = True,
    ):
        img = cv2.imread(input_path)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        color_gen = color_generator()
        for instance in self._infer(
            img,
            instance_threshold,
            mask_threshold,
            refiner_batch_size,
        ):
            bbox, score, mask = instance
            color = next(color_gen)
            if visualize_mask:
                colored_mask = np.zeros_like(img)
                colored_mask[mask] = color
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 1)

            if visualize_bbox:
                cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                cv2.putText(
                    img,
                    f"Score: {score:.2f}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        return img

    # Custom postprocessing
    def postprocess_masks(self, masks, bboxes):
        refined_masks = []
        for mask, bbox in zip(masks, bboxes):
            # Convert the boolean mask to uint8
            mask = mask.astype(np.uint8)
            # Perform connected component analysis
            num_labels, labels = cv2.connectedComponents(mask)
            # Filter instances based on size (adjust the threshold as needed)
            min_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * 0.1
            refined_mask = np.zeros_like(mask)
            for label in range(1, num_labels):
                component_size = np.sum(labels == label)
                if min_size <= component_size:
                    refined_mask[labels == label] = 1

            # kernel = np.ones((3, 3), np.uint8)
            # refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            refined_masks.append(refined_mask > 0)

        return refined_masks
