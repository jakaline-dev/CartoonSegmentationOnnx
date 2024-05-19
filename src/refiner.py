import onnxruntime as ort
import numpy as np
import cv2
from dataclasses import dataclass
from . import utils


@dataclass
class Instance:
    bbox: any
    mask: any
    score: float


class RefinerModel:
    def __init__(
        self,
        model_path: str = "models/refiner.onnx",
    ):
        self.session = ort.InferenceSession(model_path)
        self.input_name = "input"
        self.output_name = ["output"]

    def infer(
        self,
        img,
        masks,
        batch_size: int = 4,
        mask_threshold: float = 0.5,
    ):
        batches, scaler = self.preprocess(img, masks, batch_size)
        output = []
        for batch in batches:
            output_raw = self.session.run(self.output_name, {self.input_name: batch})
            output.extend(output_raw[0])
        output = np.array(output).squeeze(axis=1)
        return self.postprocess(output, scaler, mask_threshold)

    def preprocess(self, img: np.ndarray, masks: np.ndarray, batch_size: int):
        scaler: utils.Scaler = utils.get_scaler(
            img, max_size=720, padding_mode="top-left"
        )
        img = utils.scale_down(img, scaler, pad_value=0)
        num_instances = len(masks)
        masks = np.transpose(masks, axes=(1, 2, 0)).astype(np.uint8) * 255
        masks = utils.scale_down(masks, scaler, pad_value=0)
        np_concat = np.concatenate([img, masks], axis=-1)
        np_img = np_concat.astype(np.float32) / 255
        np_img = np.transpose(np_img, axes=(2, 0, 1))

        batches = []
        for i in range(0, num_instances, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, num_instances)

            batch_instances = []
            for j in range(batch_start, batch_end):
                instance = np.concatenate(
                    [np_img[:3, ...], np_img[3 + j : 4 + j, ...]], axis=0
                )
                batch_instances.append(instance)

            batch = np.stack(batch_instances, axis=0)
            batches.append(batch)
        #
        # mean = [103.53, 116.28, 123.675]
        # std = [57.375, 57.12, 58.395]
        # np_img = (img - mean) / std

        return batches, scaler

    def postprocess(
        self,
        output,
        scaler: utils.Scaler,
        mask_threshold: float,
    ):
        masks = output
        masks = utils.scale_up(masks, scaler=scaler)
        masks = masks >= mask_threshold
        return masks

    # def postprocess_mask
