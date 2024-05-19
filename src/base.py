import onnxruntime as ort
import numpy as np
from . import utils


class BaseModel:
    def __init__(
        self,
        model_path: str = "models/base.onnx",
    ):
        self.session = ort.InferenceSession(model_path)
        self.input_name = "input"
        self.output_name = ["dets", "masks"]

    # Single image only
    def infer(
        self,
        img,
        instance_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ):
        np_img, scaler = self.preprocess(img)
        output_raw = self.session.run(self.output_name, {self.input_name: np_img})
        return self.postprocess(
            output_raw,
            scaler,
            instance_threshold=instance_threshold,
            mask_threshold=mask_threshold,
        )

    def preprocess(self, img: np.ndarray):
        scaler: utils.Scaler = utils.get_scaler(
            img, max_size=640, padding_mode="center"
        )
        img = utils.scale_down(img, scaler, pad_value=114)
        mean = [103.53, 116.28, 123.675]
        std = [57.375, 57.12, 58.395]
        np_img = (img - mean) / std
        np_img = np_img.astype(np.float32)
        np_img = np.transpose(np_img, axes=(2, 0, 1))
        np_img = np.expand_dims(np_img, axis=0)
        return np_img, scaler

    def postprocess(
        self,
        output,
        scaler: utils.Scaler,
        instance_threshold: float,
        mask_threshold: float,
    ):
        dets, masks = output
        bboxes = dets[:, :, :4]
        scores = dets[:, :, 4]
        # Create a boolean mask based on the instance_threshold
        instance_mask = scores >= instance_threshold
        scores = scores[instance_mask]
        bboxes = bboxes[instance_mask]
        masks = masks[instance_mask]
        bboxes = utils.scale_up_bbox(bboxes, scaler)
        masks = utils.scale_up(masks, scaler=scaler)
        bboxes = bboxes.astype(np.int32)
        masks = masks >= mask_threshold
        return bboxes, scores, masks
