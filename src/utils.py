from typing import Tuple
import numpy as np
import cv2

from dataclasses import dataclass


@dataclass
class Instance:
    bbox: any
    mask: any
    score: float


@dataclass
class Scaler:
    w: int
    h: int
    pad_t: int
    pad_b: int
    pad_l: int
    pad_r: int
    orig_w: int
    orig_h: int


def get_scaler(img: np.ndarray, max_size: int, padding_mode: str = "center"):
    im_h, im_w = img.shape[:2]
    ori_h, ori_w = img.shape[:2]
    resize_ratio = max_size / max(im_h, im_w)
    if resize_ratio < 1:
        if im_h > im_w:
            im_h = max_size
            im_w = max(1, int(round(im_w * resize_ratio)))
        else:
            im_w = max_size
            im_h = max(1, int(round(im_h * resize_ratio)))

    pad_t: int = 0
    pad_b: int = 0
    pad_l: int = 0
    pad_r: int = 0

    if padding_mode == "center":
        pad_t = pad_b = (max_size - im_h) // 2
        pad_l = pad_r = (max_size - im_w) // 2
    else:
        pad_b = max_size - im_h
        pad_r = max_size - im_w

    return Scaler(
        w=im_w,
        h=im_h,
        pad_t=pad_t,
        pad_b=pad_b,
        pad_l=pad_l,
        pad_r=pad_r,
        orig_w=ori_w,
        orig_h=ori_h,
    )


def scale_down(img: np.ndarray, scaler: Scaler, pad_value: int = 0):
    img = cv2.resize(img, (scaler.w, scaler.h))
    img = cv2.copyMakeBorder(
        img,
        scaler.pad_t,
        scaler.pad_b,
        scaler.pad_l,
        scaler.pad_r,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    return img


# shape: B, 4
def scale_up_bbox(batch: np.ndarray, scaler: Scaler):
    batch[:, 0] -= scaler.pad_l
    batch[:, 1] -= scaler.pad_t
    batch[:, 2] -= scaler.pad_l
    batch[:, 3] -= scaler.pad_t

    batch[:, 0] *= scaler.orig_w / scaler.w
    batch[:, 1] *= scaler.orig_h / scaler.h
    batch[:, 2] *= scaler.orig_w / scaler.w
    batch[:, 3] *= scaler.orig_h / scaler.h
    return batch


def scale_up(img: np.ndarray, scaler: Scaler):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    if scaler.pad_b == 0:
        scaler.pad_b = -scaler.h
    if scaler.pad_r == 0:
        scaler.pad_r = -scaler.w
    img = img[:, scaler.pad_t : -scaler.pad_b, scaler.pad_l : -scaler.pad_r]
    img = np.transpose(img, axes=(1, 2, 0))
    img = cv2.resize(
        img,
        (scaler.orig_w, scaler.orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    else:
        img = np.transpose(img, axes=(2, 0, 1))
    return img
