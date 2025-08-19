import warnings
from enum import Enum

import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF

warnings.simplefilter("ignore")


class AugmentKind(Enum):
    HEAVY = 1
    MEDIUM = 2
    LIGHT = 3
    VALID = 4
    TEST = 5

class ToTensor:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) for v in self.classes]
        msk = np.stack(msks, axis=-1).astype(np.float32)
        background = 1 - msk.sum(axis=-1, keepdims=True)
        sample["mask"] = TF.to_tensor(np.concatenate((background, msk), axis=-1))

        for key in [k for k in sample.keys() if k != "mask"]:
            sample[key] = np.clip(sample[key], 0,10000)
            sample[key] = TF.to_tensor(sample[key].astype(np.float32) / 10_000)
        return sample


def valid_augm(sample, size=512):
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def test_augm(sample):
    augms = [A.HorizontalFlip(p=0.1)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def train_augm_heavy(sample, size=512):
    augms = [
        A.Resize(height=size, width=size, p=1.0),
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.7
        ),
        A.RandomCrop(size, size, p=1.0),
        A.HorizontalFlip(p=0.75),
        A.VerticalFlip(p=0.75),
        A.Downscale((0.5, 0.75), p=0.05),
        A.MaskDropout(max_objects=3, fill=0, fill_mask=0, p=0.1),
        # color transforms
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1),
                A.ChannelShuffle(p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1
                ),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
            ],
            p=0.8,
        ),
        # distortion
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.Perspective(p=1),
            ],
            p=0.2,
        ),
        # noise transforms
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.Sharpen(p=1),
                A.GaussianBlur(p=1),
            ],
            p=0.2,
        ),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def train_augm_light(sample, size=512):
    augms = [
        A.Resize(height=size, width=size, p=1.0),
        A.PadIfNeeded(size, size, border_mode=0, p=1.0),
        A.RandomCrop(size, size, p=1.0),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def train_augm_medium(sample, size=512):
    augms = [
        A.Resize(height=size, width=size, p=1.0),
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.7
        ),
        A.RandomCrop(size, size, p=1.0),
        A.HorizontalFlip(p=0.75),
        A.VerticalFlip(p=0.75),
        A.Downscale((0.5, 0.75), p=0.05),
        A.MaskDropout(max_objects=3, fill=0, fill_mask=0, p=0.1),

        # distortion
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.Perspective(p=1),
            ],
            p=0.2,
        ),
        # noise transforms
        A.OneOf(
            [
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.Sharpen(p=1),
                A.GaussianBlur(p=1),
            ],
            p=0.2,
        ),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])
