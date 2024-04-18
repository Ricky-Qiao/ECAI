import torch
import numpy as np
from utils import augmentations
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def build_transform(cfg, type):
    if type == "train":
        return _build_transform_train(cfg, cfg.INPUT.TRANSFORMS)
    elif type == "tpt":
        return _build_transform_tpt(cfg, cfg.INPUT.TRANSFORMS)
    else:
        return _build_transform_test(cfg, cfg.INPUT.TRANSFORMS)


def _build_transform_train(cfg, transform_choices):
    transform_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    if "random_resized_crop" in transform_choices:
        transform_train += [
            RandomResizedCrop(
                size=cfg.INPUT.SIZE,
                scale=cfg.INPUT.RRCROP_SCALE,
                interpolation=interp_mode,
            )
        ]

    if "random_flip" in transform_choices:
        transform_train += [RandomHorizontalFlip()]

    transform_train += [ToTensor()]

    if "normalize" in transform_choices:
        transform_train += [
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]

    transform_train = Compose(transform_train)
    print("Transform for Train: {}".format(transform_train))

    return transform_train


def _build_transform_test(cfg, transform_choices):
    transform_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_test += [Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode)]

    transform_test += [CenterCrop(cfg.INPUT.SIZE)]

    transform_test += [ToTensor()]

    if "normalize" in transform_choices:
        transform_test += [
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]

    transform_test = Compose(transform_test)
    print("Transform for Test: {}".format(transform_test))

    return transform_test


def get_preaugment(cfg):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    return Compose([
        RandomResizedCrop(
                size=cfg.INPUT.SIZE,
                scale=cfg.INPUT.RRCROP_SCALE,
                interpolation=interp_mode,
            ),
        RandomHorizontalFlip(),
        ])


def augmix(cfg, image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment(cfg)
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, cfg, base_transform, preprocess, n_views=2, augmix=True, 
                    severity=1):
        self.cfg = cfg
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, image):
        image = self.preprocess(self.base_transform(image))
        views = [augmix(self.cfg, image, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views

def _build_transform_tpt(cfg):
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    base_transform = Compose([
        Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode ),
        CenterCrop(cfg.INPUT.SIZE)])
    
    preprocess = Compose([
        ToTensor(),
        Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

    transform_tpt = AugMixAugmenter(cfg, base_transform, preprocess, n_views=cfg.DATALOADER.TEST.BATCH_SIZE-1)

    print("Transform for TPT: {}".format(transform_tpt))

    return transform_tpt

