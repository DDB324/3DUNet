import random
from typing import Tuple, List, Callable

import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize


class Resize:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, scale_factor=(1, self.scale, self.scale), mode='trilinear', align_corners=False,
                            recompute_scale_factor=True)
        mask = F.interpolate(mask, scale_factor=(1, self.scale, self.scale), mode='nearest',
                             recompute_scale_factor=True)
        return img[0], mask[0]


class RandomResize:
    def __init__(self, s_rank, w_rank, h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, img, mask):
        random_w = random.randint(self.w_rank[0], self.w_rank[1])
        random_h = random.randint(self.h_rank[0], self.h_rank[1])
        random_s = random.randint(self.s_rank[0], self.s_rank[1])
        self.shape = [random_s, random_h, random_w]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode='trilinear', align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode='nearest')
        return img[0], mask[0].long()


class RandomCrop:
    def __init__(self, crop_size: int = 48):
        if crop_size < 0:
            raise ValueError("Crop size must be positive.")
        self.crop_size: int = crop_size

    @staticmethod
    def _get_range(img_slices: int, crop_size: int) -> Tuple[int, int]:
        start = 0 if img_slices < crop_size else random.randint(0, img_slices - crop_size)
        end = min(start + crop_size, img_slices)
        return start, end

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_slices = img.size(1)
        start, end = self._get_range(img_slices, self.crop_size)

        cropped_img = img[:, start:end, :, :]
        cropped_mask = mask[:, start:end, :, :]
        return cropped_img, cropped_mask


class RandomFlip:
    def __init__(self, prob: float = 0.5, flip_lr: bool = True, flip_ud: bool = True):
        self.prob = prob
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud

    def _flip(self, img: torch.Tensor, prob: Tuple[float, float], flip_lr: bool, flip_ud: bool) -> torch.Tensor:
        if flip_lr and prob[0] <= self.prob:
            img = img.flip(3)

        if flip_ud and prob[1] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        img = self._flip(img, prob, self.flip_lr, self.flip_ud)
        mask = self._flip(mask, prob, self.flip_lr, self.flip_ud)
        return img, mask


class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    @staticmethod
    def _rotate(img, cnt):
        img = torch.rot90(img, cnt, [1, 2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0, self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)


class CenterCrop:
    def __init__(self, crop_max_size: int = 96, base: int = 16):
        if base <= 0:
            raise ValueError("Base value must be positive.")
        self.base: int = base  # base默认为16，4次下采样后为1
        # max_size为限制最大采样slices数，防止显存溢出，应为base倍数
        self.crop_max_size: int = crop_max_size - crop_max_size % self.base

    def __call__(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_slices: int = img.size(1)
        if img_slices < self.base:
            raise ValueError("Image size too small for cropping")

        crop_size = min(self.crop_max_size, img_slices - img_slices % self.base)

        left = img_slices // 2 - crop_size // 2
        right = left + crop_size

        crop_img = img[:, left:right, :, :]
        crop_label = label[:, left:right, :, :]
        return crop_img, crop_label


class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return normalize(img, self.mean, self.std, False), mask


class Compose:
    def __init__(self, transforms_list: List[Callable]):
        self.transforms = transforms_list

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
