import hfai
import os
from torchvision import transforms
from .statistic import *
from PIL import Image

# 自定义 Dataset 所需的头文件
# -------------------
from typing import Callable, Optional

from .base import (
    BaseDataset,
    get_data_dir
)
# -------------------

def create_transform(split):
    if split == "train":
        crop = transforms.RandomCrop(256)
    elif split == "valid":
        crop = transforms.CenterCrop(256)
    else:
        raise ValueError(f"Unknown split: {split}")

    transform = transforms.Compose([
        transforms.Resize(256),
        # crop,
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

    return transform


class ImageNet(hfai.datasets.ImageNet):

    def __init__(self, split) -> None:
        super().__init__(split)
        self.img_transform = create_transform(split)

    def __getitem__(self, indices):
        samples = super().__getitem__(indices)

        new_samples = []
        for img, label in samples:
            img = self.img_transform(img.convert("RGB"))
            new_samples.append(img)

        return new_samples


class IconNet(BaseDataset):

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super(IconNet, self).__init__()

        assert split in ["train", "valid"]
        self.transform = transform
        # 当前目录
        data_dir = get_data_dir()
        # 目录拼接
        self.path = data_dir / "IconNet" / f"processed_{split}"
        self.files = sorted([os.path.join(self.path, x) for x in os.listdir(self.path) if x.endswith(".png")])

        # self.meta = pd.read_csv(self.path / "IconNet" / f"{split}.csv", usecols=[0, 1])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, indices):
        fnames = [self.files[idx] for idx in indices]
        transformed_samples = []
        for fname in fnames:
            im = Image.open(fname)
            im = self.transform(im)
            transformed_samples.append(im)
        return transformed_samples


def coco(split):
    img_transform = create_transform(split)
    def transform(img, img_id, anno):
        img = img_transform(img)
        return img

    dataset = hfai.datasets.CocoCaption(split, transform=transform)
    return dataset


def googlecc(split):
    img_transform = create_transform(split)
    def transform(img, text):
        return img_transform(img.convert("RGB"))

    dataset = hfai.datasets.GoogleConceptualCaption(split, transform)
    return dataset


def imagenet(split):
    return ImageNet(split)


def icon(split):
    img_transform = create_transform(split)
    return IconNet(split, transform=img_transform)
