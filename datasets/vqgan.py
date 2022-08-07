import hfai
import pandas as pd
from torchvision import transforms
from .statistic import *

# 自定义 Dataset 所需的头文件
# -------------------
from typing import Callable, Optional

import pickle
from ffrecord import FileReader
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
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
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
        check_data: bool = False
    ) -> None:
        super(IconNet, self).__init__()

        assert split in ["train", "valid"]
        self.split = split
        self.transform = transform
        # 当前目录
        data_dir = get_data_dir()
        # 目录拼接
        self.data_dir = data_dir / "IconNet" / f"{split}"
        # self.fname = self.data_dir / f"{split}.ffr"
        self.reader = FileReader(self.data_dir, check_data)

        # self.meta = pd.read_csv(self.data_dir / "IconNet" / f"{split}.csv", usecols=[0, 1])

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        imgs_bytes = self.reader.read(indices)
        samples = []
        for i, bytes_ in enumerate(imgs_bytes):
            img = pickle.loads(bytes_).convert("RGB")
            samples.append(img)

        transformed_samples = []

        for img in samples:
            if self.transform:
                img = self.transform(img)
            transformed_samples.append(img)

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
