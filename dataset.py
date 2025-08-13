import os
import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset
from augment import ToTensor, train_augm, valid_augm, train_augm2, valid_augm2, train_augm3


def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


def get_crs(path):
    src = rasterio.open(path, "r")
    return src.crs, src.transform


def save_img(path, img, crs, transform):
    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=img.shape[1],
            width=img.shape[2],
            count=img.shape[0],
            dtype=img.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        dst.write(img)
        dst.close()


class Dataset(BaseDataset):
    def __init__(self, label_list, classes=None, size=128, train=False):
        self.fns = label_list
        self.augm = train_augm3 if train else valid_augm
        self.size = size
        self.train = train
        self.to_tensor = ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 1024)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)


class Dataset2(BaseDataset):
    def __init__(self, root, label_list, classes=None, size=128, train=False):
        self.fns = [os.path.join(root, "labels", x) for x in label_list]
        self.augm = train_augm2 if train else valid_augm2
        self.size = size
        self.train = train
        self.to_tensor = ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.fns[idx])
        osm = self.load_multiband(self.fns[idx].replace("labels", "osm"))

        if self.train:
            data = self.augm({"image": img, "mask": msk, "osm": osm}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk, "osm": osm}, 512)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "z": data["osm"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)


class Dataset3(BaseDataset):
    def __init__(self, root, label_list, classes=None, size=128, train=False):
        self.fns = label_list
        self.augm = train_augm if train else valid_augm
        self.size = size
        self.train = train
        self.to_tensor = ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fns[idx].replace("labels", "images"))
        msk = self.load_grayscale(self.fns[idx])

        if self.train:
            data = self.augm({"image": img, "mask": msk}, self.size)
        else:
            data = self.augm({"image": img, "mask": msk}, 512)
        data = self.to_tensor(data)

        return {"x": data["image"], "y": data["mask"], "fn": self.fns[idx]}

    def __len__(self):
        return len(self.fns)
