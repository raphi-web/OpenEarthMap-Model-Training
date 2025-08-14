import numpy as np
import rasterio
from torch.utils.data import Dataset as BaseDataset

from augment import ToTensor, valid_augm, train_augm_light, train_augm_heavy, AugmentKind, test_augm, train_augm_medium


def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)




class Dataset(BaseDataset):
    def __init__(self, label_list, classes=None, size=128, train=False, augment_kind=AugmentKind.MEDIUM):
        self.fns = label_list
        self.size = size
        self.train = train
        self.to_tensor = ToTensor(classes=classes)
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

        match augment_kind:
            case AugmentKind.LIGHT:
                self.augm = train_augm_light
            case AugmentKind.MEDIUM:
                self.augm = train_augm_medium
            case AugmentKind.HEAVY:
                self.augm = train_augm_heavy
            case AugmentKind.VALID:
                self.augm = valid_augm
            case AugmentKind.TEST:
                self.augm =  test_augm


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