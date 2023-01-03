import os

import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "inputs"))))
        count = len(self.imgs)
        if train:
            self.imgs = self.imgs[int(count*0.8):]
            self.targets = list(sorted(os.listdir(os.path.join(root,
                                "semantic_annotations"))))[int(count*0.8):]
        else:
            self.imgs = self.imgs[:int(count*0.8)]
            self.targets = list(sorted(os.listdir(os.path.join(root,
                                "semantic_annotations"))))[:int(count*0.8)]

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "inputs", self.imgs[idx])
        target_path = os.path.join(self.root, "semantic_annotations",
                                   self.targets[idx])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        if width != 512 or height != 512:
            img = img.resize((512, 512))
        target = Image.open(target_path).convert("RGB")
        width, height = target.size
        if width != 512 or height != 512:
            target = target.resize((512, 512))
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "inputs"))))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "inputs", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        if width != 512 or height != 512:
            img = img.resize((512, 512))
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.imgs[idx]  # for saving purposes

    def __len__(self):
        return len(self.imgs)
