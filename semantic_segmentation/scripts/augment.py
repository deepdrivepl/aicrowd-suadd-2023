import random
import typing

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class ToTensor(torch.nn.Module):
    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        target = F.pil_to_tensor(target)
        target = F.convert_image_dtype(target, torch.long)
        return image, target


class RandomRotate(T.RandomRotation):
    def forward(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        angle = random.randint(self.degrees[0], self.degrees[1])
        image = F.rotate(image, angle, fill=0)
        target = F.rotate(target, angle, fill=0)
        return image, target


def get_transform(train=False):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomRotate(degrees=30))
    return Compose(transforms)


def get_test_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
