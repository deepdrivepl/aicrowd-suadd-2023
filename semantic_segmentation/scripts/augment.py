import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(train=False):
    augments = []
    if train:
        augments.append(A.RandomCrop(width=190, height=275, p=0.3))
        augments.append(A.HorizontalFlip(p=0.5))
        augments.append(A.RandomRotate90(p=0.5))

    augments.append(A.Resize(190, 275))
    augments.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)))
    augments.append(ToTensorV2())
    transform = A.Compose(augments)
    return transform


def get_test_transform():
    transforms = []
    # transforms.append(A.Resize(190, 275))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)
