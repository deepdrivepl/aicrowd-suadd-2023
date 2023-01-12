import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(size=(256, 384), train=False):
    augs = []
    if train:
        augs.append(
            A.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=(-0.4, 0.6), rotate_limit=20
            )
        )
        augs.append(A.HorizontalFlip(p=0.5))

    augs.append(A.Resize(*size))
    augs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    augs.append(ToTensorV2())
    return A.Compose(augs)


def get_test_transform(size=(256, 384)):
    augs = []
    augs.append(A.Resize(*size))
    augs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    augs.append(ToTensorV2())
    return A.Compose(augs)
