from x_unet import XUnet


def get_xunet():
    return XUnet(
        dim=64,
        channels=3,
        out_dim=16,
        dim_mults=(1, 2, 4, 8),
        nested_unet_depths=(7, 4, 2, 1),
        consolidate_upsample_fmaps=True,
    )
