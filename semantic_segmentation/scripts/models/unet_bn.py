import segmentation_models_pytorch as smp

def UNET():
    return smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3,
                    classes=16, activation=None)