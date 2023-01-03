import torch

from semantic_segmentation.scripts.unet_bn import U2NET_lite


class UNETModel:
    def __init__(self):
        self.model = U2NET_lite()
        self.model.load_state_dict(torch.load('model_weights.pth'))

    def segment_single_image(self, image_to_segment):
        """
        Implements a function to segment a single image
        Inputs:
            image_to_segment - Single frame from onboard the flight

        Outputs:
            An 2D image with the pixels values corresponding
            to the label number
        """
        segmentation_results = self.model(image_to_segment)[0]
        return segmentation_results
