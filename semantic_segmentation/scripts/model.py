import albumentations as A
import torch

from semantic_segmentation.scripts.models.xunet_bn import XUNET


class UNETModel:
    def __init__(self):
        self.model = XUNET()
        self.model.load_from_checkpoint(
            torch.load("tb_logs/xunet/version_0/checkpoints")
        )

    def segment_single_image(self, image_to_segment):
        """
        Implements a function to segment a single image
        Inputs:
            image_to_segment - Single frame from onboard the flight

        Outputs:
            An 2D image with the pixels values corresponding
            to the label number
        """
        image_size = image_to_segment.shape[:2]
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.model.to(device)
        self.model.eval()
        pred = self.model(image_to_segment)
        segmentation_results = torch.argmax(pred, dim=1).cpu().detach().numpy()
        segmentation_results = A.Resize(*image_size, interpolation=0)(
            segmentation_results
        )
        return segmentation_results
