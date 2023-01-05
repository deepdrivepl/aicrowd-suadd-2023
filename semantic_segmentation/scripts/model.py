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
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.model.to(device)
        self.model.eval()
        pred = self.model(image_to_segment)
        segmentation_results = torch.argmax(pred, dim=1).cpu().detach().numpy()
        return segmentation_results
