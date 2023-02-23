import albumentations as A
import numpy as np
import torch

import models
from pl_model import SuadSemseg
import cv2

class UNETModel:
    def __init__(self):
        self.model = SuadSemseg.load_from_checkpoint("tb_logs/unet/version_1/checkpoints/epoch=99-step=200.ckpt")

    def segment_single_image(self, image_to_segment):
        """
        Implements a function to segment a single image
        Inputs:
            image_to_segment - Single frame from onboard the flight

        Outputs:
            An 2D image with the pixels values corresponding
            to the label number
        """
        infer_size = (512,512)
        
        image_size = image_to_segment.shape
        image_to_segment = np.stack((image_to_segment,)*3, axis=-1)
        image_to_segment = cv2.resize(image_to_segment, infer_size)
        image_to_segment = torch.from_numpy(image_to_segment).permute(2,0,1).unsqueeze(0)
        image_to_segment = (image_to_segment/128.0)-0.5
        
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        image_to_segment = image_to_segment.to(device)
        
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            segmentation_results = self.model(image_to_segment)
        segmentation_results = torch.argmax(pred, dim=1).cpu().detach().numpy()[0]
        segmentation_results = A.Resize(*image_size, interpolation=0)(image=segmentation_results)['image']
        return segmentation_results
