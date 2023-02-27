import albumentations as A
import argparse
import cv2
import numpy as np
import os
from PIL import Image
import torch

import models
from pl_model import SuadSemseg

root = "/home/ubuntu/suad_23/"
imgs = list(sorted(os.listdir(os.path.join(root, "inputs")), reverse=True))

def segment_single_image(model, image_to_segment):
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
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            segmentation_results = model(image_to_segment)
        segmentation_results = torch.argmax(segmentation_results, dim=1).cpu().detach().numpy()[0]
        segmentation_results = A.Resize(*image_size, interpolation=0)(image=segmentation_results)['image']
        return segmentation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--resfolder", type=str, default="/home/ubuntu/aicrowd-suadd-2023/semantic_segmentation/evaluation_photos/unet")
    # parser.add_argument("--modelpath", type=str, default="tb_logs/unet/version_2/checkpoints/epoch=199-step=400.ckpt")

    parser.add_argument("--resfolder", type=str, default="/home/ubuntu/aicrowd-suadd-2023/semantic_segmentation/evaluation_photos/xunet")
    parser.add_argument("--modelpath", type=str, default="tb_logs/xunet/version_16/checkpoints/epoch=199-step=18000.ckpt")

    # parser.add_argument("--resfolder", type=str, default="/home/ubuntu/aicrowd-suadd-2023/semantic_segmentation/evaluation_photos/u2net")
    # parser.add_argument("--modelpath", type=str, default="tb_logs/u2net/version_15/checkpoints/epoch=199-step=400.ckpt")

    args = parser.parse_args()
    model = SuadSemseg.load_from_checkpoint(args.modelpath)
    imgs = [file for file in imgs if file.endswith('.png')]

    for img_name in imgs:
        img_path = os.path.join(root, "inputs", img_name)
        img = Image.open(img_path)
        img = np.array(img)
        res = segment_single_image(model, img)
        prediction_path = os.path.join(args.resfolder, img_name)
        res = Image.fromarray(np.uint8(res), mode='L')
        res.save(prediction_path)
