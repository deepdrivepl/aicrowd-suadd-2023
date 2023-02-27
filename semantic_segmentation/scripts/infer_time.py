import albumentations as A
import argparse
import cv2
import numpy as np
import os
from PIL import Image
import torch
import time
from tqdm import tqdm

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
        
        image_to_segment = image_to_segment.cuda().half()
        start = time.time()
        with torch.no_grad():
            segmentation_results = model(image_to_segment)
        segmentation_results = torch.argmax(segmentation_results, dim=1).cpu().detach().numpy()[0]
        t = time.time() - start
        segmentation_results = A.Resize(*image_size, interpolation=0)(image=segmentation_results)['image']
        return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="XUNET")
    parser = SuadSemseg.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)
    net = getattr(models, args.net)()
    dict_args = dict(dict_args)
    dict_args["net"] = net
    model = SuadSemseg(**dict_args)

    model.eval().cuda().half()
    
    all_time = 0
    imgs = [file for file in imgs if file.endswith('.png')]

    for img_name in tqdm(imgs, total=len(imgs)):
        img_path = os.path.join(root, "inputs", img_name)
        img = Image.open(img_path)
        img = np.array(img)
        res = segment_single_image(model, img)
        all_time += res
        
    print(all_time/len(imgs))