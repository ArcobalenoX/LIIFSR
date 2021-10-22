import os
import time
import torch
from torchvision import transforms
import sys
sys.path.append("models")
import models
from utils import make_coord
from test import batched_predict
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    torch.cuda.empty_cache()
    scale = 2
    model_path = "weights/edsr-baseline-liif.pth"
    model = models.make(torch.load(model_path)['model'], load_sd=True).cuda()

    cap = cv2.VideoCapture(r"E:\Code\Python\iPython\blending\B.mp4")
    cv2.namedWindow("lr")
    cv2.namedWindow("sr")

    while cap.isOpened():
        flag, image = cap.read()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imshow("lr",image)
        h,w,c = image.shape
        image = cv2.resize(image,(w//scale,h//scale))

        st = time.time()
        img = transforms.ToTensor()(image) #[3,LR_H,LR_W]
        h, w = img.shape[-2]*scale, img.shape[-1]*scale
        coord = make_coord((h, w)).cuda() #[SR_H*SR_W,2] 左上角[-1,-1]-右下角[1,1]
        cell = torch.ones_like(coord) #[SR_H*SR_W,2] [1*2/SR_H,1*2/SR_W]
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        img = ((img - 0.5) / 0.5).cuda().unsqueeze(0)
        coord = coord.unsqueeze(0)
        cell = cell.unsqueeze(0)
        pred = batched_predict(model, img, coord, cell, bsize=30000)[0] #[1,SR_H*SR_W,3]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).cpu()

        srimage =  np.array(pred)
        cv2.imshow("sr",srimage)
        cv2.waitKey(100)
        et=time.time()
        print(f"spend time {(et-st):.3f}s")
