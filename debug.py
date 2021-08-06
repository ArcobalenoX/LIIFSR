import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import numpy as np
from torchvision import transforms
import models
import utils
from utils import make_coord,set_save_path,ssim
import datasets
from test import eval_psnr,batched_predict


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

modelpath = r'weights\edsr-baseline-liif.pth'
lr_path = r'testimg\div2klrx4\0801x4.png'
hr_path = r'load\div2k\DIV2K_valid_HR\0801.png'
sr_path = r'testimg\ouput.jpg'

lr = transforms.ToTensor()(Image.open(lr_path))
hr = transforms.ToTensor()(Image.open(hr_path))
model = models.make(torch.load(modelpath)['model'], load_sd=True).cuda()
lr = ((lr - 0.5) / 0.5).cuda().unsqueeze(0)
with torch.no_grad():
    feat = model.gen_feat(lr)
featimg= transforms.ToPILImage()(feat[0][0])
plt.imshow(featimg)