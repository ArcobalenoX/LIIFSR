import argparse
import os
from PIL import Image
import time
import torch
from torchvision import transforms
import sys
sys.path.append("models")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv
import numpy as np
from models import models
from train_L0 import batched_predict
from utils import calc_psnr, ssim

#带L0smooth梯度图
if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='save/WHURS19_edsrblx2/epoch-best.pth')
    parser.add_argument('--lrdir', default=r'data/selfWHURS/sobel/low-sobel-test')
    parser.add_argument('--lsdir', default=r'data/selfWHURS/smooth/smooth-whurs-test-low-grad')
    parser.add_argument('--hrdir', default=r'data/selfWHURS/sobel/low-sobel-test')
    args = parser.parse_args()

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    scale = 4
    lr_dir = args.lrdir
    hr_dir = args.hrdir
    ls_dir = args.lsdir
    sr_dir = os.path.join('AID', args.model.split(os.sep)[-2])
    model_name = args.model.split(os.sep)[-2]
    #sr_dir = r"testimg/WHURS19_samxhighx2_low"
    if not os.path.exists(sr_dir):
        os.makedirs(sr_dir)
    result_csv = os.path.join(sr_dir, model_name+".csv")

    psnr_cnt = []
    ssim_cnt = []

    with open(result_csv, "w+", newline='') as f:
        writer = csv.writer(f)
        for name in os.listdir(hr_dir):
            #lr_path = os.path.join(lr_dir, name)
            #lr_img = Image.open(lr_path).convert('RGB')
            ls_path = os.path.join(ls_dir, name)
            ls_img = Image.open(ls_path).convert('RGB')
            hr_path = os.path.join(hr_dir, name)
            hr_img = Image.open(hr_path).convert('RGB')
            img = transforms.Resize((int(hr_img.height/scale), int(hr_img.width/scale)), Image.BICUBIC)(hr_img)
            ls = transforms.Resize((int(hr_img.height/scale), int(hr_img.width/scale)), Image.BICUBIC)(ls_img)
            #img.save(lr_path)

            bimg = transforms.ToTensor()(img)
            bimg = ((bimg - 0.5) / 0.5).cuda().unsqueeze(0)
            bls = transforms.ToTensor()(ls)
            bls = ((bls - 0.5) / 0.5).cuda().unsqueeze(0)

            pred = batched_predict(model, bimg, bls)
            pred = (pred * 0.5 + 0.5).clamp(0, 1)

            hr = transforms.ToTensor()(hr_img).cuda().unsqueeze(0)
            psnr_v = calc_psnr(pred, hr).item()
            ssim_v = ssim(pred, hr).item()
            psnr_cnt.append(psnr_v)
            ssim_cnt.append(ssim_v)

            transforms.ToPILImage()(pred[0].cpu()).save(os.path.join(sr_dir, name).replace('jpg', 'png'))
            print(name, psnr_v, ssim_v)
            writer.writerow([name, psnr_v, ssim_v])

        psnr = np.mean(psnr_cnt)
        ssim = np.mean(ssim_cnt)
        print(f'psnr: {psnr:.4f} ssim: {ssim:.4f}')



