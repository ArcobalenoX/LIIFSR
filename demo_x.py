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
from train_x import batched_predict
from utils import calc_psnr, ssim


#通常网络
if __name__ == '__main__':
    st = time.time()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='save/WHURS19_edsrblx4/epoch-last.pth')
    parser.add_argument('--lrdir', default=r'load/selfWHURS/WHURS-test/LR/x4')
    parser.add_argument('--hrdir', default=r'load/selfWHURS/WHURS-test/GT')
    parser.add_argument('--scale', default=4)

    args = parser.parse_args()

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    model_name = args.model.split(os.sep)[-2]

    scale = int(args.scale)
    lr_dir = args.lrdir
    hr_dir = args.hrdir
    sr_dir = os.path.join('AID', model_name)
    if not os.path.exists(sr_dir):
        os.makedirs(sr_dir)
    result_csv = os.path.join(sr_dir, model_name+".csv")

    psnr_cnt = []
    ssim_cnt = []

    with open(result_csv, "w+", newline='') as f:
        writer = csv.writer(f)
        for name in os.listdir(hr_dir):
            hr_path = os.path.join(hr_dir, name)
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = transforms.Resize((int(hr_img.height/scale), int(hr_img.width/scale)), Image.BICUBIC)(hr_img)
            # lr_path = os.path.join(lr_dir, name)
            # lr_img.save(lr_path)
            # lr_img = Image.open(lr_path).convert('RGB')

            lr = transforms.ToTensor()(lr_img).cuda().unsqueeze(0)
            hr = transforms.ToTensor()(hr_img).cuda().unsqueeze(0)
            lr = ((lr - 0.5) / 0.5)
            pred = batched_predict(model, lr)
            pred = (pred * 0.5 + 0.5).clamp(0, 1)

            psnr_v = calc_psnr(pred, hr).item()
            ssim_v = ssim(pred, hr).item()
            psnr_cnt.append(psnr_v)
            ssim_cnt.append(ssim_v)

            transforms.ToPILImage()(pred[0].cpu()).save(os.path.join(sr_dir, name).replace('jpg', 'png'))
            print(name, psnr_v, ssim_v)
            writer.writerow([name, psnr_v, ssim_v])
        et = time.time()

        psnr = np.mean(psnr_cnt)
        ssim = np.mean(ssim_cnt)
        print(f'psnr: {psnr:.4f} ssim: {ssim:.4f}')
        print('time:', et-st)

