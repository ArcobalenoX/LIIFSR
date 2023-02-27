import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import sys
sys.path.append("models")
import utils
from torchvision import transforms
import random
import csv


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def torch_image_bicubic(img_path, scale):
    hr = Image.open(img_path)
    hr = ToTensor()(hr)
    c, h, w = hr.shape
    lr = utils.resize_fn(hr, (h//scale, w//scale))
    sr = utils.resize_fn(lr, (h, w))
    bhr = torch.unsqueeze(hr, 0)
    bsr = torch.unsqueeze(sr, 0)
    psnr = utils.calc_psnr(bsr, bhr)
    ssim = utils.ssim(bsr, bhr)
    return psnr, ssim



if __name__ == "__main__":
    # testset_dir = r"load/selfWHURS/WHURS-test/GT"
    #testset_dir = r"load/selfWHURS/sobel/high-sobel-test"
    testset_dir = r"load/selfAID/AID-test"
    # testset_dir = r"load/selfRSSCN/RSSCN-test"

    scale = 4

    sr_dir = r"AID/AID_bicubicx"+str(scale)
    if not os.path.exists(sr_dir):
        os.mkdir(sr_dir)

    psnr_cnt = []
    ssim_cnt = []

    result_csv = os.path.join(sr_dir, "bicubicx"+str(scale)+".csv")
    with open(result_csv, "w+", newline='') as f:
        writer = csv.writer(f)
        for name in os.listdir(testset_dir):
            img_path = os.path.join(testset_dir, name)
            if is_image_file(img_path):
                hr = Image.open(img_path)
                hr = ToTensor()(hr)
                c, h, w = hr.shape
                lr = utils.resize_fn(hr, (h // scale, w // scale))
                sr = utils.resize_fn(lr, (h, w))
                # ToPILImage()(sr).save(os.path.join(sr_dir, name).replace('jpg', 'png'))

                bhr = torch.unsqueeze(hr, 0)
                bsr = torch.unsqueeze(sr, 0)
                psnr_v = utils.calc_psnr(bsr, bhr).item()
                ssim_v = utils.ssim(bsr, bhr).item()
                writer.writerow([name, psnr_v, ssim_v])

                psnr_cnt.append(psnr_v)
                ssim_cnt.append(ssim_v)
                print(name, psnr_v, ssim_v)

    psnr = np.mean(psnr_cnt)
    ssim = np.mean(ssim_cnt)
    print(f'psnr: {psnr:.4f} ssim: {ssim:.4f}')











