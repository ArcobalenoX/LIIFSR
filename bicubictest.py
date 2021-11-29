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
    testset_dir = r"/home/ww020823/yxc/dataset/WHU-RS19-test/GT"
    testset_dir = r"/home/ww020823/yxc/dataset/selfWHURS/sobel/high-sobel-test"
    testset_dir = r"/home/ww020823/yxc/dataset/selfAID/AID-test-low"
    testset_dir = r"/home/ww020823/yxc/dataset/selfAID/AID-test"
    scale = 2

    sr_dir = "testimg/AID_bicubicx2sr"
    if not os.path.exists(sr_dir):
        os.mkdir(sr_dir)

    val_psnr = utils.Averager()
    val_ssim = utils.Averager()

    for i in os.listdir(testset_dir):
        img_path = os.path.join(testset_dir, i)
        if is_image_file(img_path):
            #tpsnr, tssim = torch_image_bicubic(img_path, scale)
            hr = Image.open(img_path)
            hr = ToTensor()(hr)
            c, h, w = hr.shape
            lr = utils.resize_fn(hr, (h // scale, w // scale))
            sr = utils.resize_fn(lr, (h, w))
            #ToPILImage()(sr).save(os.path.join(sr_dir, i).replace('jpg', 'png'))

            bhr = torch.unsqueeze(hr, 0)
            bsr = torch.unsqueeze(sr, 0)
            tpsnr = utils.calc_psnr(bsr, bhr)
            tssim = utils.ssim(bsr, bhr)

            val_psnr.add(tpsnr.item())
            val_ssim.add(tssim.item())

            print(f"{i} psnr:{tpsnr.item():.4f}  ssim:{tssim.item():.4f} ")
    print(f"psnr:{val_psnr.item():.4f}")
    print(f"ssim:{val_ssim.item():.4f}")











