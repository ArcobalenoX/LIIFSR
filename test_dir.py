import argparse
import os
from PIL import Image
import time
import csv
import torch
from torchvision import transforms
import sys
sys.path.append("models")
import utils
from models import models

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

#测试LR文件夹并保存SR图像
if __name__ == '__main__':

    torch.cuda.empty_cache()    

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=r"save\WHURS19-edsrblx2\epoch-last.pth")  
    parser.add_argument('--lrdir', default=r'E:\Code\Python\datas\RS\WHU-RS19-test\LR\x2')
    parser.add_argument('--srdir', default=r'testimg/WHURS19_edsrblx2')
    parser.add_argument('--hrdir', default=r'E:\Code\Python\datas\RS\WHU-RS19-test\GT')
    args = parser.parse_args()

    lrdir = args.lrdir
    hrdir = args.hrdir
    srdir = args.srdir
    if not os.path.exists(srdir):
        os.makedirs(srdir)

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    csv_path = srdir + '\\' + args.model.split('/')[1] + ".csv"
    print(csv_path)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for imgname in os.listdir(lrdir):

            lrpath = os.path.join(lrdir, imgname)
            lr = Image.open(lrpath)
            lr = transforms.Resize((int(lr.height / 2), int(lr.width / 2)), Image.BICUBIC)(lr)
            lr = transforms.ToTensor()(lr) #[3,H,W]       

            hrpath = os.path.join(hrdir, imgname)
            hr = Image.open(hrpath)        
            hr = transforms.ToTensor()(hr).unsqueeze(0) #[3,H,W]


            sr = single_image(model, lr).unsqueeze(0)
            psnr = utils.calc_psnr(sr, hr)
            ssim = utils.ssim(sr, hr)      

            srpath = os.path.join(srdir, imgname)

            if srpath.endswith("jpg"):
                srpath = srpath.replace("jpg", "png")
            transforms.ToPILImage()(sr[0]).save(srpath)

            psnr = f"{psnr:.4f}"
            ssim =f"{ssim:.4f}"

            print(f"{imgname} {psnr} {ssim}")

            writer.writerow([imgname,psnr,ssim]) 



