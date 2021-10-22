import argparse
import os
from PIL import Image
import time
import torch
from torchvision import transforms
import sys
sys.path.append("models")
import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--scale', default=2)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inputdir', default=r'load/div2k/DIV2K_valid_LR_bicubic/X2')
    parser.add_argument('--outputdir', default=r'testimg/sirenx2x2')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    inputdir = args.inputdir
    outputdir = args.outputdir
    if os.path.exists(outputdir)==False:
        os.makedirs(outputdir)

    for imgs in os.listdir(inputdir):
        st = time.time()
        print(f"{imgs} start...  ", end="")

        imgpath = os.path.join(inputdir, imgs)
        img = transforms.ToTensor()(Image.open(imgpath))

        h, w = img.shape[1]*args.scale, img.shape[2]*args.scale
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        bimg =((img - 0.5) / 0.5).cuda().unsqueeze(0)
        pred = batched_predict(model, bimg,coord.unsqueeze(0), cell.unsqueeze(0), bsize=1000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        output = os.path.join(outputdir, imgs)
        transforms.ToPILImage()(pred).save(output)

        et = time.time()
        print(f"spend time {(et-st):.2f}s")




