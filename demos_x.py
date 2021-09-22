import argparse
import os
from PIL import Image
import time
import torch
from torchvision import transforms
import sys
sys.path.append("models")
from models import models
from test_x import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--scale',default=2)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inputdir', default=r'E:\Code\Python\datas\RS\WHU-RS19-test\LR\x4')
    parser.add_argument('--outputdir', default=r'testimg/WHURS19-DRSENMKCAx4')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inputdir = args.inputdir
    outputdir = args.outputdir
    if os.path.exists(outputdir)==False:
        os.makedirs(outputdir)

    for imgs in os.listdir(inputdir):
        st = time.time()

        imgpath = os.path.join(inputdir, imgs)
        img = transforms.ToTensor()(Image.open(imgpath)).cuda()
        model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
        bimg =((img - 0.5) / 0.5).cuda().unsqueeze(0)
        pred = batched_predict(model, bimg)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).cpu()

        outputpath = os.path.join(outputdir, imgs).replace(".jpg", ".png")
        transforms.ToPILImage()(pred).save(outputpath)

        et = time.time()
        print(f"{imgs} spend time {(et-st):.3f}s")




