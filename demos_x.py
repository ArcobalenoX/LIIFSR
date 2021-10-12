import argparse
import os
from PIL import Image
import time
import torch
from torchvision import transforms
import sys
sys.path.append("models")
from models import models
from demo_x import single_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--inputdir', default=r'E:\Code\Python\datas\RS\WHU-RS19-test\LR\x2')
    parser.add_argument('--outputdir', default=r'testimg/WHURS19-DRSENMKCAX2')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inputdir = args.inputdir
    outputdir = args.outputdir
    if os.path.exists(outputdir)==False:
        os.makedirs(outputdir)

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    for imgname in os.listdir(inputdir):
        st = time.time()
        imgpath = os.path.join(inputdir, imgname)
        img = Image.open(imgpath)
        pred = single_image(model, img)
        outputpath = os.path.join(outputdir, imgname)
        #outputpath = outputpath.replace(".jpg", ".png")
        transforms.ToPILImage()(pred).save(outputpath)
        et = time.time()
        print(f"{imgname} spend time {(et-st):.3f}s")




