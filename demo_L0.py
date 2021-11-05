import argparse
import os
from PIL import Image
import time
import torch
from torchvision import transforms
import sys
sys.path.append("models")

from models import models
from train_L0 import batched_predict
from utils import resize_fn

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='save/ITCVD_drsencax2/epoch-best.pth')
    parser.add_argument('--lr', default=r'/home/ww020823/yxc/dataset/WHU-RS19-test/GT/airport_41.jpg')
    parser.add_argument('--ls', default=r'/home/ww020823/yxc/dataset/smooth_whurs_test/airport_41.jpg')
    parser.add_argument('--sr', default='smooth_airport41.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    st = time.time()

    img = Image.open(args.lr)
    img = transforms.Resize((int(img.height/2), int(img.width/2)), Image.BICUBIC)(img)
    img = transforms.ToTensor()(img) #[3,LR_H,LR_W]
    bimg = ((img - 0.5) / 0.5).cuda().unsqueeze(0)

    ls = Image.open(args.ls)
    ls = transforms.Resize((int(ls.height/2), int(ls.width/2)), Image.BICUBIC)(ls)
    bls = transforms.ToTensor()(ls)
    bls = ((bls - 0.5) / 0.5).cuda().unsqueeze(0)


    pred = batched_predict(model, bimg, bls)[0] #[1,SR_H*SR_W,3]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.sr)
    et = time.time()

    print(f"{os.path.basename(args.lr)} spend time {(et-st):.3f}s")
