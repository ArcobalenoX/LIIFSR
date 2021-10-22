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


def single_image(model, img):
    bimg = ((img - 0.5) / 0.5).cuda().unsqueeze(0)
    pred = batched_predict(model, bimg)[0] #[1,SR_H*SR_W,3]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).cpu()
    return pred


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='save/ITCVD_drsencax2/epoch-best.pth')
    parser.add_argument('--lr', default=r'E:\Code\Python\datas\RS\ITCVD\Test\patch\007_0_0.png')
    parser.add_argument('--sr', default='007_0_0x2.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    st = time.time()
    img = Image.open(args.lr)
    #img = transforms.Resize((int(img.height/2),int(img.width/2)),Image.BICUBIC)(img)
    img = transforms.ToTensor()(img) #[3,LR_H,LR_W]
    pred = single_image(model, img)
    transforms.ToPILImage()(pred).save(args.sr)
    et = time.time()

    print(f"{os.path.basename(args.lr)} spend time {(et-st):.3f}s")
