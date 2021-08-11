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
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='railway_station_565.jpg')
    parser.add_argument('--model', default='save/_train_edx/epoch-best.pth')
    parser.add_argument('--scale', default=2)
    parser.add_argument('--output', default='railway_station_565_2x2.jpg')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    st = time.time()
    img = Image.open(args.input)
    #img = transforms.Resize((int(img.height/2),int(img.width/2)),Image.BICUBIC)(img)
    timg = transforms.ToTensor()(img) #[3,LR_H,LR_W]

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    bimg = ((timg - 0.5) / 0.5).cuda().unsqueeze(0)
    pred = batched_predict(model, bimg)[0] #[1,SR_H*SR_W,3]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
    et=time.time()

    print(f"{input} spend time {(et-st):.3f}s")
