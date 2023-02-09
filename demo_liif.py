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
from test_liif import batched_predict

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def liif_sr_resolution(model, img, h, w):
    coord = make_coord((h, w)).cuda() #[SR_H*SR_W,2] 左上角[-1,-1]-右下角[1,1]
    cell = torch.ones_like(coord) #[SR_H*SR_W,2] [1*2/SR_H,1*2/SR_W]
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    img = ((img - 0.5) / 0.5).cuda().unsqueeze(0)
    coord = coord.unsqueeze(0)
    cell = cell.unsqueeze(0)    
    pred = batched_predict(model, img, coord, cell, bsize=30000)[0] #[1,SR_H*SR_W,3]
    #print("pred.shape——",pred.shape,pred.min(),pred.max())
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    #print("pred.shape——",pred.shape,pred.min(),pred.max())
    return pred

def liif_sr_scale(model, img, scale):
    h, w = int(img.shape[1]*scale), int(img.shape[2]*scale)
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    bimg =((img - 0.5) / 0.5).cuda().unsqueeze(0)
    pred = batched_predict(model, bimg, coord.unsqueeze(0), cell.unsqueeze(0), bsize=1000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    return pred

#原LIIF
if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='testimg/WHUedx-x2/airport_41.jpg')
    parser.add_argument('--model', default='weights/edsr-baseline-liif.pth')
    parser.add_argument('--scale', default=2)    
    parser.add_argument('--resolution') #SR_H,SR_W
    parser.add_argument('--output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    scale = float(args.scale)

    st = time.time()

    if is_image_file(args.input):
        img = transforms.ToTensor()(Image.open(args.input)) #[3,LR_H,LR_W]

        # h, w = list(map(int, args.resolution.split(',')))
        # pred = liif_sr_resolution(model, img, h, w)

        pred = liif_sr_scale(model, img, scale)
        if args.output:
            transforms.ToPILImage()(pred).save(args.output)
        else:
            transforms.ToPILImage()(pred).save("srX"+str(scale)+"_"+args.input.split(os.sep)[-1])  #_+args.input)


    elif os.path.isdir(args.input):

        inputdir = args.input
        outputdir = "sr_"+args.input
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        for imgs in os.listdir(inputdir):

            print(f"{imgs} +++...  ")

            imgpath = os.path.join(inputdir, imgs)
            img = transforms.ToTensor()(Image.open(imgpath))
            pred = liif_sr_scale(model, img, scale)

            output = os.path.join(outputdir, imgs)
            transforms.ToPILImage()(pred).save(output)



    et=time.time()

    print(f"{args.input} spend time {(et-st):.3f}s")
