import argparse
import os
import random
import numpy as np
import yaml
import math
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("models")

import datasets
import models
import utils
from train_L0 import eval_psnr_ssim,eval_lpips


#测试加入了L0smooth的网络
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="configs/test-rs/test-WHURS-L0grad.yaml")
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save_sr', default=False)
    parser.add_argument('--scale')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    if args.scale is not None:
        spec['wrapper']['args']['scale'] = int(args.scale)
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=0, pin_memory=True)

    sv_file = torch.load(args.model)
    #print(f'epoch——{sv_file["epoch"]}')

    model_spec = torch.load(args.model)['model']
    #print(model_spec['sd'])
    #print(model_spec)
    model = models.make(model_spec, load_sd=True).cuda()
    #print(model)
    parments = utils.compute_num_params(model, True)
    print("params:", parments)

    st = time.time()
    psnr, ssim = eval_psnr_ssim(loader, model, data_norm=config.get('data_norm'), verbose=True)
    lpips= eval_lpips(loader, model, data_norm=config.get('data_norm'), verbose=True)
    et = time.time()

    print(f'result: psnr={psnr:.4f} ssim={ssim:.4f} lpips={lpips:.4f}')
    print(f"cost time {et-st}")




