import argparse
import os
import random
import time
import numpy as np
import yaml
import math
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
from train_x import eval_metric


torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)


#测试通常网络
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=r"configs/test-rs/test-WHURS.yaml")
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save_sr', default=False)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    if args.scale:
        spec['wrapper']['args']['scale'] = int(args.scale)
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=0, pin_memory=True)

    sv_file = torch.load(args.model)
    print(sv_file['epoch'])
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    parments = utils.compute_num_params(model, True)
    print("params:", parments)
    st = time.time()
    psnr, ssim, lpips = eval_metric(loader, model, eval_psnr=True, eval_ssim=True, eval_lpips=True, data_norm=config.get('data_norm'), verbose=True)
    et = time.time()

    print(f'psnr: {psnr:.4f} ssim: {ssim:.4f} lpips: {lpips:.4f}')
    print(f"cost time {(et-st):.2f}")





