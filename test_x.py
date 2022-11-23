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
import sys
sys.path.append("models")
import datasets
import models
import utils
from train_x import eval_psnr_ssim, eval_lpips


torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)


#测试通常网络
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save_sr', default=False)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)



    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    spec['wrapper']['args']['scale'] = int(args.model.split('/')[-2][-1])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=0, pin_memory=True)

    sv_file = torch.load(args.model)
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    parments = utils.compute_num_params(model, True)
    print("params:", parments)
    st = time.time()
    psnr, ssim = eval_psnr_ssim(loader, model, data_norm=config.get('data_norm'), verbose=True)
    lpips = eval_lpips(loader, model, data_norm=config.get('data_norm'), verbose=True)
    et = time.time()

    print(f'result: psnr={psnr:.4f} ssim={ssim:.4f}')
    print(f'lpips={lpips:.4f}')
    print(f"cost time {et-st}")





