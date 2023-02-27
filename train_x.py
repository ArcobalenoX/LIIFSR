import argparse
import math
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append("models")
import datasets
from models import models
import utils
from models.losses import CharbonnierLoss, EdgeLoss, SSIMLoss, AdversarialLoss, PerceptualLoss

import lpips

def batched_predict(model, inp):
    model.eval()
    with torch.no_grad():
        pred = model(inp)
    return pred

def eval_metric(loader, model, eval_psnr=True, eval_ssim=True, eval_lpips=True, data_norm=None, verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]} 
        }

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).cuda()
    inp_div = torch.FloatTensor(t['div']).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).cuda()
    gt_div = torch.FloatTensor(t['div']).cuda()


    val_psnr = utils.Averager()
    val_ssim = utils.Averager()

    # Linearly calibrated models (LPIPS)
    if eval_lpips:
        lpips_fn = lpips.LPIPS(net='alex').cuda() # Can also set net = 'squeeze' or 'vgg'
    val_lpips = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = (batch['inp'] - inp_sub) / inp_div

        # inp = transforms.GaussianBlur(5, 0.5)(inp)#加噪

        pred = batched_predict(model, inp)
        pred = (pred * gt_div + gt_sub).clamp_(0, 1)

        if eval_psnr:
            psnrv = utils.calc_psnr(pred, batch['gt'])
            val_psnr.add(psnrv.item(), inp.shape[0])

        if eval_ssim:
            ssimv = utils.ssim(pred, batch['gt'])
            val_ssim.add(ssimv.item(), inp.shape[0])

        if eval_lpips:
            lpipsv = lpips_fn.forward(pred, batch['gt'])
            val_lpips.add(lpipsv.item(), inp.shape[0])

        if verbose:
            pbar.set_description(f'PSNR {val_psnr.item():.4f} SSIM {val_ssim.item():.4f} lpips {val_lpips.item():.4f}')

    return val_psnr.item(), val_ssim.item(), val_lpips.item()


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log(f'{tag} dataset: size={len(dataset)}')
    for k, v in dataset[0].items():
        log(f'  {k}: shape={tuple(v.shape)}')

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=0, pin_memory=False)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1

        if config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        elif config.get('cosine_annealing_lr') is not None:
            lr_scheduler = CosineAnnealingLR(optimizer, **config['cosine_annealing_lr'])
        else:
            lr_scheduler = None

    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1

        if config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        elif config.get('cosine_annealing_lr') is not None:
            lr_scheduler = CosineAnnealingLR(optimizer, **config['cosine_annealing_lr'])
        else:
            lr_scheduler = None

    log(f'model: #params={utils.compute_num_params(model, text=True)}')
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    train_loss = utils.Averager()

    loss_L1 = nn.L1Loss()
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    criterion_ssim = SSIMLoss()
    criterion_perceptual = PerceptualLoss()

    train_dataset = config['train_dataset']
    inp_size = train_dataset['wrapper']['args']['inp_size']
    scale = train_dataset['wrapper']['args']['scale']
    bs = train_dataset['batch_size']

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).cuda()
    inp_div = torch.FloatTensor(t['div']).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).cuda()
    gt_div = torch.FloatTensor(t['div']).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp)
        gt = (batch['gt'] - gt_sub) / gt_div

        save_img = 0
        if save_img:
            predimg = (pred * gt_div + gt_sub).clamp(0, 1)
            gtimg = (gt * gt_div + gt_sub).clamp(0, 1)
            save_image((inp * inp_div + inp_sub).clamp(0, 1), f"vis/inp.jpg", nrow=int(math.sqrt(bs)))
            save_image(predimg, f"vis/predimg.jpg", nrow=int(math.sqrt(bs)))
            save_image(gtimg, f"vis/gtimg.jpg", nrow=int(math.sqrt(bs)))


        loss_char = criterion_char(pred, gt)
        # loss_edge = criterion_edge(pred, gt)
        # loss_ssim = criterion_ssim(pred, gt)
        # pred_dual = F.interpolate(pred, scale_factor=1/scale, mode='bicubic')
        # loss_dual = criterion_char(pred_dual, inp)
        # loss_perc = criterion_perceptual(pred, gt)
        # loss = loss_char + loss_edge  + loss_perc*0.1  + loss_dual + (1-loss_ssim)*0.1
        # loss = loss_char + loss_edge  + (1-loss_ssim)*0.1
        loss = loss_L1(pred, gt) #单独L1损失

        print_loss = 0
        if print_loss:
            print(f"char: {loss_char}")
            print(f"edge: {loss_edge}")
            print(f"dual: {loss_dual}")            
            print(f"ssim: {loss_ssim}")
            print(f"perc: {loss_perc}")           
            print(f"total_loss: {loss}")


        train_loss.add(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    #多GPU并行
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_psnr = -1e18
    max_val_ssim = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = [f'epoch {epoch}/{epoch_max}']

        lr = optimizer.param_groups[0]['lr']
        #print(f"epoch--{epoch},lr--{lr}")
        writer.add_scalar('lr', lr, epoch)

        train_loss = train(train_loader, model, optimizer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append(f'train: loss={train_loss:.4f}')
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
 
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, f'epoch-{epoch}.pth'))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_psnr, val_ssim, val_lpips = eval_metric(val_loader, model_, eval_psnr=True, eval_ssim=True, eval_lpips=False,
                                            data_norm=config['data_norm'], verbose=False)
            log_info.append(f'val: psnr={val_psnr:.4f} ssim={val_ssim:.4f}')
            writer.add_scalars('psnr', {'val': val_psnr}, epoch)
            writer.add_scalars('ssim', {'val': val_ssim}, epoch)
            if val_psnr > max_val_psnr:
                max_val_psnr = val_psnr
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-psnr.pth'))
            if val_ssim > max_val_ssim:
                max_val_ssim = val_ssim
                torch.save(sv_file, os.path.join(save_path, 'epoch-best-ssim.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append(f'{t_epoch} {t_elapsed}/{t_all}')

        log(', '.join(log_info))
        writer.flush()

#在L1上增加了loss
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    torch.cuda.empty_cache()

    #载入配置文件的参数
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #保存的checkpoint路径
    save_name = args.name
    if save_name is None:
        save_name = args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('save', save_name)

    main(config, save_path)
