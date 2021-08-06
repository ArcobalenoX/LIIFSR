import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim     #64
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2          # attach coord 指定查询像素的坐标 [x,y]
            if self.cell_decode:
                imnet_in_dim += 2      #[Cell_h, Cell_w]指定查询像素的高度和宽度的两个值
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp) #[N,C,LR_H,LR_W]
        return self.feat

    def query_rgb(self, coord, cell=None):
        #coord [N, SR_H*SR_*W, 2]
        #cell [N, SR_H*SR_*W, 2]
        feat = self.feat #[N, C, LR_H, LR_W]

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)
            #[N, C, 1, SR_H*SR_*W]
            ret = ret[:, :, 0, :].permute(0, 2, 1)
            #[N, SR_H*SR_*W, C]
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            #[N, C*3*3, SR_H*, SR_*W]
 
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2  #2/LR_H/2
        ry = 2 / feat.shape[-1] / 2  #2/LR_W/2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()
        #[LR_H,LR_W,2]

        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        #[N,2,LR_H,LR_W]

        preds = []#邻近4点的预测RGB值
        areas = []#邻近4点的面积
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()#[N,SR_H*SR_W,2]
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                coord_ = coord_.flip(-1).unsqueeze(1)#[N,1,SR_H*SR_W,2]
                q_feat = F.grid_sample(feat, coord_, mode='nearest', align_corners=False)
                #[N,C*9,1,SR_H*SR_W]
                q_feat = q_feat[:, :, 0, :].permute(0, 2, 1)#[N,SR_H*SR_W,C*9]

                q_coord = F.grid_sample(feat_coord, coord_, mode='nearest', align_corners=False)
                #[N,2,1,SR_H*SR_W]
                q_coord = q_coord[:, :, 0, :].permute(0, 2, 1)#[N,SR_H*SR_W,2]
                rel_coord = coord - q_coord #[N,SR_H*SR_W,2]
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1) #[N,SR_H*SR_W,C*9+2]

                if self.cell_decode:
                    rel_cell = cell.clone()#[N,SR_H*SR_W,2]
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1) #[N,SR_H*SR_W,C*9+2+2]

                bs, q = coord.shape[:2] #bs=N q=SR_H*SR_W
                inp = inp.view(bs * q, -1)#[N*SR_H*SR_W,C*9+2+2]
                pred = self.imnet(inp) #[N*SR_H*SR_W,3]
                pred = pred.view(bs, q, -1)#[N,SR_H*SR_W,3]
                preds.append(pred) #[[N,SR_H*SR_W,3],[N,SR_H*SR_W,3],[N,SR_H*SR_W,3],[N,SR_H*SR_W,3]]

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9) #[[N,SR_H*SR_W],[N,SR_H*SR_W],[N,SR_H*SR_W],[N,SR_H*SR_W]]

        tot_area = torch.stack(areas).sum(dim=0) #[N,SR_H*SR_W]
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t #swap(areas[0],areas[3])
            t = areas[1]; areas[1] = areas[2]; areas[2] = t #swap(areas[1],areas[2])
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


