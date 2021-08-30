import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from discriminator import Discriminator


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)     # filter
        down = filtered[:, :, ::2, ::2]         # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4     # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self,use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1,
                 lr_dis=1e-4, train_crop_size=40):
        super().__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = Discriminator(train_crop_size).to(self.device)
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = Adam(self.discriminator.parameters(), lr=lr_dis,
                                        betas=(0, 0.9), eps=1e-8
                                        )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)
        self.bcewithlogits_loss = torch.nn.BCEWithLogitsLoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])

    def forward(self, fake, real):
        fake_detach = fake.detach().clone()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()

            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.reshape(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bcewithlogits_loss(d_fake_for_g, valid_score)
            # loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict
