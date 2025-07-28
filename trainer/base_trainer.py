import cv2
import math
import torch
from PIL import Image
import copy
import random
import numpy as np
import torch.nn.functional as F
import utils
from skimage.morphology import skeletonize as skimage_skeletionize
from pathlib import Path
from .trainer_utils import *
try:
    from apex import amp
except ImportError:
    print('failed to import apex')



def VThin(image, array):
    h, w = image.shape[:2]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1  
                if image[i, j] == 0 and M != 0:  
                    a = [0] * 9  
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128   
                    image[i, j] = array[sum] * 255  
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h, w = image.shape[:2]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def Xihua(binary, array, num = 10):
    image = binary.copy()
    for i in range(num):
        VThin(image, array)
        HThin(image, array)
    return image

array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
         0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
         1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]


def skeletionize(image_tensor):

    batch_size, channels, height, width = image_tensor.shape
    skeleton_tensor = torch.zeros((batch_size, channels, height, width)).float().cuda()

    for b in range(batch_size):
        for c in range(channels):
            image = image_tensor[b, c].detach().cpu().numpy() * 255
            image = image.astype(np.uint8)
            ret, binary = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)
            iThin = Xihua(binary, array)
            skeleton = torch.from_numpy(iThin).float().cuda() / 255.0
            skeleton_tensor[b, c] = skeleton

    return skeleton_tensor

def gaussian_create():
    sigma1 = sigma2 = 1
    gaussian_sum = 0
    g = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            g[i, j] = math.exp(-1/2*(np.square(i-1)/np.square(sigma1)
                                     +(np.square(j-1)/np.square(sigma2))))/(2*math.pi*sigma1*sigma2)
            gaussian_sum = gaussian_sum + g[i,j]
    g = g/gaussian_sum
    return g


def gray_fuc(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gaussian_blur(gray_img, g):
    gray_img = np.pad(gray_img, ((1,1),(1,1)), constant_values = 0)
    h, w = gray_img.shape
    new_gray_img = np.zeros([h-2, w-2])
    for i in range(h-2):
        for j in range(w-2):
            new_gray_img[i, j] = np.sum(gray_img[i: i+3, j:j+3]*g)
    return new_gray_img


def partial_derivative(new_gray_img):
    new_gray_img = np.pad(new_gray_img, ((0,1),(0,1)), constant_values = 0)
    h, w = new_gray_img.shape
    dx_gray = np.zeros([h-1, w-1])
    dy_gray = np.zeros([h-1, w-1])
    df_gray = np.zeros([h-1, w-1])
    for i in range(h-1):
        for j in range(w-1):
            dx_gray[i, j] = new_gray_img[i, j+1] - new_gray_img[i, j]
            dy_gray[i, j] = new_gray_img[i+1, j] - new_gray_img[i, j]
            df_gray[i, j] = np.sqrt(np.square(dx_gray[i, j]) + np.square(dy_gray[i, j]))
    return dx_gray, dy_gray, df_gray


def non_maximum_suppression(dx_gray, dy_gray, df_gray):
    df_gray=np.pad(df_gray,((1,1),(1,1)),constant_values=0)
    h,w=df_gray.shape
    for i in range(1,h-1):
        for j in range(1,w-1):
            if df_gray[i,j] != 0:
                gx=math.fabs(dx_gray[i-1,j-1])
                gy=math.fabs(dy_gray[i-1,j-1])
                if gx>gy:
                    weight=gy/gx
                    grad1=df_gray[i+1,j]
                    grad2=df_gray[i-1,j]
                    if gx*gy>0:
                        grad3=df_gray[i+1,j+1]
                        grad4=df_gray[i-1,j-1]
                    else:
                        grad3=df_gray[i+1,j-1]
                        grad4=df_gray[i-1,j+1]
                else:
                    weight=gx/gy
                    grad1=df_gray[i,j+1]
                    grad2=df_gray[i,j-1]
                    if gx*gy>0:
                        grad3=df_gray[i+1,j+1]
                        grad4=df_gray[i-1,j-1]
                    else:
                        grad3=df_gray[i+1,j-1]
                        grad4=df_gray[i-1,j+1]
                t1=weight*grad1+(1-weight)*grad3
                t2=weight*grad2+(1-weight)*grad4
                if df_gray[i,j]>t1 and df_gray[i,j]>t2:
                    df_gray[i,j]=df_gray[i,j]
                else:
                    df_gray[i,j]=0
    return df_gray


def double_threshold(df_gray, low, high):
    h,w=df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] < low:
                df_gray[i, j] = 1
            elif df_gray[i, j] > high:
                df_gray[i, j] = 0
            elif (df_gray[i, j - 1] > high) or (df_gray[i - 1, j - 1] > high) or (
                    df_gray[i + 1, j - 1] > high) or (df_gray[i - 1, j] > high) or (df_gray[i + 1, j] > high) or (
                    df_gray[i - 1, j + 1] > high) or (df_gray[i, j + 1] > high) or (df_gray[i + 1, j + 1] > high):
                df_gray[i, j] = 0
            else:
                df_gray[i, j] = 1
    return df_gray


def bianyuantiqu(image_tensor):
    batch_size, channels, height, width = image_tensor.shape
    edge_tensor = torch.zeros((batch_size, channels, height + 2, width + 2)).float().cuda()  
    gaussian = gaussian_create()

    for b in range(batch_size):
        for c in range(channels):
            image = image_tensor[b, c].detach().cpu().numpy()

            new_gray = gaussian_blur(image, gaussian)
            d = partial_derivative(new_gray)
            dx, dy, df = d[0], d[1], d[2]

            new_df = non_maximum_suppression(dx, dy, df)

            low_threshold = 0.1 * new_df.max()
            high_threshold = 0.3 * new_df.max()
            result = double_threshold(new_df, low_threshold, high_threshold)
            result = torch.from_numpy(result).float().cuda()
            edge_tensor[b, c] = result

    return edge_tensor


class BaseTrainer:
    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg):

        self.gen = gen 
        self.gen_ema = copy.deepcopy(self.gen)  

        self.g_optim = g_optim  
        self.g_scheduler = g_scheduler  
        # self.is_bn_gen = has_bn(self.gen) 
        self.disc = disc  
        self.d_optim = d_optim  
        self.d_scheduler = d_scheduler  
        self.cfg = cfg 

        self.cross_entropy_loss = nn.CrossEntropyLoss()  

        self.batch_size = cfg.batch_size 
        self.num_component_object = cfg.num_embeddings   
        self.num_channels = cfg.num_channels   
        self.num_postive_samples = cfg.num_positive_samples  

        [self.gen, self.gen_ema, self.disc], [self.g_optim, self.d_optim] = self.set_model(
            [self.gen, self.gen_ema, self.disc],
            [self.g_optim, self.d_optim],
        )

        self.logger = logger
        self.evaluator = evaluator
        self.cv_loaders = cv_loaders

        self.step = 1   

        self.g_losses = {}   
        self.d_losses = {}  

        self.projection_style = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        ).cuda()  

    def set_model(self, models, opts):
        # if torch.cuda.device_count()>1:
        #     models = nn.DataParallel(models)
        return models, opts


    def clear_losses(self):
        """ Integrate & clear loss json_dict """
        # g losses
        loss_dic = {k: v.item() for k, v in self.g_losses.items()}  
        loss_dic['g_total'] = sum(loss_dic.values())
        # d losses
        loss_dic.update({k: v.item() for k, v in self.d_losses.items()}) 

        self.g_losses = {}
        self.d_losses = {}

        return loss_dic


    def accum_g(self, decay=0.999):
        par1 = dict(self.gen_ema.named_parameters()) 
        par2 = dict(self.gen.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))   

    def train(self):
        return
    
    def get_codebook_detach(self, component_embeddings):
        component_objects = torch.zeros(self.num_component_object, self.num_channels).cuda()
        component_objects = component_objects + component_embeddings  # [N,C]
        component_objects = component_objects.unsqueeze(0)
        component_objects = component_objects.repeat(self.batch_size, 1, 1)  # [B N C]
        return component_objects.detach()

    def add_pixel_loss(self, out, target, self_infer):
        loss1 = F.l1_loss(out, target, reduction="mean") * self.cfg['pixel_w']
        loss2 = F.l1_loss(self_infer, target, reduction="mean") * self.cfg['pixel_w']
        self.g_losses['pixel'] = loss1 + loss2
        return loss1 + loss2

    def add_skeleton_loss(self, out, target):
        out_ske = skeletionize(out).cuda()
        target_ske = skeletionize(target).cuda()
        skeleton_loss = F.l1_loss(out_ske, target_ske, reduction = "mean") * self.cfg["skeleton_w"]
        self.g_losses['skeleton'] = skeleton_loss
        return skeleton_loss

    def add_edge_loss(self, out, target):
        out_edg = bianyuantiqu(out).cuda()
        target_edg = bianyuantiqu(target).cuda()
        edge_loss = F.l1_loss(out_edg, target_edg, reduction = "mean") * self.cfg["edge_w"]
        self.g_losses['edge'] = edge_loss
        return edge_loss

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss

    def take_contrastive_feature(self, input):
        # out = self.enc_style(input)
        out = self.projection_style(input)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def style_contrastive_loss(self, style_components_1, style_components_2, batch_size):
        _, N, _ = style_components_1.shape   
        style_contrastive_loss = 0  
        if self.cfg['contrastive_w'] == 0:   
            return style_contrastive_loss

        style_up = self.take_contrastive_feature(style_components_1)     
        style_down = self.take_contrastive_feature(style_components_2)

        for s in range(batch_size):
            if s == 0:
                negative_style_up = style_up[1:].transpose(0, 1)
                negative_style_down = style_down[1:].transpose(0, 1)

            if s == batch_size - 1:
                negative_style_up = style_up[:batch_size - 1].transpose(0, 1)
                negative_style_down = style_down[:batch_size - 1].transpose(0, 1)

            else:
                negative_style_up = torch.cat([style_up[0:s], style_up[s + 1:]], 0).transpose(0, 1)
                negative_style_down = torch.cat([style_down[0:s], style_down[s + 1:]], 0).transpose(0, 1)

            index_up = torch.LongTensor(random.sample(range(batch_size - 1), 5))
            index_down = torch.LongTensor(random.sample(range(batch_size - 1), 5))

            negative_style_up = torch.index_select(negative_style_up, 1, index_up.cuda())
            negative_style_down = torch.index_select(negative_style_down, 1, index_down.cuda())


            for i in range(N):
                style_comparisons_up = torch.cat([style_down[s][i:i + 1], negative_style_up[i]], 0)
                style_contrastive_loss += self.compute_contrastive_loss(style_up[s][i:i + 1],
                                                                        style_comparisons_up, 0.2, 0)
                style_comparisons_down = torch.cat([style_up[s][i:i + 1], negative_style_down[i]], 0)
                style_contrastive_loss += self.compute_contrastive_loss(style_down[s][i:i + 1],
                                                                        style_comparisons_down, 0.2, 0)

        style_contrastive_loss /= N  

        style_contrastive_loss *= self.cfg['contrastive_w']   
        self.g_losses['contrastive'] = style_contrastive_loss  

        return style_contrastive_loss


    def add_gan_g_loss(self, real_font, real_uni, fake_font, fake_uni):
        if self.cfg['gan_w'] == 0.:
            return 0.

        g_loss = -(fake_font.mean() + fake_uni.mean())
        g_loss *= self.cfg['gan_w']
        self.g_losses['gen'] = g_loss

        return g_loss

    def add_gan_d_loss(self, real_font, real_uni, fake_font, fake_uni):
        if self.cfg['gan_w'] == 0.:
            return 0.

        d_loss = (F.relu(1. - real_font).mean() + F.relu(1. + fake_font).mean()) + \
                 F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()

        d_loss *= self.cfg['gan_w']
        self.d_losses['disc'] = d_loss

        return d_loss

    def d_backward(self):
        with utils.temporary_freeze(self.gen):
            d_loss = sum(self.d_losses.values())
            d_loss.backward()

    def g_backward(self):
        """
        g_backward
        """
        with utils.temporary_freeze(self.disc):
            g_loss = sum(self.g_losses.values())
            g_loss.backward()

    def save(self, cur_loss, method, save_freq=None):
        if method not in ['all', 'last', 'all-last']:
            return

        step_save = False
        last_save = False
        if method == 'all' or (method == 'all-last' and self.step % save_freq == 0):
            step_save = True
        if method == 'last' or method == 'all-last':
            last_save = True
        assert step_save or last_save

        save_dic = {
            'generator': self.gen.state_dict(),
            'generator_ema': self.gen_ema.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'optimizer': self.g_optim.state_dict(),
            'epoch': self.step,
            'loss': cur_loss
        }

        if self.disc is not None:
            save_dic['discriminator'] = self.disc.state_dict()
            save_dic['d_optimizer'] = self.d_optim.state_dict()
            save_dic['d_scheduler'] = self.d_scheduler.state_dict()

        ckpt_dir = self.cfg['work_dir'] / "checkpoints" / self.cfg['unique_name']
        step_ckpt_name = "{:06d}-{}.pth".format(self.step, self.cfg['name'])
        last_ckpt_name = "last.pth"
        step_ckpt_path = Path.cwd() / ckpt_dir / step_ckpt_name
        last_ckpt_path = ckpt_dir / last_ckpt_name

        log = ""
        if step_save:
            torch.save(save_dic, str(step_ckpt_path))
            log = "Checkpoint is saved to {}".format(step_ckpt_path)

            if last_save:
                utils.rm(last_ckpt_path)
                last_ckpt_path.symlink_to(step_ckpt_path)
                log += " and symlink to {}".format(last_ckpt_path)

        if not step_save and last_save:
            utils.rm(last_ckpt_path) 
            torch.save(save_dic, str(last_ckpt_path))
            log = "Checkpoint is saved to {}".format(last_ckpt_path)

        self.logger.info("{}\n".format(log))

    def baseplot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,
            "train/skeleton_loss": losses.skeleton.val,
            "train/edge_loss": losses.edge.val
        }

        if self.disc is not None:
            tag_scalar_dic.update({
                'train/d_loss': losses.disc.val,
                'train/g_loss': losses.gen.val,
                'train/d_real_font': discs.real_font.val,
                'train/d_real_uni': discs.real_uni.val,
                'train/d_fake_font': discs.fake_font.val,
                'train/d_fake_uni': discs.fake_uni.val,
            })

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  Ske {L.skeleton.avg: 7.3f} Edg {L.edge.avg: 7.3f} D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  R_font {D.real_font_acc.avg:7.3f}  F_font {D.fake_font_acc.avg:7.3f}"
            "  R_uni {D.real_uni_acc.avg:7.3f}  F_uni {D.fake_uni_acc.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
