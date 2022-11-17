import argparse
import os
import math
import numpy as np
import cv2
import glob
from skimage import transform
from skimage import measure
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
 
"""
热力图显示图像差异

""" 
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
 
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
 
def mse2psnr(mse):
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))
 
def plot_heatmap(image, heat_map, alpha=0.5, display=False, save=None, cmap='viridis', axis='on', 
                 dpi=80, verbose=False):
    height = image.shape[0]
    width = image.shape[1]
    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))
    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)
 
    if display:
        # display
        plt.imshow(image)
        plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
        plt.axis(axis)
        plt.show()
 
    if save is not None:
        if verbose:
            print('save image: ' + save)
            
        H, W, C = image.shape
        figsize = W / float(dpi), H / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        ax.imshow(image)
        ax.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
 
        ax.set(xlim=[0, W], ylim=[H, 0], aspect=1)
        fig.savefig(save, dpi=dpi, transparent=True)
        
def to_bin(img, lower, upper):
    return (lower < img) & (img < upper)
 
 
def plot_diff_map(im_BSL, im_OCT, im_GT, heatmap, thres=0.4, alpha=0.5, display=False,
                 save=None, cmap='viridis', axis='on', dpi=80, verbose=False):
    height, width, _ = im_BSL.shape
    # resize heat map
    heatmap_resized = transform.resize(heatmap, (height, width))
    # normalize heat map
    max_value = np.max(heatmap_resized)
    min_value = np.min(heatmap_resized)
    normalized_heatmap = (heatmap_resized - min_value) / (max_value - min_value)
    # capture regions
    bin_map = to_bin(normalized_heatmap, thres, 1.0)
    label_map = measure.label(bin_map, connectivity=2)
    props = measure.regionprops(label_map)
 
    plot_im = im_BSL.copy()
    plot_im[~bin_map] = 0
 
    if save is not None:
        if verbose:
            print('save image: ' + save)
            
        H, W, C = im_BSL.shape
        figsize = W / float(dpi), H / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        ax.imshow(im_BSL)
        ax.imshow(normalized_heatmap, alpha=alpha)
#         ax.imshow(plot_im, alpha=alpha)
        ax.axis(axis)
 
        for i in range(len(props)):
            if props[i].bbox_area >= 100:
                bbox_coord = props[i].bbox
                ax.add_patch(
                    patches.Rectangle(
                        (bbox_coord[1], bbox_coord[0]),
                        bbox_coord[3] - bbox_coord[1],
                        bbox_coord[2] - bbox_coord[0],
                        edgecolor='y',
                        linewidth = 6,
                        fill=False
                    ))
                psnr = calculate_psnr(im_OCT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255., \
                                      im_GT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255.) - \
                       calculate_psnr(im_BSL[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255., \
                                      im_GT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255.)
 
                h_aln = 'right' if W - bbox_coord[1] < 50 else 'left'
 
                if bbox_coord[0] < 20:
                    ax.text(bbox_coord[1], bbox_coord[2], "{:+.2f}".format(psnr), color='r', 
                            verticalalignment='top', horizontalalignment=h_aln, fontsize=26)
                else:
                    ax.text(bbox_coord[1], bbox_coord[0], "{:+.2f}".format(psnr), color='r',
                            verticalalignment='bottom', horizontalalignment=h_aln, fontsize=26)
        
        ax.set(xlim=[0, W], ylim=[H, 0], aspect=1)
        fig.savefig(save, dpi=dpi, transparent=True)
#     plt.show()
 
def plot_diff_patch(im_BSL, im_OCT, im_GT, heatmap, thres=0.4, alpha=0.5, display=False, 
                 save=None, cmap='viridis', axis='on', dpi=80, verbose=False):
    H, W, C = im_BSL.shape
    # resize heat map
    heatmap_resized = transform.resize(heatmap, (H, W))
    # normalize heat map
    max_value = np.max(heatmap_resized)
    min_value = np.min(heatmap_resized)
    normalized_heatmap = (heatmap_resized - min_value) / (max_value - min_value)
    # capture regions
    bin_map = to_bin(normalized_heatmap, 0.4, 1.0)
    label_map = measure.label(bin_map, connectivity=2)
    props = measure.regionprops(label_map)
    bbox_err = []
 
    for i in range(len(props)):
        if props[i].bbox_area >= 100:
            bbox_coord = props[i].bbox
            err = np.mean(normalized_heatmap[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3]])
            psnr = calculate_psnr(im_OCT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255., \
                                  im_GT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255.) - \
                   calculate_psnr(im_BSL[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255., \
                                  im_GT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :]*255.)
            bbox_err.append((i, err, psnr))
            
    bbox_err.sort(key=lambda x:x[1], reverse=True)
    im_diff = np.clip(im_OCT - im_BSL + 0.5, 0.0, 1.0)
    save_dir20= 'diff6_cvpr'
    save_path20 = os.path.join(save_dir20, base_name.replace('Layer_HRLRadd_SRResNet_16B64C_alpha=0.5', '')+'.png')
    im_diff20=im_diff*255
    cv2.imwrite(save_path20,im_diff20[:, :, [2, 1, 0]])
 
    num_bbox = min(len(bbox_err), 5)
    # Plot patches
    fig, axes = plt.subplots(nrows=num_bbox, ncols=4, figsize=(15,15))
    if axes.ndim == 1:
        axes = [axes]
    for i in range(num_bbox):
        ind, err, psnr = bbox_err[i]
        bbox_coord = props[ind].bbox
 
        axes[i][0].imshow(im_GT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :])
        axes[i][1].imshow(im_BSL[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :])
        axes[i][2].imshow(im_OCT[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :])
        axes[i][3].imshow(im_diff[bbox_coord[0]:bbox_coord[2], bbox_coord[1]:bbox_coord[3], :])

        axes[i][3].text(bbox_coord[3]-bbox_coord[1], bbox_coord[2]-bbox_coord[0], f"{psnr:+.2f}", color='r', fontsize=16)
        axes[i][3].text(bbox_coord[3]-bbox_coord[1], 0, f"{bbox_coord}", color='r', fontsize=16)
 
    fig.savefig(diff_patch_path, dpi=300, bbox_inches='tight', transparent=False)
    # plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--BSLDIR', default=r'testimg/WHURS19-DRSENMKCAx2')
    parser.add_argument('--OCTDIR', default=r'testimg/WHURS19-EDSRx2')
    parser.add_argument('--GTDIR', default=r'E:\Code\Python\datas\RS\WHU-RS19-test\HR\x2')
    args = parser.parse_args()


    #folder_BSL = r"E:\Code\Python\liif-self\testimg\DRSENMKCA-x2PNG"
    #folder_OCT = r"E:\Code\Python\liif-self\testimg\DRSENS-x2PNG"
    #folder_GT = r"E:\Code\Python\datas\RS\ITCVD_patch\ITCVD_test_patch"

    folder_BSL = args.BSLDIR
    folder_OCT = args.OCTDIR
    folder_GT = args.GTDIR

    diff_patch_dir = r'testimg/MKCA_EDSR_X2pathch'
    diff_map_dir = r'testimg/MKCA_EDSR_X2map'

    crop_border = 4
    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_OCT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')


    patch_size = 32
    stride = 10

    for img_path in img_list:
        print(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_OCT = cv2.imread(img_path)[:, :, [2, 1, 0]] / 255.
        #cv2.cvtColor(im_OCT, cv2.COLOR_BGR2RGB)
        im_BSL = cv2.imread(os.path.join(folder_BSL, base_name + '.png'))[:, :, [2, 1, 0]] / 255.
        im_GT = cv2.imread(os.path.join(folder_GT, base_name.replace('x2', '') + '.jpg'))[:, :, [2, 1, 0]] / 255.
        #im_GT = cv2.imread(os.path.join(folder_GT, base_name + '.png'))[:, :, [2, 1, 0]] / 255.

        H, W, C = im_OCT.shape
        H_axis = np.arange(0, H - patch_size, stride)
        W_axis = np.arange(0, W - patch_size, stride)
        err_map = np.zeros((len(H_axis), len(W_axis)))
        inv_map = np.zeros((len(H_axis), len(W_axis)))
        total_err = np.mean((im_OCT - im_BSL)**2)

        for i, h in enumerate(H_axis):
            for j, w in enumerate(W_axis):
                patch_OCT = im_OCT[h:h+patch_size, w:w+patch_size, :]
                patch_BSL = im_BSL[h:h+patch_size, w:w+patch_size, :]
                patch_err = np.sum((patch_OCT - patch_BSL)**2) / (H*W*C)

                err_map[i, j] = mse2psnr(patch_err)
                inv_map[i, j] = mse2psnr(total_err- patch_err)

        if not os.path.exists(diff_patch_dir):
            os.mkdir(diff_patch_dir)
        diff_patch_path = os.path.join(diff_patch_dir, base_name + '.png')

        if not os.path.exists(diff_map_dir):
            os.mkdir(diff_map_dir)
        diff_map_path = os.path.join(diff_map_dir, base_name + '.png')

        #plot_heatmap(im_BSL, inv_map, alpha=0.7, save=save_path, axis='off', display=False)
        plot_diff_map(im_BSL, im_OCT, im_GT, inv_map, alpha=0.5, save=diff_map_path, axis='off', display=False)
        plot_diff_patch(im_BSL, im_OCT, im_GT, inv_map, alpha=0.5, save=diff_patch_path, axis='off', display=False)
