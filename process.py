from PIL import Image, ImageDraw
import cv2
import os
import glob
from shutil import copyfile
import lpips
import os
import numpy as np


def cal_lpips(hr_img_path, sr_img_path):
    # Can also set net = 'squeeze' or 'vgg'
    loss_fn = lpips.LPIPS(net='alex').cuda()
    hr_path_list = []
    sr_path_list = []

    for root, _, fnames in sorted(os.walk(hr_img_path, followlinks=True)):
        for fname in fnames:
            path = os.path.join(hr_img_path, fname)
            hr_path_list.append(path)

    for root, _, fnames in sorted(os.walk(sr_img_path, followlinks=True)):
        for fname in fnames:
            path = os.path.join(sr_img_path, fname)
            sr_path_list.append(path)
    dist_ = []
    for i in range(len(hr_path_list)):
        hr_img = lpips.im2tensor(lpips.load_image(hr_path_list[i])).cuda()
        sr_img = lpips.im2tensor(lpips.load_image(sr_path_list[i])).cuda()
        dist = loss_fn.forward(hr_img, sr_img)
        dist_.append(dist.mean().item())

    print('Avarage Distances: %.3f' % (sum(dist_)/len(hr_path_list)))
    return np.mean(dist_)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def copy_img(src_dir, meth, dst_dir, img_name):
    src = os.path.join(src_dir, img_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst = os.path.join(dst_dir, meth+'_'+img_name)
    copyfile(src, dst)


def copy_imgs(img_name):
    img_path = img_name + '.png'
    dst_dir = os.path.join(r'AID/visualcmp', img_name)
    copy_img(r'data\selfAID\AID-test', 'GT', dst_dir, img_name + '.jpg')
    copy_img(r'AID\AID_bicubicx4', 'bicubic', dst_dir, img_path)
    copy_img(r'AID\WHURS19_carnx4', 'carn', dst_dir, img_path)
    copy_img(r'AID\WHURS19_drsensx4', 'drsen', dst_dir, img_path)
    copy_img(r'AID\WHURS19_edsrblx4', 'edsr', dst_dir, img_path)
    copy_img(r'AID\WHURS19_L0Sgradx4_high', 'l0', dst_dir, img_path)
    copy_img(r'AID\WHURS19_lgcnetx4', 'lgcnet', dst_dir, img_path)
    copy_img(r'AID\WHURS19_panx4', 'pan', dst_dir, img_path)
    copy_img(r'AID\WHURS19_spsr_l1x4', 'spsr-l1', dst_dir, img_path)
    copy_img(r'AID\WHURS19_spsr_percx4', 'spsr-prec', dst_dir, img_path)
    copy_img(r'AID\WHURS19_spsrpa_charedgessimx4', 'sppa', dst_dir, img_path)
    copy_img(r'AID\WHURS19_vapsr_l1x4', 'vapsr', dst_dir, img_path)
    copy_img(r'AID\WHURS19_rcanx4', 'rcan', dst_dir, img_path)
    copy_img(r'AID\WHURS19_rdnx4', 'rdn', dst_dir, img_path)


def crop_ROI(dir,):
    fout = dir + 'result' + '/'
    for file in os.listdir(dir):
        if is_image_file(file):
            file_fullname = os.path.join(dir, file)
            img = Image.open(file_fullname)
            a = [201, 112, 503, 188]
            box = (a)
            ROI = img.crop(box)
            out_path = fout + '/' + file
            ROI.save(out_path)


def cv2_crop(in_dir, out_dir, x=0, y=0, w=100):
    for file in os.listdir(in_dir):
        if is_image_file(file):
            file_fullname = os.path.join(in_dir, file)
            img = Image.open(file_fullname)
            a = [x, y, x + w, y + w]
            box = (a)
            ROI = img.crop(box)

            # upscale

            out_path = os.path.join(out_dir, file)
            ROI.save(out_path)

            im = cv2.imread(file_fullname)
            cv2.rectangle(im, (x, y), (x + w, y + w), (0, 255, 0), 3)
            cv2.imshow("image", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            out_draw = os.path.join(out_dir, "rect/" + file)
            cv2.imwrite(out_draw, im)


def scale_crop(dir, result_dir, x=80, y=100, width=200, scale=3):
    # capture an image
    pyFile = glob.glob(os.path.join(dir, "*.png"))
    pyFile += glob.glob(os.path.join(dir, "*.jpg"))
    # pyFile += glob.glob(os.path.join(dir, "*.bmp"))

    sign_dir = os.path.join(result_dir, "sign")
    sub_dir = os.path.join(result_dir, "sub")
    if not os.path.exists(sign_dir):
        # os.makedirs(result_dir)
        os.makedirs(sign_dir)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Traverse the picture
    for img_path in pyFile:
        im = Image.open(img_path)
        draw = ImageDraw.Draw(im)

        aspect_ratio = im.size[0] / im.size[1]  # Aspect ratio
        # Intercepting a selection image
        im_ = im.crop((x, y, x + width, (y + width) // aspect_ratio))
        # Box out of the selection
        draw.rectangle((x, y, x + width, (y + width) // aspect_ratio),
                       outline='red', width=3)  # width是线条的宽度

        if scale != 1:
            # im_ = im_.resize(im.size) # Call the resize function to enlarge the submap to the original image size
            width1 = int(im_.size[0] * scale)
            height1 = int(im_.size[1] * scale)
            im_ = im_.resize((width1, height1), Image.BICUBIC)

        # Get the file name
        # img_name = os.path.basename(img_path)
        _, img_name = os.path.split(img_path)
        img_name, _ = os.path.splitext(img_name)

        # Save submap and original image with marquee
        im_.save(os.path.join(sub_dir, img_name + '_sub.png'))
        im.save(os.path.join(sign_dir, img_name + '_sign.png'))


if __name__ == '__main__':

    # for i in os.listdir(r"AID\visualcmp"):
    #     print(i)
    cmpimg = "baseballfield_75"
    copy_imgs(cmpimg)
    imgs_dir = os.path.join(r"AID\visualcmp",cmpimg)
    scale_crop(imgs_dir, imgs_dir, x=400, y=200, width=200, scale=3)
