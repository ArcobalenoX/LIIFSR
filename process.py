import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import Image, ImageDraw
import glob, os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import os
from PIL import Image, ImageDraw
import glob, os


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def crop_ROI():
    fin = '/home/guanwp/zhangwenlong/paperresult/ESRGAN/Set5_ESRGAN/'
    fout = fin + 'result' + '/'
    for file in os.listdir(fin):
        if is_image_file(file):
            file_fullname = os.path.join(fin, file)
            img = Image.open(file_fullname)

            a = [201, 112, 503, 188]
            box = (a)
            ROI = img.crop(box)

            out_path = fout + '/' + file
            ROI.save(out_path)


def cv2_crop():
    fin = '/home/guanwp/zhangwenlong/paperresult/SRGANx4_RANK_3style_V1_esrgan_pre_46d7/val_set14/'
    fout = 'result' + '/'
    for file in os.listdir(fin):
        if is_image_file(file):
            file_fullname = os.path.join(fin, file)
            img = Image.open(file_fullname)

            x = 216
            y = 266
            w = 206
            a = [x, y, x + w, y + w]
            box = (a)
            ROI = img.crop(box)
            # upscale

            out_path = os.path.join(fout, file)
            ROI.save(out_path)

            im = cv2.imread(file_fullname)
            cv2.rectangle(im, (x, y), (x + w, y + w), (0, 255, 0), 3)
            cv2.imshow("image", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            out_draw = os.path.join(fout, "rect/" + file)
            cv2.imwrite(out_draw, im)

def crop():
    dir = 'D:\Documents\截图/'  # 图片所在目录

    # 选框左上角坐标(x, y)，宽度width，高度自动计算得出
    x = 0
    y = 0
    width = 100

    # 获取图像
    pyFile = glob.glob(os.path.join(dir, "*.png"))
    pyFile += glob.glob(os.path.join(dir, "*.jpg"))
    pyFile += glob.glob(os.path.join(dir, "*.bmp"))
    result_path = os.path.join(dir, "result")

    # 判断是否存在result子目录，若不存在则创建
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # 遍历图片
    for img_path in pyFile:
        im = Image.open(img_path)
        draw = ImageDraw.Draw(im)

        aspect_ratio = im.size[0] / im.size[1]  # 长宽比
        # 截取选区图像
        im_ = im.crop((x, y, x + width, (x + width) // aspect_ratio))
        # 框出选区
        draw.rectangle((x, y, x + width, (x + width) // aspect_ratio), outline='red', width=3)  # width是线条的宽度

        im_ = im_.resize(im.size)  # 调用resize函数将子图放大到原图大小

        # 获取文件名
        _, img_name = os.path.split(img_path)
        img_name, _ = os.path.splitext(img_name)

        # 保存子图与含有选框的原图
        im_.save(os.path.join(result_path, img_name + '_sub_image.png'))
        im.save(os.path.join(result_path, img_name + '_ori_image.png'))


def scale_crop():
    dir = '*******************'  # the direction of the result

    # the(x, y), and the width
    x = 80
    y = 100
    width = 200
    scale = 4

    # capture an image
    pyFile = glob.glob(os.path.join(dir, "*.png"))
    pyFile += glob.glob(os.path.join(dir, "*.jpg"))
    pyFile += glob.glob(os.path.join(dir, "*.bmp"))
    result_path = os.path.join(dir, "result")

    # if the in result
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Traverse the picture
    for img_path in pyFile:
        im = Image.open(img_path)
        draw = ImageDraw.Draw(im)

        aspect_ratio = im.size[0] / im.size[1]  # Aspect ratio
        # Intercepting a selection image
        im_ = im.crop((x, y, x + width, (x + width) // aspect_ratio))
        # Box out of the selection
        draw.rectangle((x, y, x + width, (x + width) // aspect_ratio), outline='red', width=3)  # width是线条的宽度

        # im_ = im_.resize(im.size) # Call the resize function to enlarge the submap to the original image size
        width1 = int(im_.size[0] * scale)
        height1 = int(im_.size[1] * scale)
        im_ = im_.resize((width1, height1), Image.ANTIALIAS)

        # Get the file name
        _, img_name = os.path.split(img_path)
        img_name, _ = os.path.splitext(img_name)

        # Save submap and original image with marquee
        im_.save(os.path.join(result_path, img_name + '_sub_image.png'))
        im.save(os.path.join(result_path, img_name + '_ori_image.png'))

