from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import os
import glob

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

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

    sign_dir = os.path.join(result_dir, "_sign")
    sub_dir = os.path.join(result_dir, "_sub")
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
        draw.rectangle((x, y, x + width, (y + width) // aspect_ratio), outline='red', width=3)  # width是线条的宽度

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
        im_.save(os.path.join(sub_dir, os.path.basename(dir) + img_name + '_sub.png'))
        im.save(os.path.join(sign_dir, os.path.basename(dir) + img_name + '_sign.png'))

if __name__ == '__main__':
    # print(os.path.basename('result/crop/WHURS19_edsrblx2'))
    scale_crop("result/WHURS19_lgcnetx4", "result/crop/WHURS19_lgcnetx4", x=300, y=300, width=200, scale=1)


