import random

import cv2
import os
import argparse
import numpy as np

# 预处理脚本

def set_border_black(binary_img, border_size):
    height, width = binary_img.shape
    binary_img[:border_size, :] = 0
    binary_img[height-border_size:, :] = 0
    binary_img[:, :border_size] = 0
    binary_img[:, width-border_size:] = 0

def contrast_stretching(img, low_percent=1, high_percent=99):
    """
    对比度拉伸
    :param img: 输入图像
    :param low_percent: 低百分位数
    :param high_percent: 高百分位数
    :return: 输出图像
    """
    low_val, high_val = np.percentile(img, [low_percent, high_percent])
    img_stretched = (img - low_val) / (high_val - low_val) * 255
    img_stretched[img_stretched < 0] = 0
    img_stretched[img_stretched > 255] = 255
    return img_stretched.astype(np.uint8)

def run_pretreat(opt):
    pre_treat_images_dir = './binary_images';
    if not os.path.exists(pre_treat_images_dir):
        os.mkdir(pre_treat_images_dir)

    # 读取彩色图像
    image_path = opt.source
    img = cv2.imread(image_path)

    width, height = img.shape[:2]

    maxNum = width
    minNum = height
    if width < height:
        maxNum = height
        minNum = width

    if maxNum < 1000:
        rate = maxNum * 1.0 / minNum
        randNum = random.randint(100, 300) + 1000
        if width > height:
            img = cv2.resize(img, (int(randNum / rate), randNum))
        else:
            img = cv2.resize(img, (randNum, int(randNum / rate)))

    # 将彩色图像转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对比度拉伸
    stretched_img = contrast_stretching(gray_img)
    # equalized_img = cv2.equalizeHist(stretched_img)

    # 3. 高斯滤波
    # gaussian_blur_img = cv2.GaussianBlur(stretched_img, (3, 3), 0)

    # 双边滤波
    # filtered_img = cv2.bilateralFilter(equalized_img, 9, 75, 75)

    binary_img = cv2.adaptiveThreshold(stretched_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    # 6. 对二值图像执行开运算
    kernel = np.ones((3, 3), np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 5. 对二值图像执行闭运算，填补间断的白线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)

    # 统计二值化图像中黑白像素点的数量比例
    black_pixels = np.sum(closed_img == 0)
    white_pixels = np.sum(closed_img == 255)
    ratio = black_pixels / white_pixels

    # 7. 根据比例确定二值化后的前景和背景颜色
    if ratio > 0.5:
        foreground_color = 0
        background_color = 255
    else:
        foreground_color = 255
        background_color = 0

    img = closed_img

    # 7. 去除外边缘的白色框
    height, width = binary_img.shape
    set_border_black(img, border_size=int(height * 0.05))

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{pre_treat_images_dir}/{image_name}.jpg"
    writeResult = cv2.imwrite(filename, img)
    if writeResult:
        print("存储预处理后的图片【" + filename + "】存储成功！")
    else:
        print("存储预处理后的图片【" + filename + "】存储失败！")

    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cropped_images/IMG_0164_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_pretreat(opt)
