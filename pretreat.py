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
    gray_img = contrast_stretching(gray_img)
    std_dev = np.std(gray_img)  # 像素标准差
    i = 0
    while std_dev < 60:
        c = 255 / np.log(1 + np.max(gray_img))
        log_transformed = c * np.log(1 + gray_img)
        log_transformed = np.array(log_transformed, dtype=np.uint8)
        gray_img = log_transformed  # 更新图像
        std_dev = np.std(gray_img)  # 像素标准差
        print(f'After stretch #{i + 1}, the standard deviation is: {std_dev}')
        i = i + 1

    # 结果图像
    stretched_img = gray_img

    ret, binary_img = cv2.threshold(stretched_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 分别对对黑白像素的数量计数
    white_pixels = np.sum(binary_img == 255)
    black_pixels = np.sum(binary_img == 0)

    # 如果是白色像素多（认为是白色背景），则要进行字符像素值的反转
    if white_pixels > black_pixels:
        binary_img = cv2.bitwise_not(binary_img)

    # 7. 去除外边缘的白色框
    # height, width = binary_img.shape
    # set_border_black(img, border_size=int(height * 0.05))

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{pre_treat_images_dir}/{image_name}.jpg"
    writeResult = cv2.imwrite(filename, binary_img)
    if writeResult:
        print("存储预处理后的图片【" + filename + "】存储成功！")
    else:
        print("存储预处理后的图片【" + filename + "】存储失败！")

    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cropped_images/IMG_0170_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_pretreat(opt)
