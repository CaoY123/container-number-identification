import numpy as np
import os
import argparse

import cv2

def set_border_black(binary_img, border_size):
    height, width = binary_img.shape
    binary_img[:border_size, :] = 0
    binary_img[height-border_size:, :] = 0
    binary_img[:, :border_size] = 0
    binary_img[:, width-border_size:] = 0

def check_corner_pixels(img):
    height, width = img.shape
    corner_pixels = [
        img[0, 0],
        img[0, width - 1],
        img[height - 1, 0],
        img[height - 1, width - 1]
    ]
    black_count = sum([1 for pixel in corner_pixels if pixel == 0])
    return black_count

def run_pretreat(opt):
    pre_treat_images_dir = './binary_images'
    if not os.path.exists(pre_treat_images_dir):
        os.mkdir(pre_treat_images_dir)

    image_path = opt.source
    # 读取图像并转为灰度图
    img = cv2.imread(image_path)
    # 1. 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 对灰度图像进行双边滤波去除噪声
    bilateral_filtered_img = cv2.bilateralFilter(gray_img, 9, 200, 200)

    # 3. 使用自适应阈值处理
    adaptive_thresh = cv2.adaptiveThreshold(bilateral_filtered_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 9, 2)

    # 4. 设置边缘像素为黑色
    set_border_black(adaptive_thresh, border_size=5)

    # 5. 检查角落像素，如果大部分为黑色，则反转图像
    black_count = check_corner_pixels(adaptive_thresh)
    if black_count >= 3:
        adaptive_thresh = cv2.bitwise_not(adaptive_thresh)

    # 6. 中值滤波器去除椒盐噪声
    denoised_img = cv2.medianBlur(adaptive_thresh, 5)

    # 保存图像
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{pre_treat_images_dir}/{image_name}.jpg"
    writeResult = cv2.imwrite(filename, denoised_img)
    if writeResult:
        print(f"存储预处理后的图片【{filename}】存储成功！")
    else:
        print(f"存储预处理后的图片【{filename}】存储失败！")

    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cropped_images/IMG_0180_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_pretreat(opt)
