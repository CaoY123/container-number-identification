import random

import cv2
import os
import argparse
import numpy as np

# 预处理脚本

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

if __name__ == '__main__':

    pre_treat_images_dir = './binary_images';
    if not os.path.exists(pre_treat_images_dir):
        os.mkdir(pre_treat_images_dir)

    # 读取解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='picture file')
    opt = parser.parse_args()

    # 读取彩色图像
    image_path = opt.source
    img = cv2.imread(image_path)

    # 将彩色图像转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对比度拉伸
    stretched_img = contrast_stretching(gray_img)

    # 3. 高斯滤波
    gaussian_blur_img = cv2.GaussianBlur(stretched_img, (3, 3), 0)

    # 4. 使用 Otsu's 二值化 - 这里的二值化很容易出意外，应当根据不同的图像特征选择合适的二值化策略
    ret, binary_img1 = cv2.threshold(gaussian_blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, binary_img2 = cv2.threshold(gaussian_blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. 选择平均像素值较低的图像
    if np.mean(binary_img1) < np.mean(binary_img2):
        binary_img = binary_img1
    else:
        binary_img = binary_img2

    # 自适应二值化
    # block_size = 11  # 必须是奇数
    # C = 2
    # binary_img = cv2.adaptiveThreshold(gaussian_blur_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    # 6. 对二值图像执行开运算
    kernel = np.ones((1, 1), np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 7. 应用中值滤波器进一步减少椒盐噪声
    # ksize = 17
    # final_img = cv2.medianBlur(opened_img, ksize)

    img = opened_img
    width, height = img.shape

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



    # 7. 应用中值滤波器进一步减少椒盐噪声
    # final_img = cv2.medianBlur(opened_image, ksize)

    # 使用自适应阈值方法进行二值化
    # binary_img = cv2.adaptiveThreshold(equalized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 进行腐蚀操作
    # eroded_image = cv2.erode(final_img, kernel, iterations=iterations)

    # # # 进行膨胀操作
    # dilated_image = cv2.dilate(eroded_image, kernel, iterations=iterations)

    # # # 进行闭操作
    # closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # img = closed_image

    # img = cv2.dilate(img, kernel, iterations=iterations)

    # img = cv2.erode(img, kernel, iterations=iterations)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{pre_treat_images_dir}/{image_name}.jpg"
    writeResult = cv2.imwrite(filename, binary_img)
    if writeResult:
        print("存储预处理后的图片【" + filename + "】存储成功！")
    else:
        print("存储预处理后的图片【" + filename + "】存储失败！")
