import cv2
import numpy as np
import os
import argparse

import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # 1. 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 边缘检测
    edges = cv2.Canny(gray_img, 100, 200)

    # 3. 图像腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_img = cv2.erode(edges, kernel)

    # 4. 平滑处理（形态学闭运算）
    closing = cv2.morphologyEx(eroded_img, cv2.MORPH_CLOSE, kernel)

    # 5. 移除小对象
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50  # 设置最小面积阈值
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(closing, [cnt], 0, 0, -1)

    return closing


def run_pretreat(opt):
    pre_treat_images_dir = './binary_images'
    if not os.path.exists(pre_treat_images_dir):
        os.mkdir(pre_treat_images_dir)

    image_path = opt.source
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行高斯模糊
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 使用Sobel算子在x和y方向上计算梯度
    sobel_x = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度的绝对值
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)

    # 将梯度值转换为8位无符号整数（0-255）
    scaled_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    scaled_sobel_y = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))

    # 使用阈值处理梯度值，只保留较强的边缘
    thresh_min = 50
    thresh_max = 200
    binary_output_x = np.zeros_like(scaled_sobel_x)
    binary_output_y = np.zeros_like(scaled_sobel_y)
    binary_output_x[(scaled_sobel_x >= thresh_min) & (scaled_sobel_x <= thresh_max)] = 255
    binary_output_y[(scaled_sobel_y >= thresh_min) & (scaled_sobel_y <= thresh_max)] = 255

    # 将x和y方向的边缘检测结果组合起来
    combined = cv2.bitwise_or(binary_output_x, binary_output_y)

    # 3. 图像腐蚀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # eroded_img = cv2.erode(edges, kernel)

    # 4. 平滑处理（形态学闭运算）
    # closing = cv2.morphologyEx(eroded_img, cv2.MORPH_CLOSE, kernel)

    # 5. 移除小对象
    # contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min_area = 50  # 设置最小面积阈值
    # for cnt in contours:
    #     if cv2.contourArea(cnt) < min_area:
    #         cv2.drawContours(closing, [cnt], 0, 0, -1)

    # 保存图像
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{pre_treat_images_dir}/{image_name}.jpg"
    writeResult = cv2.imwrite(filename, combined)
    if writeResult:
        print(f"存储预处理后的图片【{filename}】存储成功！")
    else:
        print(f"存储预处理后的图片【{filename}】存储失败！")

    return filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cropped_images/IMG_0164_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_pretreat(opt)
