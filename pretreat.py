import numpy as np
import os
import argparse
import cv2

# 检查分辨率并调整
def resize_image(img, max_dim=2300):
    height, width = img.shape[:2]
    max_original_dim = max(height, width)
    if max_original_dim < max_dim:
        scale = max_dim / max_original_dim
        new_height, new_width = int(height * scale), int(width * scale)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Apply sharpening filter
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened_img = cv2.filter2D(resized_img, -1, sharpening_kernel)
        return sharpened_img

    return img

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

def contour_repair(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    repaired_img = binary_img.copy()

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(repaired_img, [approx], 0, 255, 2)

    return repaired_img

def run_pretreat(opt):
    pre_treat_images_dir = './binary_images'
    if not os.path.exists(pre_treat_images_dir):
        os.mkdir(pre_treat_images_dir)

    image_path = opt.source
    # 读取图像并转为灰度图
    img = cv2.imread(image_path)

    # 检查分辨率并调整
    img = resize_image(img)

    # 1. 转换为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 使用高斯滤波器进行平滑
    smoothed_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])
    smoothed_img = cv2.filter2D(smoothed_img, -1, kernel)
    # 2. 对灰度图像进行双边滤波去除噪声
    bilateral_filtered_img = cv2.bilateralFilter(smoothed_img, 11, 800, 800)

    # 4. 使用自适应阈值处理进行局部二值化
    adaptive_thresh = cv2.adaptiveThreshold(bilateral_filtered_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, blockSize=11, C=2)

    # 4. 设置边缘像素为黑色
    set_border_black(adaptive_thresh, border_size=5)

    # # 5. 检查角落像素，如果大部分为黑色，则反转图像
    black_count = check_corner_pixels(adaptive_thresh)
    if black_count >= 3:
        adaptive_thresh = cv2.bitwise_not(adaptive_thresh)

    # 6. 中值滤波器去除椒盐噪声
    denoised_img = cv2.medianBlur(adaptive_thresh, 5)

    # # 6. 使用非局部均值去噪
    denoised_img = cv2.fastNlMeansDenoising(denoised_img, h=30, templateWindowSize=7, searchWindowSize=21)

    # 7. 去除外边缘的白色框
    set_border_black(denoised_img, border_size=5)

    # 8. 使用中值滤波器进一步去除盐噪声
    denoised_img = cv2.medianBlur(denoised_img, 5)

    closing_kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(denoised_img, cv2.MORPH_CLOSE, closing_kernel)

    # 9. 修复边框
    repaired_img = contour_repair(closed_img)

    # 膨胀操作
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(repaired_img, dilation_kernel, iterations=2)

    # 10. 保存图像
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"{pre_treat_images_dir}/{image_name}.jpg"
    writeResult = cv2.imwrite(filename, dilated_img)
    if writeResult:
        print(f"存储预处理后的图片【{filename}】存储成功！")
    else:
        print(f"存储预处理后的图片【{filename}】存储失败！")

    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cropped_images/IMG_0157_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_pretreat(opt)