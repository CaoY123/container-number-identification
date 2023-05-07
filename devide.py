import sys
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import shutil

plt.style.use('seaborn')

# 对预处理后的二值图片进行字符的分割
# 标准化后的每个字符图片的大小：
NORMAL_IMAGE_SIZE = (62, 64)

def filter_contours(contours, binary_img, min_aspect_ratio=0.3, max_aspect_ratio=3.0, max_width_ratio=1 / 10,
                    max_height_ratio=0.88, min_width_ratio=1 / 50, min_height_ratio=1 / 10):
    filtered_contours = []
    img_height, img_width = binary_img.shape

    max_width = img_width * max_width_ratio
    max_height = img_height * max_height_ratio
    min_width = img_width * min_width_ratio
    min_height = img_height * min_height_ratio

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # 根据宽高比和宽高像素数量过滤轮廓
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and min_width <= w <= max_width and min_height <= h <= max_height:
            filtered_contours.append(contour)

    return filtered_contours

def find_contours(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]))
    return sorted_contours

import cv2

# def find_contours(binary_img, height_threshold=0.9, width_threshold=0.2):
#     contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     refined_contours = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#
#         img_height, img_width = binary_img.shape[:2]
#
#         if (h / img_height > height_threshold) or (w / img_width > width_threshold):
#             # 在大轮廓中寻找小轮廓
#             roi = binary_img[y:y + h, x:x + w]
#             small_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#             for small_contour in small_contours:
#                 refined_contours.append(small_contour)
#         else:
#             refined_contours.append(contour)
#
#     sorted_contours = sorted(refined_contours, key=lambda ctr: (cv2.boundingRect(ctr)[0], cv2.boundingRect(ctr)[1]))
#     return sorted_contours

def white_pixel_ratio(img):
    white_pixels = np.sum(img == 255)
    total_pixels = img.size
    return white_pixels / total_pixels

def save_chars(binary_img, contours, output_dir, binary_img_path):
    t = 0

    # 计算每个轮廓的白色像素占比和白色像素数
    contour_ratios_and_counts = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_img = binary_img[y:y + h, x:x + w].copy()
        ratio = white_pixel_ratio(char_img)
        white_pixel_count = np.sum(char_img == 255)
        contour_ratios_and_counts.append((contour, ratio, white_pixel_count, x, y))

    # 按白色像素数从高到低，白色像素占比从高到低对轮廓进行排序
    sorted_contours = sorted(contour_ratios_and_counts, key=lambda x: (x[2], x[1]), reverse=True)

    # 保留前11个轮廓（如果存在的话）
    top_contours = sorted_contours[:11]

    # 按照x坐标从左到右，y坐标从上到下再排序
    top_contours = sorted(top_contours, key=lambda x: (x[3], x[4]))

    for contour, ratio, _, x, y in top_contours:
        w, h = cv2.boundingRect(contour)[2:]

        char_img = binary_img[y:y + h, x:x + w].copy()

        char_img = cv2.resize(char_img, NORMAL_IMAGE_SIZE)

        char_dir = os.path.join(output_dir, chr(97 + t))
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)

        file_path = os.path.join(char_dir, f"{os.path.splitext(os.path.basename(binary_img_path))[0]}{t}.jpg")
        t += 1

        cv2.imwrite(file_path, char_img)
        print(f"保存分割后的图片【{file_path}】成功！")

def process_image(binary_img_path):
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)
    filename_without_ext = os.path.splitext(os.path.basename(binary_img_path))[0]

    output_dir = f'./singledigit/{filename_without_ext}'

    # 删除现有的 output_dir，然后重新创建它
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    contours = find_contours(binary_img)
    # 过滤轮廓
    filtered_contours = filter_contours(contours, binary_img)
    save_chars(binary_img, filtered_contours, output_dir, binary_img_path)

    return output_dir

def run_devide(opt):
    single_image_dir = process_image(opt.source)
    print('===============================初步切割结束===============================')
    return single_image_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='binary_images/IMG_0164_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_devide(opt)
    sys.exit()