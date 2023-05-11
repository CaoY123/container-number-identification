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
NORMAL_IMAGE_SIZE = (100, 100)

def close_char_edges(char_img):
    # 定义一个形态学操作的核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # 执行闭运算
    closed_char_img = cv2.morphologyEx(char_img, cv2.MORPH_CLOSE, kernel)

    return closed_char_img

def process_char_image(char_img):
    # 查找轮廓及其层次结构
    contours, hierarchy = cv2.findContours(char_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result_img = np.zeros_like(char_img)

    if hierarchy is not None and len(hierarchy) > 0:
        outer_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
        if len(outer_contours) > 0:
            # 找到最大的外部轮廓
            max_contour_index = max(range(len(outer_contours)), key=lambda i: cv2.contourArea(outer_contours[i]))
            max_contour = outer_contours[max_contour_index]

            # 将最大的外部轮廓填充为白色
            cv2.drawContours(result_img, [max_contour], 0, 255, -1)

            # 查找最大外部轮廓的子轮廓
            max_contour_global_index = contours.index(max_contour)
            children = np.where(hierarchy[0][:, 3] == max_contour_global_index)[0]

            # 将子轮廓填充为黑色
            if len(children) > 0:
                for child in children:
                    cv2.drawContours(result_img, contours, child, 0, -1)

                    # 查找子轮廓的子轮廓（内部轮廓）
                    grandchildren = np.where(hierarchy[0][:, 3] == child)[0]

                    # 将内部轮廓填充为白色
                    if len(grandchildren) > 0:
                        for grandchild in grandchildren:
                            cv2.drawContours(result_img, contours, grandchild, 255, -1)

            # 轻微膨胀，扩大黑色区域的面积
            kernel = np.ones((3, 3), np.uint8)
            char_img = cv2.dilate(result_img, kernel, iterations=1)

    return char_img


def filter_contours(contours, binary_img, min_aspect_ratio=0.3, max_aspect_ratio=3.0, max_width_ratio=1 / 7,
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

        # 执行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(char_img, connectivity=8)
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # 跳过背景
        largest_component = np.where(labels == largest_label)

        # 将不属于最大连通域且与最大连通域无交集的白色像素设为黑色
        for label in range(1, num_labels):
            if label == largest_label:
                continue

            label_coords = np.where(labels == label)
            if not np.any(labels[largest_component] == label):
                continue

            char_img[label_coords] = 0

        # 重新计算字符部分的边界
        white_pixel_coords = np.where(char_img == 255)
        x_min, x_max = np.min(white_pixel_coords[1]), np.max(white_pixel_coords[1])
        y_min, y_max = np.min(white_pixel_coords[0]), np.max(white_pixel_coords[0])

        # 裁剪多余的部分
        char_img = char_img[y_min:y_max + 1, x_min:x_max + 1]

        # 将图片调整为统一大小
        char_img = cv2.resize(char_img, NORMAL_IMAGE_SIZE)

        # 对字符图片进行形态学膨胀操作，填补字符上的黑色缺损
        # 定义一个形态学膨胀操作的核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        char_img = cv2.dilate(char_img, kernel)

        # 执行连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(char_img, connectivity=8)
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # 跳过背景
        largest_component = np.where(labels == largest_label)

        # 将不属于最大连通域且与最大连通域无交集的白色像素设为黑色
        for label in range(1, num_labels):
            if label == largest_label:
                continue

            label_coords = np.where(labels == label)
            if not np.any(labels[largest_component] == label):
                continue

            char_img[label_coords] = 0

        # char_img = process_char_image(char_img)
        char_img = close_char_edges(char_img)

        char_dir = os.path.join(output_dir, chr(97 + t))
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)

        file_path = os.path.join(char_dir, f"{os.path.splitext(os.path.basename(binary_img_path))[0]}{t}.jpg")
        t += 1

        # char_img = cv2.resize(char_img, NORMAL_IMAGE_SIZE)
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
    parser.add_argument('--source', type=str, default='binary_images/IMG_0155_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_devide(opt)
    sys.exit()