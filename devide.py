import sys
from PIL import Image
import shutil
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# 对预处理后的二值图片进行字符的分割

# 标准化后的每个字符图片的大小：
NORMAL_IMAGE_SIZE = (32, 32)

def split_characters(binary_img, min_width_ratio=0.01, min_height_ratio=0.1, max_width_ratio=0.15, max_height_ratio=0.9, row_threshold_ratio=0.2):
    # 使用连通组件分析获取字符区域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=4)

    # 获取图像的宽度和高度
    img_height, img_width = binary_img.shape

    if img_width > img_height:
        # 计算宽度和高度的阈值
        min_width = int(min_width_ratio * img_width)
        min_height = int(min_height_ratio * img_height)
        max_width = int(max_width_ratio * img_width)
        max_height = int(max_height_ratio * img_height)

        # 计算行间距阈值
        row_threshold = int(row_threshold_ratio * img_height)

        # 获取字符区域的宽度和高度
        char_widths = stats[:, cv2.CC_STAT_WIDTH]
        char_heights = stats[:, cv2.CC_STAT_HEIGHT]

        # 获取字符区域的位置、宽度和高度信息
        char_data = []
        for i, (width, height) in enumerate(zip(char_widths, char_heights)):
            # 过滤不符合宽度和高度要求的字符
            if min_width <= width <= max_width and min_height <= height <= max_height:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                char_data.append((x, y, w, h))

        # Sort char_data by y coordinate
        char_data.sort(key=lambda c: c[1])

        # Group char_data by row
        lines = [[char_data[0]]]
        for char in char_data[1:]:
            # If char is close to the last char in the current line, append it to the current line
            if abs(char[1] - lines[-1][-1][1]) < row_threshold:
                lines[-1].append(char)
            else:  # Otherwise, start a new line
                lines.append([char])
    else:
        # 表明有可能是单列情况
        # 计算宽度和高度的阈值
        min_width = int(min_height_ratio * img_width)
        min_height = int(min_width_ratio * img_height)
        max_width = int(max_height_ratio * img_width)
        max_height = int(max_width_ratio * img_height)

        # 获取字符区域的宽度和高度
        char_widths = stats[:, cv2.CC_STAT_WIDTH]
        char_heights = stats[:, cv2.CC_STAT_HEIGHT]

        # 获取字符区域的位置、宽度和高度信息
        char_data = []
        for i, (width, height) in enumerate(zip(char_widths, char_heights)):
            # 过滤不符合宽度和高度要求的字符
            if min_width <= width <= max_width and min_height <= height <= max_height:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                char_data.append((x, y, w, h))

        # 对单列的字符，按照纵坐标从上到下进行排序
        char_data.sort(key=lambda c: c[1])

        # Group char_data by row
        lines = [[char_data[0]]]
        for char in char_data[1:]:
            lines.append([char])

    # Sort each line by x coordinate
    for line in lines:
        line.sort(key=lambda c: c[0])

    # Flatten lines into a single list of chars
    char_data = [char for line in lines for char in line]

    # 裁剪字符图像
    characters = []
    for x, y, w, h in char_data:
        character_image = binary_img[y:y+h, x:x+w]
        characters.append(character_image)

    return characters

def process_image(opt):
    # 打开图像文件
    binary_img_path = opt.source
    # 读入已经二值化后的图像
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)

    # 获取不包含扩展名的图像名，以作为后面建文件夹所用
    filename_without_ext = os.path.splitext(os.path.basename(binary_img_path))[0]

    single_save_dir = './singledigit'
    ing_save_dir = single_save_dir + '/' + filename_without_ext
    if not os.path.exists(ing_save_dir):
        os.mkdir(ing_save_dir)
    else:
        shutil.rmtree(ing_save_dir)
        os.makedirs(ing_save_dir)

    filtered_images = split_characters(binary_img)

    # 选取前11个最大的图像
    selected_images = filtered_images[:11]

    images = selected_images

    # 保存选取的图像到单独的文件
    single_image_dir = []
    for i, image in enumerate(images):
        tmp_save_dir = ing_save_dir + '/' + chr(97 + i)
        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)
        filePath = tmp_save_dir + '/' + filename_without_ext + str(i) + '.jpg'

        threshold = 10  # set your threshold here

        # find the indices where the image intensity is greater than or equal to the threshold
        y_indices, x_indices = np.where(image >= threshold)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # 裁剪图像
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

        # 归一化操作：
        normalized_image = cv2.resize(cropped_image, NORMAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        writeResult = cv2.imwrite(filePath, normalized_image)
        if writeResult:
            print('保存分割后的图片【' + filePath + '】成功！')
        else:
            print('保存分割后的图片【' + filePath + '】失败！')

    # 返回单个字符图像的文件路径列表
    return single_image_dir

def run_devide(opt):
    single_image_dir = process_image(opt)
    print('===============================初步切割结束===============================')
    return single_image_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='binary_images/IMG_0162_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_devide(opt)
    sys.exit()