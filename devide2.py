import sys
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import shutil
plt.style.use('seaborn')

# 投影法分割字符
# 标准化后的每个字符图片的大小：
NORMAL_IMAGE_SIZE = (32, 32)

def visualize_projection(projection):
    plt.plot(projection)
    plt.xlabel("索引")
    plt.ylabel("投影值")
    plt.show()


def projection(img, axis=0):
    return np.sum(img > 128, axis=axis)


def find_valleys(projection, min_width=5, min_depth=5):
    valleys = []
    width = 0
    start_idx = 0
    for idx, value in enumerate(projection):
        if value < min_depth:
            if width == 0:
                start_idx = idx
            width += 1
        else:
            if width >= min_width:
                valleys.append((start_idx, idx))
            width = 0

    # 添加最后一个区间
    valleys.append((start_idx, len(projection)))

    return valleys


def split_lines(image, valleys, flag=0):
    lines = []
    if len(valleys) > 1:
        for i in range(len(valleys) - 1):
            start = valleys[i][1]
            end = valleys[i + 1][0]
            if flag == 0:
                lines.append(image[:, start:end])
            else:
                lines.append(image[start:end, :])
    return lines


def process_image(opt):
    binary_img_path = opt.source
    # 读入已经二值化后的图像
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)

    # 获取不包含扩展名的图像名，以作为后面建文件夹所用
    filename_without_ext = os.path.splitext(os.path.basename(binary_img_path))[0]

    # 数组vertical_projection、horizontal_projection的每个像素值为沿鸽子方向上的白色像素的数量
    # 计算垂直投影，是一个一维数组，其反映了从上到下像素分布的情况，大小为从左到右的像素数
    vertical_projection = projection(binary_img, axis=0)
    # 计算水平投影，是一个一维数组，其反映了从左到右像素分布的情况，大小为从上到下的像素数
    horizontal_projection = projection(binary_img, axis=1)

    # 水平方向上字符的间隔数组，是一个元祖对数组，每一个元祖的第一个数表示间隔的开始，第二个数表示间隔的结束
    vertical_valleys = find_valleys(vertical_projection)
    horizontal_valleys = find_valleys(horizontal_projection)

    # 确保存储这张图片分割结果的文件夹已经建立
    dir_name = './singledigit/' + filename_without_ext
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)

    # 多行字符情况
    # 这里的lines每行图片的数组
    lines = split_lines(binary_img, horizontal_valleys, 1)
    t = 0
    for line in lines:
        # 对每行进行垂直投影
        line_projection = projection(line, axis=0)
        # 计算每行的垂直谷值
        line_vertical_valleys = find_valleys(line_projection)
        # 使用垂直谷值将字符分割
        chars = split_lines(line, line_vertical_valleys)

        if len(chars) > 1:
            # 表示字符排列在两行上
            # 对每个字符进行进一步处理
            for char_img in chars:
                # 检查字符的高度和宽度是否在指定的百分比范围内
                min_height_percentage = 0.1  # 最小高度百分比
                max_height_percentage = 0.9  # 最大高度百分比
                min_width_percentage = 0.01  # 最小宽度百分比
                max_width_percentage = 0.1  # 最大宽度百分比

                height, width = char_img.shape
                height_percentage = height / binary_img.shape[0]
                width_percentage = width / binary_img.shape[1]

                # 检查字符的高度和宽度是否在指定的百分比范围内
                if (min_height_percentage <= height_percentage <= max_height_percentage) and \
                        (min_width_percentage <= width_percentage <= max_width_percentage):
                    # 保存分割后的字符图片
                    tmp_dir_name = dir_name + '/' + chr(97 + t)

                    if not os.path.exists(tmp_dir_name):
                        os.mkdir(tmp_dir_name)

                    filePath = tmp_dir_name + '/' + filename_without_ext + str(t) + '.jpg'
                    t = t + 1

                    if not char_img is None:
                        threshold = 128  # 设置阈值

                        # 找到大于等于阈值的像素的索引
                        y_indices, x_indices = np.where(char_img >= threshold)
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)

                        # 裁剪图像，并处理黑色区域的边界
                        cropped_image = char_img[max(0, y_min - 1):min(y_max + 2, char_img.shape[0]),
                                        max(0, x_min - 1):min(x_max + 2, char_img.shape[1])]

                        # 归一化操作：
                        normalized_image = cv2.resize(cropped_image, NORMAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                        writeResult = cv2.imwrite(filePath, normalized_image)
                        if writeResult:
                            print('保存分割后的图片【' + filePath + '】成功！')
                        else:
                            print('保存分割后的图片【' + filePath + '】失败！')
        else:
            # 表示字符排列在一列上
            for char_img in chars:
                # 检查字符的高度和宽度是否在指定的百分比范围内
                min_height_percentage = 0  # 最小高度百分比
                max_height_percentage = 1  # 最大高度百分比
                min_width_percentage = 0.05  # 最小宽度百分比
                max_width_percentage = 0.95  # 最大宽度百分比

                height, width = char_img.shape
                height_percentage = height / binary_img.shape[0]
                width_percentage = width / binary_img.shape[1]

                # 检查字符的高度和宽度是否在指定的百分比范围内
                if (min_height_percentage <= height_percentage <= max_height_percentage) and \
                        (min_width_percentage <= width_percentage <= max_width_percentage):
                    # 保存分割后的字符图片
                    tmp_dir_name = dir_name + '/' + chr(97 + t)

                    if not os.path.exists(tmp_dir_name):
                        os.mkdir(tmp_dir_name)

                    filePath = tmp_dir_name + '/' + filename_without_ext + str(t) + '.jpg'
                    t = t + 1

                    if not char_img is None:
                        threshold = 128  # 设置阈值

                        # 找到大于等于阈值的像素的索引
                        y_indices, x_indices = np.where(char_img >= threshold)
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)

                        # 裁剪图像，并处理黑色区域的边界
                        cropped_image = char_img[max(0, y_min - 1):min(y_max + 2, char_img.shape[0]),
                                        max(0, x_min - 1):min(x_max + 2, char_img.shape[1])]

                        # 归一化操作：
                        normalized_image = cv2.resize(cropped_image, NORMAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                        writeResult = cv2.imwrite(filePath, normalized_image)
                        if writeResult:
                            print('保存分割后的图片【' + filePath + '】成功！')
                        else:
                            print('保存分割后的图片【' + filePath + '】失败！')



def run_devide(opt):
    single_image_dir = process_image(opt)
    print('===============================字符分割结束===============================')
    return single_image_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='binary_images/meitu_00032_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_devide(opt)
    sys.exit()