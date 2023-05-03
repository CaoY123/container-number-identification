import sys

import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# 对预处理后的二值图片进行字符的分割

# 标准化后的每个字符图片的大小：
NORMAL_IMAGE_SIZE = (32, 32)

def visualize_projection(projection):
    plt.plot(projection)
    plt.xlabel("Index")
    plt.ylabel("Projection Value")
    plt.show()

# 计算图像在指定轴上的投影。它接收一个二值化的图像和一个轴参数。返回在指定轴上的投影。
# 当axis=0时，计算垂直投影；当axis=1时，计算水平投影。
def projection(img, axis=0):
    return np.sum(img > 128, axis=axis)

# 找到投影中的谷值，即字符间的空隙。这个函数接受投影数据，以及最小宽度和最小深度作为参数。
# 最小宽度和最小深度用于筛选谷值，确保找到的谷值是字符之间的有效空隙。
def find_valleys(projection, min_width=5, min_depth=15):
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


# 根据谷值分割图像。这个函数接收一个图像和谷值列表，然后根据谷值将图像分割成多个部分。
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



# flag为0，进行初次切割，flag为1，对单个字符进一步进行处理
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
        for char_img in chars:
            # 保存分割后的字符图片
            tmp_dir_name = dir_name + '/' + chr(97 + t)

            if not os.path.exists(tmp_dir_name):
                os.mkdir(tmp_dir_name)

            filePath = tmp_dir_name + '/' + filename_without_ext + str(t) + '.jpg'
            t = t + 1

            if not char_img is None:
                # 保存前将图片的分辨率重置为20 * 20
                # char_img_r = cv2.resize(char_img, (20, 20))
                writeResult = cv2.imwrite(filePath, char_img)
                if writeResult:
                    print('保存分割后的图片【' + filePath + '】成功！')
                else:
                    print('保存分割后的图片【' + filePath + '】失败！')

        return dir_name

def narmalize_process(single_image_dir):
    for image_dir in os.listdir(single_image_dir):
        image_dir = single_image_dir + '/' + image_dir + '/'
        for image_path in os.listdir(image_dir):
            image_path = image_dir + image_path
            if image_path.endswith('.jpg'):
                image_path = str(image_path)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # 调整图像大小为 NORMAL_IMAGE_SIZE(x * y)
                # img = cv2.resize(img, NORMAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

                # 归一化操作
                # 统计每行水平方向上的白色像素数
                row_white_pixel_count = np.sum(img == 255, axis=1)

                # 统计每列垂直方向上的白色像素数
                col_white_pixel_count = np.sum(img == 255, axis=0)

                # 获取字符的高度和宽度
                # char_height = np.where(row_white_pixel_count > 15)[0][-1] - np.where(row_white_pixel_count > 0)[0][0] + 1
                # char_width = np.where(col_white_pixel_count > 15)[0][-1] - np.where(col_white_pixel_count > 0)[0][0] + 1

                # 裁剪出字符部分
                img_cropped = img[
                              np.where(row_white_pixel_count > 15)[0][0]:np.where(row_white_pixel_count > 0)[0][-1] + 1,
                              np.where(col_white_pixel_count > 15)[0][0]:np.where(col_white_pixel_count > 0)[0][-1] + 1]

                img = cv2.resize(img_cropped, NORMAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(image_path, img)

def run_devide(opt):
    single_image_dir = process_image(opt)
    print('===============================初步切割结束===============================')
    narmalize_process(single_image_dir)
    print('===============================尺寸归一化结束===============================')
    return single_image_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='binary_images/IMG_0155_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_devide(opt)
    sys.exit()
