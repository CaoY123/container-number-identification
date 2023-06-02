import cv2
import sys
import numpy as np

def mosaic_enhancement(image_paths):
    images = []
    max_height = 0
    max_width = 0

    # 读取图片并找到最大的高度和宽度
    for path in image_paths:
        image = cv2.imread(path)
        images.append(image)
        height, width, _ = image.shape
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    # 调整图片尺寸为最大尺寸
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (max_width, max_height))
        resized_images.append(resized_image)

    # 上下拼接前两张图片
    top = cv2.hconcat([resized_images[0], resized_images[1]])

    # 上下拼接后两张图片
    bottom = cv2.hconcat([resized_images[2], resized_images[3]])

    # 左右拼接上下拼接后的两张图片
    output = cv2.vconcat([top, bottom])

    return output

if __name__ == "__main__":
    # 输入四张图片的路径
    image_paths = ['data_enhance/mosaic/IMG_0147.JPG', 'data_enhance/mosaic/IMG_0150.JPG', 'data_enhance/mosaic/IMG_0168.JPG', 'data_enhance/mosaic/IMG_0368.JPG']

    # 进行马赛克增强处理
    output_image = mosaic_enhancement(image_paths)

    # 保存输出的图像
    cv2.imwrite('data_enhance/mosaic/to/output.jpg', output_image)
    print('===============================The end of mosaic procedure===============================')
    sys.exit()
