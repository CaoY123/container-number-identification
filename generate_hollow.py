import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageChops
import random
import shutil
import numpy as np
from config import base_dir
import cv2

# 生成用于训练模板匹配识别的图片

# create a list of characters to use in the images
characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# 设置最终保存图片的统一的大小：512 * 512像素
TARGET_IMAGE_SIZE = (256, 256)

# base_dir = os.path.dirname(os.path.abspath(__file__))

def generate_hollow_image(char, index, outline_width=5):
    # Set the image size and font
    image_size_height = 480
    image_size_width = 256
    image_size = (image_size_width, image_size_height)
    font_boldness = random.uniform(0.1, 1.5)

    # Create an image object and draw black character
    image = Image.new("1", image_size, 0)
    draw = ImageDraw.Draw(image)

    # Select a random font
    font_path = random.choice(font_paths)

    # Calculate the minimum font size to fill the image
    font_size = min(image_size_height, image_size_width)
    font = ImageFont.truetype(font_path, font_size)

    # Draw the character on the image with white color and boldness factor
    font_bold = ImageFont.truetype(font_path, font_size)
    font_bold.boldness = int(font_size * font_boldness)

    draw.text((0, 0), char, font=font_bold, fill=1)

    # Convert the PIL image to a NumPy array
    img_array = np.array(image, dtype=np.uint8) * 255

    # Apply morphological operations
    kernel = np.ones((outline_width, outline_width), np.uint8)
    erosion = cv2.erode(img_array, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    hollow_image = cv2.subtract(dilation, erosion)

    # Convert the NumPy array back to a PIL image
    hollow_image_pil = Image.fromarray(hollow_image)

    # 加入随机噪声
    # for i in range(int(0.005 * image_size_height * image_size_width)):
    #     x = random.randint(0, image_size_width - 1)
    #     y = random.randint(0, image_size_height - 1)
    #     if random.random() > 0.5:
    #         hollow_image_pil.putpixel((x, y), 0)
    #     else:
    #         hollow_image_pil.putpixel((x, y), 1)

    print('字符: ' + char + '序号: 【' + str(index) + '】图片生成...')
    return hollow_image_pil

fonts_dir = os.path.join(base_dir, "fonts")
font_paths = []
# 遍历fonts文件夹
for root, dirs, files in os.walk(fonts_dir):
    for file in files:
        # 检查文件是否为.ttf字体文件
        if file.endswith(".ttf") or file.endswith(".TTF"):
            # 将字体文件路径添加到font_paths列表中
            font_paths.append(os.path.join(root, file))

# set the output directory and number of images to generate
output_dir = output_dir = os.path.join(base_dir, 'generated_images/')
num_images = len(font_paths)

# create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# for i in range(num_images):
#     # randomly select a character from the list
#     char = random.choice(characters)
#
#     img = generate_image2(char, i)
#     img.save(output_dir + str(i) + '_' + char + '.png')

# 设置边框粗细范围
outline_width_range = [6, 10]

for i in range(0, len(characters)):
    char = characters[i]
    save_dir = output_dir + char + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     shutil.rmtree(save_dir)
    #     os.makedirs(save_dir)

    for j in range(num_images):
        # 为每个字体生成 3 种不同边框粗细的图片
        outline_widths = random.sample(outline_width_range, min(num_images, 2))
        for k in range(2):
            # 为每个字体生成 min(num_images, 4) 种不同边框粗细的图片
            outline_width = outline_widths[k]
            # 生成空心字符
            index = (j + 1) * 1000 + k
            img = generate_hollow_image(char, index, outline_width)

            # 将PIL.Image.Image对象转换为NumPy数组
            img_np = np.array(img.convert('L'))

            # 归一化操作
            img_np[img_np != 0] = 1

            # 统计每行水平方向上的白色像素数
            row_white_pixel_count = np.sum(img_np == 1, axis=1)

            # 统计每列垂直方向上的白色像素数
            col_white_pixel_count = np.sum(img_np == 1, axis=0)

            rows_greater_than_15 = np.where(row_white_pixel_count > 15)[0]
            rows_greater_than_0 = np.where(row_white_pixel_count > 0)[0]

            if rows_greater_than_15.size > 0 and rows_greater_than_0.size > 0:
                char_height = rows_greater_than_15[-1] - rows_greater_than_0[0] + 1
            else:
                # 处理异常情况，例如使用默认值
                continue

            # 获取字符的高度和宽度
            # char_height = np.where(row_white_pixel_count > 15)[0][-1] - np.where(row_white_pixel_count > 0)[0][0] + 1

            cols_greater_than_15 = np.where(col_white_pixel_count > 15)[0]
            cols_greater_than_0 = np.where(col_white_pixel_count > 0)[0]

            if cols_greater_than_15.size > 0 and cols_greater_than_0.size > 0:
                char_width = cols_greater_than_15[-1] - cols_greater_than_0[0] + 1
            else:
                # 处理异常情况，例如使用默认值
                continue
            # char_width = np.where(col_white_pixel_count > 15)[0][-1] - np.where(col_white_pixel_count > 0)[0][0] + 1

            # 裁剪出字符部分
            left = np.where(col_white_pixel_count > 0)[0][0]
            upper = np.where(row_white_pixel_count > 0)[0][0]
            right = np.where(col_white_pixel_count > 0)[0][-1] + 1
            lower = np.where(row_white_pixel_count > 0)[0][-1] + 1

            img_cropped = img.crop((left, upper, right, lower))

            # 进行图片大小的调整
            img_resized = img_cropped.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)

            img_resized.save(save_dir + char + '_' + str(index) + '.png')

print("==============================the end of procedure==============================")
sys.exit()
