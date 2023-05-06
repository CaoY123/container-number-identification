import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageChops
import random
import shutil
import numpy as np
from config import base_dir
from scipy import ndimage
import cv2

# 生成用于训练模板匹配识别的图片

# create a list of characters to use in the images
characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# 设置最终保存图片的统一的大小：512 * 512像素
TARGET_IMAGE_SIZE = (512, 512)

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

    print('图片编号: ' + str(index) + ': 图片名: 【' + str(index) + '_' + char + '.png】, 字符字重: 【' + str(font_boldness) + '】')
    return hollow_image_pil

def generate_image3(char, index):
    # Set the image size and font
    image_size_height = 480
    image_size_width = 256
    image_size = (image_size_width, image_size_height)

    # font_paths = ["arial.ttf", "times.ttf", "verdana.ttf"]
    font_size = 500
    font_boldness = random.uniform(0.1, 1.5)

    # Create an image object and draw black character
    image = Image.new("1", image_size, 0)
    draw = ImageDraw.Draw(image)

    # Select a random font
    font_path = random.choice(font_paths)
    font = ImageFont.truetype(font_path, font_size)

    # Get the size of the character
    char_bbox = font.getbbox(char)

    # Calculate the position to draw the character
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]

    # Calculate the x and y offsets necessary to center the character
    x_offset = (image_size[0] - char_width) // 2
    y_offset = (image_size[1] - char_height) // 2

    # Calculate the left and top coordinates
    left = x_offset - char_width
    top = y_offset - char_height

    # Calculate the right and bottom coordinates
    right = x_offset + char_width
    bottom = y_offset + char_height

    # Calculate the maximum x and y offsets for the random positioning
    max_x_offset = right - char_width
    max_y_offset = bottom - char_height

    # Generate a random position within the maximum offsets, and add the centering offsets
    char_position = (random.randint(left, max_x_offset), random.randint(top, max_y_offset))

    # Check if the character position is within the image bounds, and adjust if necessary
    if char_position[0] < 0:
        char_position = (0, char_position[1])
    elif char_position[0] + char_width > image_size[0]:
        char_position = (image_size[0] - char_width, char_position[1])
    if char_position[1] < 0:
        char_position = (char_position[0], 0)
    elif char_position[1] + char_height > image_size[1]:
        char_position = (char_position[0], image_size[1] - char_height)

    # Draw the character on the image with white color and boldness factor
    font_bold = ImageFont.truetype(font_path, font_size)
    font_bold.boldness = int(font_size * font_boldness)
    draw.text(char_position, char, font=font_bold, fill=1)

    # Print the image information
    print('第' + str(index) + '张图片: 图片名: 【' + str(index) + '_' + char + '.png】, 字符宽: 【' + str(char_width) + '】, 字符高: 【' + str(char_height) + '】, 字符位置: 【' + str(char_position) + '】, 字符字重: 【' + str(font_boldness) + '】')
    return image

def generate_image4(char, index):
    # Set the image size and font
    image_size_height = 480
    image_size_width = 256
    image_size = (image_size_width, image_size_height)
    # font_paths = ["arial.ttf", "times.ttf", "verdana.ttf"]
    font_boldness = random.uniform(0.1, 1.5)

    # Create an image object and draw black character
    image = Image.new("1", image_size, 0)
    draw = ImageDraw.Draw(image)

    # Select a random font
    font_path = random.choice(font_paths)

    # Calculate the minimum font size to fill the image
    font_size = min(image_size_height, image_size_width)
    font = ImageFont.truetype(font_path, font_size)

    # Get the size of the character
    char_bbox = font.getbbox(char)

    # Calculate the position to draw the character
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]

    # Calculate the x and y offsets necessary to center the character
    x_offset = (image_size[0] - char_width) // 2
    y_offset = (image_size[1] - char_height) // 2

    # Set the character position to the calculated offsets
    char_position = (x_offset, y_offset)

    # Draw the character on the image with white color and boldness factor
    font_bold = ImageFont.truetype(font_path, font_size)
    font_bold.boldness = int(font_size * font_boldness)
    draw.text(char_position, char, font=font_bold, fill=1)

    # Print the image information
    print('第' + str(index) + '张图片: 图片名: 【' + str(index) + '_' + char + '.png】, 字符宽: 【' + str(char_width) + '】, 字符高: 【' + str(char_height) + '】, 字符位置: 【' + str(char_position) + '】, 字符字重: 【' + str(font_boldness) + '】')
    return image



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

for i in range(0, len(characters)):
    char = characters[i]
    save_dir = output_dir + char + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     shutil.rmtree(save_dir)
    #     os.makedirs(save_dir)

    for j in range(num_images):
        # 生成实心字符
        index = j
        img = generate_image4(char, j)
        # 生成空心字符(每次生成一种，需要的时候把下面两句放开，上面两句关闭)
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
