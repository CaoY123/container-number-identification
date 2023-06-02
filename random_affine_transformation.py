from PIL import Image, ImageOps
import random

# 平移操作
def translate(image, x_shift, y_shift):
    translated_image = image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))
    return translated_image

# 旋转操作
def rotate(image, angle):
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    return rotated_image

# 缩放操作
def scale(image, scale_factor):
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    return scaled_image

# 错切操作
def shear(image, shear_factor):
    sheared_image = image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))
    return sheared_image

if __name__ == "__main__":
    # 读入原始图片
    original_image = Image.open("data_enhance/random_affine_transformation/IMG_0147.JPG")

    # 设置参数
    x_shift = 50
    y_shift = 50
    rotation_angle = 30
    scale_factor = 0.9
    shear_factor = 0.3

    # 进行平移操作并保存结果图像
    translated_image = translate(original_image, x_shift, y_shift)
    save_dir = 'data_enhance/random_affine_transformation/to/';
    translated_image.save(save_dir + "translated_image.jpg")

    # 进行旋转操作并保存结果图像
    rotated_image = rotate(original_image, rotation_angle)
    rotated_image.save(save_dir + "rotated_image.jpg")

    # 进行缩放操作并保存结果图像
    scaled_image = scale(original_image, scale_factor)
    scaled_image.save(save_dir + "scaled_image.jpg")

    # 进行错切操作并保存结果图像
    sheared_image = shear(original_image, shear_factor)
    sheared_image.save(save_dir + "sheared_image.jpg")
