import os
from PIL import Image
import PIL.ImageOps

# 定义源文件夹和目标文件夹
src_folder = 'D:\\学习相关\\学习重要文件\\毕业设计相关\\识别数据集\\English\\Fnt'
dest_folder = 'D:\\学习相关\\学习重要文件\\毕业设计相关\\识别数据集\\English\\CHANGED'

# 获取源文件夹中的所有子文件夹
subfolders = [f.name for f in os.scandir(src_folder) if f.is_dir()]

# 遍历子文件夹
for subfolder in subfolders:
    # 创建对应的目标子文件夹
    os.makedirs(os.path.join(dest_folder, subfolder), exist_ok=True)

    # 获取子文件夹中的所有图片
    images = [f for f in os.scandir(os.path.join(src_folder, subfolder)) if
              f.is_file() and f.name.endswith(('.png', '.jpg', '.jpeg'))]

    # 遍历图片
    for image in images:
        # 打开图片
        img = Image.open(image.path)

        # 反转图片
        inverted_image = PIL.ImageOps.invert(img)

        # 扩展白色字符的区域到图片的边界
        bbox = inverted_image.getbbox()
        cropped_image = inverted_image.crop(bbox)
        # resized_image = cropped_image.resize((128, 128))

        # 保存反转后的图片到目标文件夹
        save_path = os.path.join(dest_folder, subfolder, image.name)
        cropped_image.save(save_path)

        # 打印保存信息
        print(f"Image {image.name} has been inverted and saved to {save_path}")
