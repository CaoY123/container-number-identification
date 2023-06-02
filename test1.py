import cv2
import os

NORMAL_IMAGE_SIZE = (32, 32)

if __name__ == "__main__":
    # 打开图像文件
    binary_img_path = 'D:\\Study\\number-identification-system\\python\\container-number-identification\\generated_images\\O\\img025-00178.png'
    # 读入已经二值化后的图像
    binary_img = cv2.imread(binary_img_path, cv2.IMREAD_GRAYSCALE)

    # 归一化操作：
    normalized_image = cv2.resize(binary_img, NORMAL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    filename_without_ext = os.path.splitext(os.path.basename(binary_img_path))[0]
    # 保存带有检测框的图像
    output_path = filename_without_ext + '_with_boxes.jpg'
    cv2.imwrite(output_path, normalized_image)