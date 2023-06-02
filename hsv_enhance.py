import cv2
import numpy as np
import sys


def enhance_hsv(image, hue_factor, saturation_factor, value_factor):
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 分离HSV通道
    h, s, v = cv2.split(hsv_image)

    # 调整色调
    h = np.clip(h * hue_factor, 0, 255).astype(np.uint8)

    # 调整饱和度
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

    # 调整亮度
    v = np.clip(v * value_factor, 0, 255).astype(np.uint8)

    # 合并调整后的通道
    enhanced_hsv_image = cv2.merge([h, s, v])

    # 将图像转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image


if __name__ == "__main__":
    # 读取图像文件
    image = cv2.imread("data_enhance/hsv_enhance/IMG_0147.JPG")

    # 设置增强系数（增大系数以使变化更明显）
    hue_factor = 1.5  # 色调增强系数
    saturation_factor = 2.0  # 饱和度增强系数
    value_factor = 1.2  # 亮度增强系数

    # 进行HSV颜色空间增强
    enhanced_image = enhance_hsv(image, hue_factor, saturation_factor, value_factor)

    # 保存增强后的图像
    cv2.imwrite("data_enhance/hsv_enhance/to/enhanced_image.jpg", enhanced_image)
    print('===============================The end of hsv enhance procedure===============================')
    sys.exit()
