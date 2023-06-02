import cv2
import sys

if __name__ == "__main__":
    # 读取图像文件
    image = cv2.imread("data_enhance/blur/IMG_0168.JPG")

    # 进行模糊处理（增大高斯核的大小）
    blurred_image = cv2.GaussianBlur(image, (89, 89), 0)

    # 保存模糊处理后的图像
    cv2.imwrite("data_enhance/blur/to/blurred_image.jpg", blurred_image)
    print('===============================The end of blur procedure===============================')
    sys.exit()
