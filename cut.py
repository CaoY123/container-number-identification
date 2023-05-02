import cv2
import os
import argparse

def run_cut(opt):
    labelPath = opt.label

    # 读取txt文件
    with open(labelPath, "r") as f:
        lines = f.readlines()

    # 读取图像
    image_path = opt.source
    image = cv2.imread(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 创建一个文件夹用于保存裁剪后的目标部分
    if not os.path.exists("cropped_images"):
        os.mkdir("cropped_images")

    # 遍历每个目标
    for i, line in enumerate(lines):
        # 解析每个目标的位置信息
        class_id, x, y, width, height = line.split()
        x, y, width, height = float(x), float(y), float(width), float(height)

        # 计算目标区域的左上角和右下角坐标
        image_height, image_width, _ = image.shape
        x1 = int((x - width / 2) * image_width)
        y1 = int((y - height / 2) * image_height)
        x2 = int((x + width / 2) * image_width)
        y2 = int((y + height / 2) * image_height)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 裁剪目标区域
        cropped = image[y1:y2, x1:x2]

        # 保存裁剪后的目标部分到文件中
        filename = f"cropped_images/{image_name}_{i}.jpg"
        writeResult = cv2.imwrite(filename, cropped)

        if writeResult:
            print("写入" + filename + "成功")
        else:
            print("写入" + filename + "失败")

    return filename

# 裁剪已经标定的部分
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='runs/detect/exp13/IMG_0155.JPG',help='picture file')
    parser.add_argument('--label', type=str, default='runs/detect/exp13/labels/IMG_0155.txt', help='标定框坐标文件')
    opt = parser.parse_args()
    print(opt)

    run_cut(opt)

