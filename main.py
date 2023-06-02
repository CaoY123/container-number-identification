import sys
import argparse
import glob
import os
import detect
import cut
import pretreat
import devide
import recognition

def get_image_file(files_list):
    for file in files_list:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return file
    return None

def main(opt):
    # 1. 标注图片
    # 创建一个 Namespace 对象，并设置属性
    detect_opt = argparse.Namespace()
    detect_opt.weights = 'best.pt'
    detect_opt.source = opt.source
    detect_opt.img_size = 640
    detect_opt.conf_thres = 0.25
    detect_opt.iou_thres = 0.45
    detect_opt.device = '0'
    detect_opt.view_img = False
    detect_opt.save_txt = False
    detect_opt.save_conf = False
    detect_opt.nosave = False
    detect_opt.classes = None
    detect_opt.agnostic_nms = False
    detect_opt.augment = False
    detect_opt.update = False
    detect_opt.project = 'runs/detect'
    detect_opt.name = 'exp'
    detect_opt.exist_ok = False
    detect_opt.no_trace = False

    detected_image_file, detect_save_dir = detect.run_detection(detect_opt)
    print(f"Result save directory from caller: {detect_save_dir}")

    # ****************************************************************
    # 2. 裁剪已经标定的图片
    # 读取文件夹下的所有文件
    all_files = glob.glob(os.path.join(detect_save_dir, '*'))
    # 获取文件夹下的唯一图片文件
    # detected_image_file = get_image_file(all_files)
    image_file_path = ''
    if detected_image_file:
        image_file_path = os.path.relpath(os.path.join(detected_image_file), os.getcwd())
        print(f"Image file path: {image_file_path}")
    else:
        print("================================No image files found=============================")
        sys.exit()

    # 获取名为 "labels" 的子文件夹下的标签文件（假设标签文件的扩展名是 .txt）
    labels_folder_path = os.path.join(detect_save_dir, 'labels')
    label_files = glob.glob(os.path.join(labels_folder_path, '*.txt'))

    # 从列表中获取第一个标签文件的路径（相对路径）
    label_file_path = ''
    if label_files:
        label_file_path = os.path.relpath(label_files[0], os.getcwd())
        print(f"Label file path: {label_file_path}")
    else:
        print("================================No label files found=============================")
        sys.exit()

    cut_opt = argparse.Namespace()
    cut_opt.source = image_file_path
    cut_opt.label = label_file_path
    cut_filename = cut.run_cut(cut_opt)
    print('cut_filename: ' + cut_filename)
    # ****************************************************************
    # 3. 对裁剪的图片进行预处理
    pretreat_opt = argparse.Namespace()
    pretreat_opt.source = cut_filename
    pretreated_filepath = pretreat.run_pretreat(pretreat_opt)
    print("pretreated_filepath: " + pretreated_filepath)

    # ****************************************************************
    # 4. 对预处理后的图片进行分割并归一化
    devide_opt = argparse.Namespace()
    devide_opt.source = pretreated_filepath
    devided_file_dir = devide.run_devide(devide_opt)
    print("devided_file_dir: " + devided_file_dir)

    # ****************************************************************
    # 5. 对分割且归一化后的图片进行识别
    recognition_opt = argparse.Namespace()
    recognition_opt.source = devided_file_dir
    result_str = recognition.run_recognition(recognition_opt)
    print("RESULT:" + result_str )
    return result_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='datasets/test/IMG_20160611_145519.jpg', help='picture file')
    opt = parser.parse_args()
    main(opt)
    sys.exit()
