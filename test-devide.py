import cv2
import os
import argparse
import numpy as np

def get_segmented_characters(img_path):
    # Load image
    img = cv2.imread(img_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    # Create a vertical kernel and apply morphological opening to remove vertical lines
    kernel = np.ones((5, 1), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find connected components and filter out components that are too small or too large
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity, cv2.CV_32S)
    filtered_stats = []
    for i, stat in enumerate(stats):
        if i == 0: # Skip background component
            continue
        x, y, w, h, area = stat
        if w < 10 or h < 10: # Filter out components that are too small
            continue
        if w > img.shape[1] / 2 or h > img.shape[0] / 2: # Filter out components that are too large
            continue
        filtered_stats.append(stat)

    # Sort components by x position
    filtered_stats = sorted(filtered_stats, key=lambda x: x[0])

    # Extract individual characters
    characters = []
    for i, stat in enumerate(filtered_stats):
        x, y, w, h, area = stat
        char = binary[y:y+h, x:x+w]

        # Resize character image to fixed size (32x32)
        char = cv2.resize(char, (32, 32), interpolation=cv2.INTER_AREA)

        # Remove white space borders around character
        non_zeros = cv2.findNonZero(char)
        x_min, y_min, x_max, y_max = cv2.boundingRect(non_zeros)
        char = char[y_min:y_max, x_min:x_max]

        characters.append(char)

    return characters

def run_devide(opt):
    single_dir = './singledigit';
    if not os.path.exists(single_dir):
        os.mkdir(single_dir)

    img_path = opt.source
    ima_name = os.path.splitext(os.path.basename(img_path))[0]

    save_dir = single_dir + '/' + ima_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    chars = get_segmented_characters(img_path)
    # 保存到save_dir中
    for i, char in enumerate(chars):
        char_name = os.path.join(save_dir, f"{i}.jpg")
        cv2.imwrite(char_name, char)
        print('第' + str(i) + '张图片保存完毕...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='cropped_images/IMG_0155_0.jpg', help='picture file')
    opt = parser.parse_args()
    run_devide(opt)
