import cv2
import os
import re

def extract_number(filename : str):
    # 從檔名中提取數字
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def convert_frames_to_video(input_folder, output_file, fps):
    # 獲取所有影像檔案的檔名
    images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images = sorted(images, key = extract_number)
    print(images)
    # 獲取第一張影像的尺寸
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    # 定義影片編碼器和輸出影片檔案
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼器
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(input_folder, image))
        video.write(frame)

    video.release()

# 使用範例
input_folder = './enhance'  # 替換為你的影像幀資料夾路徑
output_file = 'output_video.mp4'  # 輸出影片檔案名稱
fps = 30  # 設定每秒幀數

convert_frames_to_video(input_folder, output_file, fps)
