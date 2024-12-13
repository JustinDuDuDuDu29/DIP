import cv2
import os
import re

def extract_number(filename : str):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def convert_frames_to_video(input_folder, output_file, fps):

    images = [img for img in os.listdir(input_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images = sorted(images, key = extract_number)
    # print(images)

    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(input_folder, image))
        video.write(frame)

    video.release()

input_folder = './enhance' 
output_file = 'output_video.mp4' 
fps = 30 

convert_frames_to_video(input_folder, output_file, fps)
