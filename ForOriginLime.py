import glob
import numbers
import os
import cv2

import numpy as np

from lime import lime_enhancement

# from t import enhance_image
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    


def makeVideo(self, video_path, output_path):
    '''
        export a video by using image as input
    '''
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(30)
    for root, dir, files in os.walk(video_path):
        for file in files:
            picDir = os.path.join(root, file)
            img = cv2.imread(picDir)
            break
    h,w,c = img.shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # for root, dir, files in os.walk(video_path):
    #     for file in files:
    print(
        sorted(glob.glob(video_path+'/*.jpg'))
    )
    for file in sorted(glob.glob(video_path+'/*.jpg'),key=numericalSort):
        # picDir = os.path.join(root, file)
        print(file)
        img = cv2.imread(file)
        # result = lime_enhancement(img)
        out.write(img)
    
    out.release
            
    

def process_video(self, video_path, output_path):
    for root, dir, files in os.walk(video_path):
        for file in files:
            picDir = os.path.join(root, file)
            img = cv2.imread(picDir)
            # using lime to enhance image
            result = lime_enhancement(img)
            print(os.path.join(output_path, file))
            cv2.imwrite(os.path.join(output_path, file), result)
            


if __name__ == "__main__":
    
    video_path = "runs\detect\predict"  # Input video path
    output_path = ".\\highwayPre.mp4"  # Output video path
    
    makeVideo(video_path, output_path)
