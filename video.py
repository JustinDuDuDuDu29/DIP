import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

        vidcap = cv2.VideoCapture('videoplayback.mp4')
        success,image = vidcap.read()
        count = 0
        while success:
                cv2.imwrite("./video/frame%d.jpg" % count, image)     # save frame as JPEG file      
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
        print("count num : ", count)


if __name__ == "__main__":
        main()
