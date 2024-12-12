import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

        vidcap = cv2.VideoCapture('cartest.mp4')
        success,image = vidcap.read()
        # for i in range(100):
        count = 0
        while success:
        # print('./high_res/frame_%04d.png' % i)
                # image = cv2.imread('./high_res/frame_%04d.png' % i)
        # image = cv2.resize(image, (512, 512))
                cv2.imwrite("./cartest/frame%d.jpg" % count, image) 
                success,image = vidcap.read()      
                print('Read a new frame: ', success)
                count += 1
        print("done")


if __name__ == "__main__":
        main()
