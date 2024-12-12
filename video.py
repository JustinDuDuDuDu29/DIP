import cv2

def main():

        vidcap = cv2.VideoCapture('cartest.mp4')
        success,image = vidcap.read()
        count = 0
        while success:
                cv2.imwrite("./cartest/frame%d.jpg" % count, image) 
                success,image = vidcap.read()      
                print('Read a new frame: ', success)
                count += 1
        print("done")

if __name__ == "__main__":
        main()
