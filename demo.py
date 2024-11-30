import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import argparse
import os

from exposure_enhancement import *

   
def gamma_correction(img: np.ndarray, gamma: int = 1):

        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        output_img = cv2.LUT(img, lookUpTable)
        return output_img



def erase_car_light(image1: np.ndarray, image2 : np.ndarray) -> np.ndarray:
        
        L0 = np.max(image1, axis = -1)
        L1 = np.max(image2, axis = -1)
        erase_image = L1 - L0

        _, binary_image = cv2.threshold(erase_image, 5, 255,  cv2.THRESH_BINARY_INV)
        
        cv2.imshow("origin coor", np.hstack([binary_image]))
        cv2.waitKey(0)

        contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # * 對車燈區域進行gamma值調整
        image1_copy = image1.copy()
        for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                car_light_region = image1[y:y + h, x:x + w]
                
                corrected_region = gamma_correction(car_light_region, gamma=5)

                image1_copy[y:y + h, x:x + w] = corrected_region


        cv2.imshow("origin coor", np.hstack([image1, image1_copy]))
        cv2.waitKey(0)

        return erase_image


def main(args):
        im_dir = args.folder
        sub_filename = ['jpg', 'bmp', 'png']
        file_path    = []
        for e in sub_filename:
                file_path.extend(glob.glob(im_dir + "*." + e))
        images_enhance = [cv2.imread(file) for file in file_path]

        # car_detection(images_enhance[0])

        erase_image  = erase_car_light(images_enhance[0], images_enhance[1])

        # cv2.imshow("ILL", erase_image)
        # cv2.waitKey(0)

        # for i, img in enumerate(images_enhance):
        #         # print(img.shape)
        #         #* execute lime 
        #         restore_img = enhance_image_exposure(img, args.gamma, args.lambda_, sigma=args.sigma ,eps=args.eps)
        #         cv2.imwrite(os.path.join("./enhance/" + f"{i}.png"), img = restore_img)
        #         cv2.imshow("i", restore_img)
        #         cv2.waitKey(0)
        
        
        

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="DIP final project")
        parser.add_argument("-f", '--folder', default = "./demo/enhance/", type = str, help = "the folder where your images store")
        parser.add_argument("-g", '--gamma', default= 0.6, type=float, help="the gamma correction parameter.")
        parser.add_argument("-s", '--sigma', default = 200, type = int, help = "kernel size")
        parser.add_argument("-l", '--lambda_', default= 0.15, type=float,
                            help="the weight for balancing the two terms in the illumination refinement optimization objective.")
        parser.add_argument("-eps", default=1e-3, type=float, help="constant to avoid computation instability.")
        # parser.add_argument
        args = parser.parse_args()

        main(args)