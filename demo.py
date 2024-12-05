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


def erase_car_light(image1: np.ndarray, image2: np.ndarray, gamma:int = 20) -> np.ndarray:
    L0 = np.max(image1, axis=-1)
    L1 = np.max(image2, axis=-1)
    erase_image = L1 - L0

    _, binary_image = cv2.threshold(erase_image, 5, 255, cv2.THRESH_BINARY)
    _, binary_image_2 = cv2.threshold(erase_image, 5, 255, cv2.THRESH_BINARY_INV)
    black_background = np.zeros_like(image2)

    # 所有車燈以外的區域
    non_mask_image = cv2.bitwise_or(image2, black_background, mask=binary_image)

    contours, _ = cv2.findContours(binary_image_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     resize_bg = np.zeros_like(image2)
#     for con in contours:
#         t = cv2.approxPolyDP(con, 10, True)
#         center, radius = cv2.minEnclosingCircle(t)
#         # print("radius",radius)
#         resize_bg = cv2.circle(resize_bg, center=(int(center[0]), int(center[1])), radius=int(radius) + 2, color=(255, 255, 255), thickness=-1)

#     resize_bg = cv2.cvtColor(resize_bg, cv2.COLOR_BGR2GRAY)
#     _, binary_resize_bg = cv2.threshold(resize_bg, 254, 255, cv2.THRESH_BINARY)
#     _, binary_resize_bg_2 = cv2.threshold(resize_bg, 254, 255, cv2.THRESH_BINARY_INV)
#     non_mask_image = cv2.bitwise_or(image2, black_background, mask=binary_resize_bg_2)

    contours_2, _ = cv2.findContours(binary_image_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image1_copy = image1.copy()
    for contour in contours_2:
        black_background_2 = np.zeros_like(image1_copy)
        t = cv2.approxPolyDP(contour, 10, True)

        center2, radius2 = cv2.minEnclosingCircle(t)
        # print("radius2",radius2)

        # 車燈為白的圖片
        binary_mask_img = cv2.circle(black_background_2, center=(int(center2[0]), int(center2[1])), radius=int(radius2), color=(255, 255, 255), thickness=-1)
        # 轉成binary image
        gray_bmi = cv2.cvtColor(binary_mask_img, cv2.COLOR_BGR2GRAY)
        # 車燈的部分為1其餘為0(*255才看的到)
        _, binary_slimg = cv2.threshold(gray_bmi, 200, 1, cv2.THRESH_BINARY)
        car_light_img = cv2.bitwise_and(image1_copy, image1_copy, mask=binary_slimg)
        corrected_region = gamma_correction(car_light_img, gamma)
        non_mask_image = cv2.bitwise_or(non_mask_image, corrected_region)

    cv2.imwrite("./erase_car_light_2.png", non_mask_image)


    return non_mask_image


def main(args):
        imo_dir = args.folderde
        ime_dir = args.folderen

        sub_filename    = ['jpg', 'bmp', 'png']
        file_path_o     = []
        file_path_en    = []

        for e in sub_filename:
                file_path_o.extend(glob.glob(imo_dir + "*." + e))
                file_path_en.extend(glob.glob(ime_dir + "*." + e))
                
        images_original = [cv2.imread(fo) for fo in file_path_o]
        images_enhance = [cv2.imread(fe) for fe in file_path_en]

        for i in range(len(images_original)):
                # print(img.shape)
                #* execute lime 
                restore_img = erase_car_light(images_original[i], images_enhance[i], args.gamma)
                cv2.imwrite(os.path.join("./enhance/" + f"{i}.png"), img = restore_img)
                cv2.imshow("i", restore_img)
                cv2.waitKey(0)
        
        
        

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="DIP final project")
        parser.add_argument("-fd", '--folder_de', default = "./demo/default", type = str, help = "the folder where your original images store")
        parser.add_argument("-fe", '--folder_en', default = "./demo/enhance/", type = str, help = "the folder where your enhance images store")
       
        parser.add_argument("-g", '--gamma', default= 0.6, type=float, help="the gamma correction parameter.")
        # parser.add_argument("-s", '--sigma', default = 200, type = int, help = "kernel size")
        # parser.add_argument("-l", '--lambda_', default= 0.15, type=float,
                        #     help="the weight for balancing the two terms in the illumination refinement optimization objective.")
        # parser.add_argument("-eps", default=1e-3, type=float, help="constant to avoid computation instability.")
        # parser.add_argument
        args = parser.parse_args()

        main(args)