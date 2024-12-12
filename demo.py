import numpy as np
import cv2
import re
import glob
import argparse
import os

def gamma_correction(img: np.ndarray, gamma: int = 1):
    '''
    implement the gamma correction

    Parameters
    ----------
    img : input image
    gamma : pixel values to the power of gamma
    '''
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    output_img = cv2.LUT(img, lookUpTable)
    return output_img

def erase_car_light(image1: np.ndarray, image2: np.ndarray, gamma:int = 20, light_threshold:int = 5) -> np.ndarray:
    '''
    Dream to circle the car light area and using gamma correction to adjust

    Parameters
    ----------
    image1 : the original image
    image2 : the image that be processed by lime algorithm
    gamma  : the gamma correction parameter
    light_threshold : the value of the car light threshold
    '''
    L0 = np.max(image1, axis=-1)
    L1 = np.max(image2, axis=-1)
    erase_image = L1 - L0

    _, binary_image = cv2.threshold(erase_image, light_threshold, 255, cv2.THRESH_BINARY)
    _, binary_image_2 = cv2.threshold(erase_image, light_threshold, 255, cv2.THRESH_BINARY_INV)
    black_background = np.zeros_like(image2)

    #*  the image that doesn't contain the car light
    non_mask_image = cv2.bitwise_or(image2, black_background, mask=binary_image)

    contours, _ = cv2.findContours(binary_image_2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #*  use to increase the contours radius
    # resize_bg = np.zeros_like(image2)
    # for con in contours:
    #     t = cv2.approxPolyDP(con, 10, True)
    #     center, radius = cv2.minEnclosingCircle(t)
    #     #* print("radius",radius)
    #     resize_bg = cv2.circle(resize_bg, center=(int(center[0]), int(center[1])), radius=int(radius), color=(255, 255, 255), thickness=-1)

    # resize_bg = cv2.cvtColor(resize_bg, cv2.COLOR_BGR2GRAY)
    # _, binary_resize_bg = cv2.threshold(resize_bg, 254, 255, cv2.THRESH_BINARY)
    # _, binary_resize_bg_2 = cv2.threshold(resize_bg, 254, 255, cv2.THRESH_BINARY_INV)
    # non_mask_image = cv2.bitwise_or(image2, black_background, mask=binary_resize_bg_2)

    # contours_2, _ = cv2.findContours(binary_resize_bg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image1_copy = image1.copy()
    for contour in contours:
        black_background_2 = np.zeros_like(image1_copy)
        t = cv2.approxPolyDP(contour, 10, True)

        center2, radius2 = cv2.minEnclosingCircle(t)
        # print("radius2",radius2)

        #* the image that car light is white (channels are 3)
        binary_mask_img = cv2.circle(black_background_2, center=(int(center2[0]), int(center2[1])), radius=int(radius2), color=(255, 255, 255), thickness=-1)
        #* convert to binary image(channels is 1)
        gray_bmi = cv2.cvtColor(binary_mask_img, cv2.COLOR_BGR2GRAY)
        #* convert to binary image that car light is 255 else 0
        _, binary_slimg = cv2.threshold(gray_bmi, 200, 255, cv2.THRESH_BINARY)
        car_light_img = cv2.bitwise_and(image1_copy, image1_copy, mask=binary_slimg)
        corrected_region = gamma_correction(car_light_img, gamma)
        non_mask_image = cv2.bitwise_or(non_mask_image, corrected_region)

    return non_mask_image

def extract_number(filename : str):
    '''
    Using to extract number in file path 

    Parameters
    ----------
    filename : just filename
    '''
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def main(args):
        imo_dir = args.folder_de
        ime_dir = args.folder_en

        sub_filename    = ['jpg', 'bmp', 'png']
        file_path_o     = []
        file_path_en    = []

        for e in sub_filename:

                file_path_o.extend(glob.glob(imo_dir + "*." + e))
                file_path_en.extend(glob.glob(ime_dir + "*." + e))
                
        file_path_o = sorted(file_path_o, key = extract_number)
        file_path_en = sorted(file_path_en, key = extract_number)
        # print("file_path_o", file_path_o)
        # print('file_path_en', file_path_en)
        for fo in file_path_o:
               print(fo)

        images_original = [cv2.imread(fo) for fo in file_path_o]
        images_enhance = [cv2.imread(fe) for fe in file_path_en]

        file_len = min(len(images_original), len(images_enhance)) 

        for i in range(file_len):
                # cv2.imshow("img", np.hstack([images_original[i], images_enhance[i]]))
                # cv2.waitKey(0)
                restore_img = erase_car_light(images_original[i], images_enhance[i], args.gamma, args.light_threshold)
                cv2.imwrite(os.path.join(args.folder_s + f"{i}.png"), img = restore_img)
                print(i)


        

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="DIP final project")
        parser.add_argument("-fd", '--folder_de', default = "./demo/default/", type = str, help = "the folder where your original images store")
        parser.add_argument("-fe", '--folder_en', default = "./demo/enhance/", type = str, help = "the folder where your enhance images store")
        parser.add_argument("-fs", '--folder_s', default = "./enhance3/", type = str, help = "path to save process images" )
        parser.add_argument("-g", '--gamma', default= 20, type = float, help = "the gamma correction parameter.")     
        parser.add_argument("-lt", '--light_threshold', default = 5, type = int, help = "determine the car light threshold")  
        args = parser.parse_args()
        main(args)