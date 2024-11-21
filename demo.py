import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import argparse
import os

from exposure_enhancement import *

def main(args):
        im_dir = args.folder
        sub_filename = ['jpg', 'bmp', 'png']
        file_path    = []
        for e in sub_filename:
                file_path.extend(glob.glob(im_dir + "*." + e))
        images_enhance = [cv2.imread(file) for file in file_path]


        # L0 = np.max(images_enhance[0], axis = -1)
        # L1 = np.max(images_enhance[1], axis = -1)

        # cv2.imshow("ILL", L1-L0)
        # cv2.waitKey(0)
        for i, img in enumerate(images_enhance):
                # print(img.shape)
                #* execute lime 
                restore_img = enhance_image_exposure(img, args.gamma, args.lambda_, sigma=args.sigma ,eps=args.eps)
                cv2.imwrite(os.path.join("./enhance/" + f"{i}.png"), img = restore_img)
                cv2.imshow("i", restore_img)
                cv2.waitKey(0)
        
        
        

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