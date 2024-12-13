import os 
import cv2
import numpy as np
import argparse
from denoising import denoising
from utils import *

import matlab.engine
import time

eng = matlab.engine.start_matlab()

def main(img,i,name,**args):
    Img = img / 255.    
    starttime = time.time()
    I_intial0 = Illumination(Img,'max_c')    # Get the initial illumination map by max value of three channel
    I_intial = I_intial0.tolist()

    #for displaying: initial illumination map
    # I_in = np.array(I_intial0)
    # I_init = np.ones(img.shape)
    # I_init[:,:,0],I_init[:,:,1],I_init[:,:,2] =I_in,I_in,I_in
    # epsilon = 1e-6  # A small constant to prevent division by zero
    # I_init = np.clip(I_init, epsilon, np.inf)  # Clip values to avoid zeros
    # cv2.imshow("Initial Illumination Map", I_in)  # Display the initial illumination map

    I_optimized = eng.demo_L0_RTV(matlab.double(I_intial))      # optimize the illimination map
    I_optimized = np.array(I_optimized)
    
    I_enhances = Adapt_gamma(I_optimized,0.7)           # adaptive gamma correction
    
    I_unenhence = np.ones(img.shape)
    I_unenhence[:,:,0],I_unenhence[:,:,1],I_unenhence[:,:,2] = I_optimized,I_optimized,I_optimized 
    I_enhence = np.ones(img.shape)
    I_enhence[:,:,0],I_enhence[:,:,1],I_enhence[:,:,2] = I_enhances,I_enhances,I_enhances 
    
    R = Img / I_unenhence       # Reflection Image

    #for displaying
    # cv2.imshow("I_enhence", I_enhence)
    # cv2.imshow("I_unenhence", I_unenhence)
    # cv2.imshow("R", R)

    R_d = guideFilter(Img,R,(35,35),0.00001)  # Get difference of the origin and the Smooth reflection image, which is the detail of the image
    d = R - R_d 
    R_ref = R_d + 1.3*d

    #for displaying
    # cv2.imshow("R_d", R_d)
    # cv2.imshow("d", d)
    # cv2.imshow("R_ref", R_ref)

    d_uint8 = (d * 255).clip(0, 255).astype(np.uint8)
    gray_d = cv2.cvtColor(d_uint8, cv2.COLOR_BGR2GRAY)
    _,halo_mask = cv2.threshold(gray_d, 10, 255, cv2.THRESH_BINARY) # Get the halo mask from the detail image

    I_halo = I_enhence.copy()
    I_halo[halo_mask > 0] *= args['maskratio']       # lower the halo area enhancement

    #for displaying
    # cv2.imshow("gray", gray_d)
    # cv2.imshow("halo mask", halo_mask)
    # cv2.imshow("I_enhance", I_enhence)
    # I_d = I_enhence - I_init
    # cv2.imshow("I_d" , I_d)
    # cv2.imshow("I_halo",  I_halo)

    I_halo_enhance = eng.demo_L0_RTV(matlab.double(I_halo)) # optimize the processed illumination map
    I_halo_enhance= np.array( I_halo_enhance)

    #for displaying
    # cv2.imshow("I_halo_enhance",   I_halo_enhance)
    # I_diff = I_enhence - I_halo_enhance
    # cv2.imshow("I_diff",   I_diff)

    Out_d = R_ref * I_halo_enhance     # combine refined illumination and optimized reflection maps

    Out_d = denoising(Out_d)    # denoise the final enhanced images
    endtime = time.time()
    print('The ' + name+' Time:' +str(endtime-starttime)+'s.') 

    Out = (Out_d*255.).clip(0, 255).astype(np.uint8)
    cv2.imwrite(args['output']+name+'_Out.png',Out)
    #for displaying
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the OpenCV window

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Low Light Image Enhancement")
    
    parser.add_argument("--input", type=str, default="./data/input/", help='path to input image')
    parser.add_argument("--output", type=str, default="./Result/", help='path to output image')
    parser.add_argument("--maskratio", type=float, default= 0.3, help='the degree of suppressing the light effect')
    argspar = parser.parse_args()
    
    # print("\n### Testing LLIE model ###")
    # print("> Parameters:")
    # for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
    #     print('\t{}: {}'.format(p, v))
    # print('\n')
    
    filepath = argspar.input
    files =os.listdir(filepath)

    for i in range (len(files)):
        name = files[i][:-4]
        img = cv2.imread(filepath+ '/'+files[i])
        main(img,i,name,**vars(argspar))