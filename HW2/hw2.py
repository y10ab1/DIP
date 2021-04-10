import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 
from utils import util
from utils import util2


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input path')
    parser.add_argument('--output', '-o', required=True, help='output path')
    return parser.parse_args()


if __name__ == '__main__':

    args = process_command()
    target = args.output.split('/')[-1].split('.')[0]

    print(target)

#Generate picture

if target == 'result1':
    #1st order edge detection
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    (G, theta) = util2.Edge_Det_1st(img)
    cv2.imwrite(args.output, G)

elif target == 'result2':
    #2st order edge detection
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    new_img = util2.Edge_Det_2nd(img)
    cv2.imwrite(args.output, new_img)
    util.histogram_draw(new_img, target)
elif target == 'result3':
    #Canny edge detection
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    new_img = util2.Edge_Det_Canny(img)
    cv2.imwrite(args.output, new_img)
    #util.histogram_draw(img2, target)
    
elif target == 'result4':
    #Apply an crispening method
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    new_img = util2.Edge_Cris(img)
    cv2.imwrite(args.output, new_img) 
    #util.histogram_draw(img2, target)
    
elif target == 'result5':
    #Generate edge map of result4 as result5
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    new_img = util2.Edge_Det_Canny(img)
    cv2.imwrite(args.output, new_img)
    #util.histogram_draw(img2, target)

elif target == 'result6':
    #Edge crispening as best as you can

    new_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    
    new_img = util.power_law_filter(new_img, gamma=1.95)
    new_img = util.power_law_filter(new_img, gamma=1.85)
    new_img = util2.Edge_Cris(new_img)
    new_img = util2.low_pass_filter(new_img)
    new_img = util2.low_pass_filter(new_img)
    new_img = util2.Edge_Det_Canny(new_img)
    new_img = util2.Edge_Det_2nd(new_img) 
    #new_img = util2.low_pass_filter(new_img) 
    cv2.imwrite(args.output, new_img)
    #util.histogram_draw(img, target)
    
elif target == 'result7':
    # Make sample3 like sample4

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    #new_img = util2.translate(img)
    
    new_img = util2.rotate(img, theta=90, Ci=img.shape[0]//2, Cj=img.shape[1]//2)
    new_img = util2.translate(new_img, tx=-100, ty=-300)
    new_img = util2.scale(new_img, Sx=1.8, Sy=1.8)
    new_img = util2.translate(new_img, tx = 50, ty = -100)
    for i in range(19):
        new_img = util2.Around(new_img)
    new_img *= 2
    cv2.imwrite(args.output, new_img)
    #util.histogram_draw(img2, target)

elif target == 'result8':
    # Make sample5 like sample6
    new_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    #new_img = util2.Black_hole(img)
    #cv2.imwrite(args.output, new_img)
    new_img = util2.D(new_img)
    #cv2.imwrite('./result/pin_dist.jpg', new_img)
    new_img = util2.Black_hole(new_img , 8, step = 1, theta_offset=1)
    new_img = util2.Black_hole(new_img , 16, step = 1, theta_offset=1)
    new_img = util2.Black_hole(new_img , 64, step = 1, theta_offset=1)
    new_img = util2.Black_hole(new_img , 256, step = 1, theta_offset=1)
    cv2.imwrite(args.output, new_img)
    

    #util.histogram_draw(img2, target)
