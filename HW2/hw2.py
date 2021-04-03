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
    img[:,:] = G[:,:]
    cv2.imwrite(args.output, img)

elif target == 'result2':
    #2st order edge detection
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(args.output, img2)

elif target == 'result3':
    #Canny edge detection
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img2 = np.zeros(img.shape, np.uint8)
    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)
    
elif target == 'result4':
    #Apply an crispening method
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    img2 = np.zeros(img.shape, np.uint8)
    cv2.imwrite(args.output, img2) 
    util.histogram_draw(img2, target)
    
elif target == 'result5':
    #Generate edge map of result4 as result5
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)

elif target == 'result6':
    #Edge crispening as best as you can

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    
    
    cv2.imwrite(args.output, img)
    util.histogram_draw(img, target)
    
elif target == 'result7':
    # Make sample3 like sample4

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)

elif target == 'result8':
    # Make sample5 like sample6

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)
