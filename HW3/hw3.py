import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 
from utils import util
from utils import util2
from utils import util3


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
    #
    new_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    new_img = util3.boundary_extract(new_img)
    cv2.imwrite(args.output, new_img)

elif target == 'result2':
    #
    new_img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    new_img = util3.hole_filling(new_img, 189, 49)
    new_img = util3.hole_filling(new_img, 169, 84)
    new_img = util3.hole_filling(new_img, 304, 163)
    new_img = util3.hole_filling(new_img, 195, 298)
    new_img = util3.hole_filling(new_img, 59, 122)
    new_img = util3.hole_filling(new_img, 232, 63)
    new_img = util3.hole_filling(new_img, 93, 264)
    new_img = util3.hole_filling(new_img, 93, 128)
    cv2.imwrite(args.output, new_img)
    
    num_objects = util3.object_counter(new_img)
    print('num_objects: ',num_objects)

elif target == 'result3':
    #
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    feature_vector_set = util3.laws_method(img)
    
    for i in range(9):
        cv2.imwrite(f'./result/law{i+1}.png', feature_vector_set[i])
    
elif target == 'result4':
    #
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(args.output, new_img) 
    
elif target == 'result5':
    #
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(args.output, new_img)

    
    

