import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 
from utils import util


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

if target == '1_result':
    img = cv2.imread(args.input)
    img_gray = util.to_gray(img)
    cv2.imwrite(args.output, img_gray)

elif target == '2_result':
    img = cv2.imread(args.input)
    img2 = np.zeros(img.shape, np.uint8)
    for i in range(img2.shape[1]):
        img2[:,i,:] = img[:,img2.shape[1]-1-i,:]

    cv2.imwrite(args.output, img2)

elif target == '3_result':
    img = cv2.imread(args.input)
    img2 = np.zeros(img.shape, np.uint8)
    img2 = img/5
    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)
    util.histogram_draw(img, 'sample2')
    
elif target == '4_result':
    img = cv2.imread(args.input)
    img2 = np.zeros(img.shape, np.uint8)
    img2 = img*5
    cv2.imwrite(args.output, img2) 
    util.histogram_draw(img2, target)
    
elif target == '5_result':
    #Global histogram eq
    img = cv2.imread(args.input)
    img2 = util.to_gray(img)

    histo_np = img2[:][:].ravel()
    hist = np.zeros(256)
    
    for i in histo_np:
        hist[i] += 1
    
    #Create trans func 
    CDF = np.zeros(256)
    CDF = hist.cumsum()

    trans_f = np.zeros(256)
    for i in range(256):
        trans_f[i] = round((CDF[i] - CDF.min())*255/(CDF.max() - CDF.min()))

    #Map the img and plothistogram2
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i][j][0] = trans_f[img2[i][j][0]]
    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)

elif target == '6_result':
    # Local histogram EQ
    

    img = cv2.imread(args.input)
    img = util.to_gray(img) 

    matrix_edge_size = 7

    img2 = util.Padding(img, (matrix_edge_size-1)/2)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j][0] = util.Local_EQ(i+1, j+1, img2, b = matrix_edge_size, color = 0)
    
    
    #cv2.imwrite(args.output, img)
    cv2.imwrite(f'./result/6_result_{matrix_edge_size}x{matrix_edge_size}.jpg', img)

    util.histogram_draw(img, target)
    
elif target == '7_result':
    # Enhance

    img = cv2.imread(args.input)
    img2 = util.to_gray(img)
    img2 = img2//1
    #histo = []
    
    #for i in range(img.shape[0]):
    #    for j in range(img.shape[1]):
    #        histo.append(img[i][j][0])
    histo_np = img2[:][:].ravel()
    hist = np.zeros(256)
    
    for i in histo_np:
        hist[i] += 1
    
    #Create trans func 
    CDF = np.zeros(256)
    #for i in range(256):
    #    CDF[i] = CDF[i-1] + hist[i]
    
    CDF = hist.cumsum()

    trans_f = np.zeros(256)
    for i in range(256):
        trans_f[i] = round((CDF[i] - CDF.min())*255/(CDF.max() - CDF.min()))

    #Map the img and plothistogram2
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i][j][0] = trans_f[img2[i][j][0]]
    

    cv2.imwrite(args.output, img2)
    util.histogram_draw(img2, target)
    

elif target == '8_result':
    # remove noise from sample7
    
    img = cv2.imread(args.input)
    ori_img = cv2.imread(args.input)
    img2 = util.Padding(img)

    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
            img[i][j] = util.Noise_Mask(i+1, j+1, img2, b = 3)
    print("PSNR =", util.PSNR(ori_img, img))
    cv2.imwrite(args.output, img)

elif target == '9_result':
    # remove noise from sample8
    
    img = cv2.imread(args.input)
    ori_img = cv2.imread(args.input)
    img2 = util.Padding(img)
    
   
    for i in range(4):
        img2 = util.Padding(img)
    
        for i in range(img2.shape[0]-2):
            for j in range(img2.shape[1]-2):
                img[i][j] = util.Noise_Mask_med(i+1, j+1, img2, b = 1)

    print("PSNR =",util.PSNR(ori_img, img))
    cv2.imwrite(args.output, img)


