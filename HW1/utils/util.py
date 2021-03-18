import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input path')
    parser.add_argument('--output', '-o', required=True, help='output path')
    return parser.parse_args()

def to_gray(img):
    img_gray = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gray[i][j][0] = (img[i][j][0]*299 +img[i][j][1]*587+img[i][j][2]*114 +500)/1000
    return img_gray

#generate histogram
def histogram_draw(img, name):
    histo = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histo.append(img[i][j][0])
    histo_np = np.array(histo)
    
    plt.hist(histo_np, bins = range(256)) 
    plt.title("histogram")
    plt.savefig(f'./result/{name}_histogram.jpg')
    plt.close()


def PSNR(imgI, imgK):
    imgI = to_gray(imgI)    
    imgK = to_gray(imgK)    
    MSE = 0
    for i in range(imgI.shape[0]):
        for j in range(imgI.shape[1]):
            MSE += (int(imgI[i][j][0]) - int(imgK[i][j][0]))**2
    
    MSE = MSE/((imgI.shape[0])*(imgI.shape[1]))
    psnr = 20*(np.log10((255)/(MSE)**(1/2)))
    return psnr

def Noise_Mask(Ci, Cj, img, b = 2):
    Masked_val = 0
    
    mask = np.array([[1, b, 1], [b, b**2, b], [1, b, 1]])
    for i in range(3):
        for j in range(3):
            Masked_val += mask[i][j] * img[Ci+i-1][Cj+j-1][0]
    Masked_val = Masked_val // ((b+2)**2)
    return Masked_val 


def Noise_Mask_med(Ci, Cj, img, b = 2):
    Masked_val = 0
    Med = []
    for i in range(3):
        for j in range(3):
            Med.append(img[Ci+i-1][Cj+j-1][0])
    MAXMIN = max(min(Med[:3]), min(Med[1:4]), min(Med[2:5]),min(Med[3:6]), min(Med[4:7]), min(Med[5:8]),min(Med[6:9]))
    MINMAX = min(max(Med[:3]), max(Med[1:4]), max(Med[2:5]),max(Med[3:6]), max(Med[4:7]), max(Med[5:8]),max(Med[6:9]))
    #print((MAXMIN+MINMAX)/2)
    Masked_val = (int(MAXMIN)+int(MINMAX))//2
    #Masked_val = median(Med[:9])
    return Masked_val 

def Padding(img, pad_num = 1):

    img2 = np.zeros((img.shape[0]+2, img.shape[1]+2, img.shape[2]), np.uint8)
    
    # edges
    for i in range(img.shape[1]):
        img2[0][i+1] = img[0][i]
    for i in range(img.shape[1]):
        img2[img2.shape[0]-1][i+1] = img[img.shape[0]-1][i]
    for i in range(img.shape[0]):
        img2[i+1][0] = img[i][0]
    for i in range(img.shape[0]):
        img2[i+1][img2.shape[1]-1] = img[i][img.shape[1]-1]


    # corner
    img2[0][0] = img[0][0]
    img2[0][img2.shape[1]-1] = img[0][img.shape[1]-1]
    img2[img2.shape[0]-1][0] = img[img.shape[0]-1][0]
    img2[img2.shape[0]-1][img2.shape[1]-1] = img[img.shape[0]-1][img.shape[1]-1]
    
    # center
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i+1][j+1] = img[i][j]
    
    if pad_num == 1:
        return img2
    else :
        return Padding(img2, pad_num = pad_num-1)

def Local_EQ(Ci, Cj, img, b = 3, color = 0):
    max_intensity = 0
    rank = 0
    offset = (b-1)//2
    hist = []
    for i in range(b):
        for j in range(b):
            hist.append(img[Ci+i-offset][Cj+j-offset][color])
            if img[Ci+i-offset][Cj+j-offset][color] > img[Ci][Cj][color]:
                continue    
            rank += 1
    max_intensity = max(hist)
    return (max_intensity * rank) // (b**2)


