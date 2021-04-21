import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 

# Add some argument
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input path')
    parser.add_argument('--output', '-o', required=True, help='output path')
    return parser.parse_args()

# Compute noe channel image (gray)
def to_gray(img):
    img_gray = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gray[i][j][0] = (img[i][j][0]*299 +img[i][j][1]*587+img[i][j][2]*114 +500)/1000
    return img_gray

# Generate histogram
def histogram_draw(img, name):
    histo = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histo.append(img[i][j])
    histo_np = np.array(histo)
    
    plt.hist(histo_np, bins = range(256)) 
    plt.title("histogram")
    plt.savefig(f'./result/{name}_histogram.jpg')
    plt.close()

def power_law_filter(img, gamma = 0.65):
    gamma_corrected = np.array(255*(img / 255) ** gamma, np.uint8) 
    return gamma_corrected

def log_filter(img, c = 0):
    if c == 0:
        c = 255/(np.log(1 + np.max(img))) 
    new_img = np.array(c*(np.log10(img+1.1)), np.uint8)
    return new_img

def Reverse_log_filter(img, c = 1):
    c = 255/(np.log(1 + np.max(img))) 
    new_img = np.array(c*(np.log10(1/(img+1.1))), np.uint8)
    return new_img

def Reverse_img(img):
    img2 = np.zeros(img.shape, np.uint8)
    for i in range(img2.shape[1]):
        img2[:,i,:] = img[:,img2.shape[1]-1-i,:]
    return img2

# Compute PSNR
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

# Denoise
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
            Med.append(img[Ci+i-1][Cj+j-1])
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

def DEnoise(img):
    img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    new_img = np.zeros((img2.shape[0]-2,img2.shape[1]-2))
    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
            new_img[i,j] = Noise_Mask_med(i+1, j+1, img2, b = 1)
    print(new_img)
    return new_img

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

def GHE(img2):
    img2 = img2//1
    
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
    return img2

def LHE(img, matrix_edge_size = 13):

    img2 = Padding(img, (matrix_edge_size-1)/2)
    img = np.zeros((400,560,1), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img2[i+(matrix_edge_size-1)//2][j+(matrix_edge_size-1)//2][0] \
                    = Local_EQ(i+(matrix_edge_size-1)//2, j+(matrix_edge_size-1)//2, img2, b = matrix_edge_size, color = 0)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j,0] = img2[i+(matrix_edge_size-1)//2][j+(matrix_edge_size-1)//2][0]

    return img
