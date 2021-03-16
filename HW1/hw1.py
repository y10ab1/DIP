import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 

def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='input path')
    parser.add_argument('--output', '-o', required=True, help='output path')
    return parser.parse_args()

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
    for i in range(imgI.shape[0]):
        for j in range(imgI.shape[1]):
            MSE += (imgI[i][j] - imgK[i][j])**2
    MSE /= imgI.shape[0]*imgI.shape[1]
    psnr = 20*np.log10((255)/(MSE)**(1/2))
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
                rank += 1
    max_intensity = max(hist)
    return (max_intensity * rank) // (b**2)

def to_gray(img):
    img_gray = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_gray[i][j][0] = (img[i][j][0]*299 +img[i][j][1]*587+img[i][j][2]*114 +500)/1000
    return img_gray

if __name__ == '__main__':

    args = process_command()
    target = args.output.split('/')[-1].split('.')[0]

    print(target)

#Generate picture

if target == '1_result':
    img = cv2.imread(args.input)
    img_gray = to_gray(img)
    cv2.imwrite(args.output, img_gray)

elif target == '2_result':
    img = cv2.imread(args.input)
    img2 = np.zeros(img.shape, np.uint8)

    for i in range(img2.shape[1]):
        img2[:,i] = img[:,img2.shape[1]-1-i]

    cv2.imwrite(args.output, img2)

elif target == '3_result':
    img = cv2.imread(args.input)
    img2 = np.zeros(img.shape, np.uint8)
    img2 = img/5
    cv2.imwrite(args.output, img2)
    histogram_draw(img2, target)
    histogram_draw(img, 'sample2')
    
elif target == '4_result':
    img = cv2.imread(args.input)
    img2 = np.zeros(img.shape, np.uint8)
    img2 = img*5
    cv2.imwrite(args.output, img2) 
    histogram_draw(img2, target)
    
elif target == '5_result':
    #Global histogram eq
    img = cv2.imread(args.input)
    img2 = to_gray(img)
    #histo = []
    #for i in range(img.shape[0]):
    #    for j in range(img.shape[1]):
    #        histo.append(img[i][j][0])
    

    #histo_np = np.array(histo)
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
    histogram_draw(img2, target)

elif target == '6_result':
    # Local histogram EQ
    

    img = cv2.imread(args.input)
    
    img3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img3[i][j] = (img[i][j][0]*299 +img[i][j][1]*587+img[i][j][2]*114 +500)/1000
    
    img2 = Padding(img3, 15)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img3[i][j][0] = Local_EQ(i+1, j+1, img2, b = 31, color = 0)
    
    
    cv2.imwrite(args.output, img3)
    histogram_draw(img3, target)
    
elif target == '7_result':
    # Enhance

    img = cv2.imread(args.input)
    img2 = to_gray(img)
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
    histogram_draw(img2, target)
    

elif target == '8_result':
    # remove noise from sample7
    
    img = cv2.imread(args.input)
    ori_img = img
    img2 = Padding(img)

    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
            img[i][j] = Noise_Mask(i+1, j+1, img2, b = 3)

    cv2.imwrite(args.output, img)

elif target == '9_result':
    # remove noise from sample8
    
    img = cv2.imread(args.input)
    ori_img = img
    img2 = Padding(img)
    
    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
           img[i][j] = Noise_Mask_med(i+1, j+1, img2, b = 1)
    
    img2 = Padding(img)
    
    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
           img[i][j] = Noise_Mask_med(i+1, j+1, img2, b = 1)
    
    img2 = Padding(img)
    
    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
           img[i][j] = Noise_Mask_med(i+1, j+1, img2, b = 1)
    
    img2 = Padding(img)
    
    for i in range(img2.shape[0]-2):
        for j in range(img2.shape[1]-2):
           img[i][j] = Noise_Mask_med(i+1, j+1, img2, b = 1)
      
    cv2.imwrite(args.output, img)


