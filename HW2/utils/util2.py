import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 
from numba import jit
import time
from functools import cache
from numpy.linalg import inv


def Edge_Det_1st(img, K = 2):
    #Default Sobel
    Kx = np.array(([-1, 0, 1], [-K, 0, K], [-1, 0, 1]), np.float32)
    Ky = np.array(([1, K, 1], [0, 0, 0], [-1, -K, -1]), np.float32)
                
    Ix = cv2.filter2D(img,-1, Kx)
    Iy = cv2.filter2D(img,-1, Ky)
    G = np.hypot(Ix, Iy)
    G = (G/G.max()*255)

    theta = np.arctan2(Iy, Ix)
                                           
    img[:,:] = G[:,:] 
    return (img, theta)

def Edge_Det_2nd(img):
    G_low_pass = np.array(([1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]), np.float32) * (1/273)
    
    new_img = cv2.filter2D(img,-1, G_low_pass)

    K1 = np.array(([-1, 0, -1],
                   [ 0, 4, 0],
                   [-1, 0, -1]), np.float32) /4
    K = np.array(([-2, 1, -2],
                  [ 1, 4, 1],
                  [-2, 1, -2]), np.float32) / 8

    new_img = cv2.filter2D(img, -1, K1)
    Zi, Zj = np.where(new_img <= 0)
    
    new_img[Zi,Zj] = 0

    ZC = np.array(([ 1, 1, 1],
                   [ 1, 0, 1],
                   [ 1, 1, 1]), np.float32)
    new_img = cv2.filter2D(new_img, -1, ZC) 

    return new_img


def non_max_suppression(img, D):
    Z = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    a0i, a0j = np.where(((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180)))
    a45i, a45j = np.where((22.5 <= angle) & (angle< 67.5))
    a90i, a90j = np.where((67.5 <= angle) & (angle < 112.5))
    a135i, a135j = np.where((112.5 <= angle) & (angle< 157.5))

    for i,j in zip(a0i, a0j):
        if i==img.shape[0]-1 or j==img.shape[1]-1 or i==0 or j ==0:
            continue
        if img[i, j+1] > img[i,j] or img[i, j-1] > img[i,j]:
            Z[i,j] = 0
        else :
            Z[i,j] = img[i,j]
    
    for i,j in zip(a45i, a45j):
        if i==img.shape[0]-1 or j==img.shape[1]-1 or i==0 or j ==0:
            continue
        if img[i+1, j-1] > img[i,j] or img[i-1, j+1] > img[i,j]:
            Z[i,j] = 0
        else :
            Z[i,j] = img[i,j]

    for i,j in zip(a90i, a90j):
        if i==img.shape[0]-1 or j==img.shape[1]-1 or i==0 or j ==0:
            continue
        if img[i+1, j] > img[i,j] or img[i-1, j] > img[i,j]:
            Z[i,j] = 0
        else :
            Z[i,j] = img[i,j]

    for i,j in zip(a135i, a135j):
        if i==img.shape[0]-1 or j==img.shape[1]-1 or i==0 or j ==0:
            continue
        if img[i-1, j-1] > img[i,j] or img[i+1, j+1] > img[i,j]:
            Z[i,j] = 0
        else :
            Z[i,j] = img[i,j]

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    highThreshold = 35
    lowThreshold = 20
    print('H',highThreshold)
    print('L',lowThreshold)
    res = np.zeros((img.shape[0],img.shape[1]), np.float32)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = np.uint8(255)
    res[weak_i, weak_j] = np.uint8(25)
    
    return res

def hysteresis(img):

    K = np.array(([1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]), np.float32)

    new_img = cv2.filter2D(img, -1, K)
    new_img[new_img < 255] = 0
    
    return new_img

def horiz_Cri(img):
    H = np.array([1, 1, 1, 0, 1, 1, 1])/5
    new_img = cv2.filter2D(img,-1, H)
    return new_img

def Around(img):
    K = np.array(([1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]), np.float32)/8
    new_img = cv2.filter2D(img, -1, K)
    return new_img

def Edge_Det_Canny(img):

    #Noise reduction
    G_low_pass = np.array(([2, 4, 5, 4, 2],
                           [4, 9, 12, 9, 4],
                           [5, 12, 15, 12, 5],
                           [4, 9, 12, 9, 4],
                           [2, 4, 5, 4, 2]), np.float32) * (1/159)

    new_img = cv2.filter2D(img,-1, G_low_pass)
    #new_img = cv2.filter2D(new_img,-1, G_low_pass)
    
    #Compute Grad and theta
    new_img, theta = Edge_Det_1st(new_img)

    #Non-maximal suppression
    start_time = time.time()
    new_img = non_max_suppression(new_img, theta)
    print('Time used: {} sec'.format(time.time()-start_time))

    #Hysteretix thresholding
    new_img = threshold(new_img)

    #Connected component labeling method
    new_img = hysteresis(new_img)
    #new_img = horiz_Cri(new_img)
    return new_img

def Edge_Cris(img):
    #Unsharp Masking
    
    G_low_pass = np.array(([2, 4, 5, 4, 2],
                           [4, 9, 12, 9, 4],
                           [5, 12, 15, 12, 5],
                           [4, 9, 12, 9, 4],
                           [2, 4, 5, 4, 2]), np.float32) * (1/159)
    FL = cv2.filter2D(img,-1, G_low_pass)
    F  = img

    c = 4/5

    new_img = (c/(2*c-1))*F - ((1-c)/(2*c-1))*FL

    return new_img

def low_pass_filter(img, times = 5):

    G_low_pass = np.array(([2, 4, 5, 4, 2],
                           [4, 9, 12, 9, 4],
                           [5, 12, 15, 12, 5],
                           [4, 9, 12, 9, 4],
                           [2, 4, 5, 4, 2]), np.float32) * (1/159)
    for i in range(times):
        new_img = cv2.filter2D(img,-1, G_low_pass)

    return new_img

def translate(img, tx=100, ty=-500, f=0):
    T = np.array(([1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1 ]), np.int32)
    
    Ii ,Ij = np.where(img >= 0)
    Iv = img[Ii, Ij]
    I1 = np.ones(Ii.size)
    Ori_index = np.array((Ii, Ij, I1), np.int32)
    
    new_index = T.dot(Ori_index)
    new_index = np.around(new_index)
    new_index = new_index.astype(int)
    new_img = np.zeros((img.shape))
    
    for i, j, v in zip(new_index[0], new_index[1], Iv):
        if (0<= i < img.shape[0]) and (0<= j < img.shape[1]):
            new_img[i, j] = v
            continue
        if f ==1 and j < 0 and v > 0:
            print((i,j,v))
    
    return new_img

def scale(img, Sx=1.2, Sy=1.2):
    S = np.array(([Sx, 0, 0],
                  [0, Sy, 0],
                  [0, 0, 1 ]), np.float32)
    
    Ii ,Ij = np.where(img >= 0)
    Iv = img[Ii, Ij]
    I1 = np.ones(Ii.size)
    Ori_index = np.array((Ii, Ij, I1), np.int32)
    
    new_index = S.dot(Ori_index)
    new_index = np.around(new_index)
    new_index = new_index.astype(int)
    new_img = np.zeros((img.shape))
    
    for i, j, v in zip(new_index[0], new_index[1], Iv):
        if (0<= i < img.shape[0]) and (0<= j < img.shape[1]):
            new_img[i, j] = v
    
    return new_img
    

def rotate(img, theta = 90, Ci=0, Cj=0):
    t = np.array([np.radians(theta)])
    R = np.array(([np.cos(t), -np.sin(t), 0],
                  [np.sin(t), np.cos(t), 0],
                  [0,         0,        1 ]), np.float32)
    Ii ,Ij = np.where(img >= 0)
    Iv = img[Ii, Ij]
    I1 = np.ones(Ii.size)
    Ii -= Ci
    Ij -= Cj
    Ori_index = np.array((Ii, Ij, I1), np.int32)
    
    new_index = R.dot(Ori_index)
    #new_index = new_index.astype(np.int32)
    new_img = np.zeros((img.shape))
    new_index[0] += Ci
    new_index[1] += Cj
    new_index = np.around(new_index)
    new_index = new_index.astype(int)
    for i, j, v in zip(new_index[0], new_index[1], Iv):
        if (0<= i < img.shape[0]) and (0<= j < img.shape[1]):
            new_img[i, j] = v
    
    return new_img


def distor(img,U,V,X,Y):

    new_img = np.zeros((img.shape))

    x = X
    y = Y
    Iv = img[x, y]
    x2 = x**2
    y2 = y**2
    xy = x*y
    I = np.ones(x.size)

    Output_points = np.array((I,
                              x,
                              y,
                              x2,
                              xy,
                              y2), np.int32)
    Input_points = np.array((U,V),np.int32) 
    
    coef_matrix = np.array(([-1.0e-5],
                           [0.025],
                           [0.0],
                           [0.0],
                           [0.0],
                           [0.0]))
    print('coef', coef_matrix)
    
    img_matrix = np.array((np.where(img>=0)), np.int32)

    new_index = np.array(result_matrix[1],result_matrix[2]) 
    new_index = np.around(new_index)
    new_index = new_index.astype(int)
    print('new_index',new_index)
    print('Iv',Iv)
    for i, j, v in zip(new_index[0], new_index[1], Iv):
        if (0<= i < img.shape[0]) and (0<= j < img.shape[1]):
            new_img[i, j] = v
    return new_img

def D(img,  k = -0.0001):
    Ci = img.shape[0]//2
    Cj = img.shape[1]//2

    xx,yy = np.where(img>=0)
     
    V = img[xx, yy]
    xx -= Ci
    yy -= Cj
    
    r,theta = cv2.cartToPolar(xx.astype(np.float32),yy.astype(np.float32),angleInDegrees=True)
    
    r_d = r*(1-k*((np.absolute(r))**2))
    r_d /= 28

    Dx, Dy = cv2.polarToCart(r_d,theta,angleInDegrees=True)

    Dx += Ci
    Dy += Cj
    new_img = np.zeros(img.shape)
    Dx = np.around(Dx).astype(int)
    Dy = np.around(Dy).astype(int)
    for i, j, v in zip(Dx, Dy, V):
        if (0<= i < img.shape[0]) and (0<= j < img.shape[1]):
            new_img[i, j] = v
    return new_img

def b_rotate(img, theta = 90, Ci=0, Cj=0, ri=0, rj=0):
    t = np.array([np.radians(theta)])
    R = np.array(([np.cos(t), -np.sin(t), 0],
                  [np.sin(t), np.cos(t), 0],
                  [0,         0,        1 ]), np.float32)
    Ii ,Ij = np.where(img >= 0)
    Iv = img[Ii, Ij]
    I1 = np.ones(Ii.size)
    Ii -= Ci
    Ij -= Cj
    Ori_index = np.array((Ii, Ij, I1), np.int32)
    
    new_index = R.dot(Ori_index)
    new_index[0] += Ci
    new_index[1] += Cj
    new_index = np.around(new_index)
    new_index = new_index.astype(int)
    for i, j, v in zip(new_index[0], new_index[1], Iv):
        if (0<= i < img.shape[0]) and (0<= j < img.shape[1]) and ((Ci-ri) <= i <= (Ci+ri)) and ((Cj-rj) <= j <= (Cj+rj)):
            img[i, j] = v
    
    return img

def Black_hole(img, R = 512, step = 50, theta_offset = 1):
    new_img = img
    for r in range(1, R, step):
        print(r,'/', R, ',theta=', theta_offset)
        new_img = b_rotate(new_img, theta = theta_offset, Ci=img.shape[0]//2, Cj=img.shape[1]//2, ri=r, rj =r)
    return new_img



