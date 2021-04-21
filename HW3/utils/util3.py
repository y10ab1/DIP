import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 
from numba import jit
import time
from functools import cache
from numpy.linalg import inv

def erode(img):

    ori_img = img
    BE = np.array(([1, 1, 1], 
                   [1, 1, 1], 
                   [1, 1, 1]), np.float32) / 9
    blank_img = np.zeros(img.shape)

    M = cv2.filter2D(img, -1, BE)
    Ix ,Iy = np.where(M < 255)
    img[Ix, Iy] = 0
    
    return img

def boundary_extract(img, iter_num = 1):

    new_img = np.zeros(img.shape)
    new_img[:,:] = img[:,:]
    for i in range(iter_num):
        new_img = erode(new_img)
    img = img - new_img
    return img

def hole_filling(img, sx = 0, sy = 0):
    ori_img = np.zeros(img.shape)
    C_img = np.zeros(img.shape)
    ori_img[:,:] = img[:,:]
    bx, by = np.where(ori_img < 250)
    wx, wy = np.where(ori_img > 250)
    C_img[bx,by] = 255
    C_img[wx,wy] = 0
    Start = np.zeros(img.shape)
    Start[sx, sy] = 255
    H = np.array(([0, 1, 0], 
                  [1, 1, 1], 
                  [0, 1, 0]), np.float32)
    G = cv2.filter2D(Start, -1, H)
    preG = np.zeros(img.shape)
    
    #End Time
    End = False
    while not End:
        x,y = np.where(G>0)
        G[x,y] = 255
        for ix, iy in zip(x,y):
            if C_img[ix,iy] == 0 :
                G[ix,iy] = 0

        fit = False
        Ix, Iy = np.where(G > 0)
        for ix, iy in zip(Ix,Iy):
            if G[ix,iy] != preG[ix,iy] :
                fit = False
                #print(fit)
                break
            fit = True
        if fit:
            #print(fit)
            ori_img[Ix,Iy] = G[Ix,Iy]        
            End = True
        else:
            preG[:,:] = G[:,:]
            G = cv2.filter2D(G, -1, H)

    return ori_img

def counter(img, sx=0, sy=0):
    ori_img = np.zeros(img.shape)
    ori_img[:,:] = img[:,:]
    Start = np.zeros(img.shape)
    Start[sx, sy] = 255
    H = np.array(([1, 1, 1], 
                  [1, 1, 1], 
                  [1, 1, 1]), np.float32)
    G = cv2.filter2D(Start, -1, H)
    preG = np.zeros(img.shape)
    
    #End Time
    End = False
    while not End:
        x,y = np.where(G > 0)
        G[x,y] = 50
        for ix, iy in zip(x,y):
            if img[ix,iy] == 0 :
                G[ix,iy] = 0

        fit = False
        Ix, Iy = np.where(G > 0)
        for ix, iy in zip(Ix,Iy):
            if G[ix,iy] != preG[ix,iy] :
                fit = False
                #print(fit)
                break
            fit = True
        if fit:
            #print(fit)
            ori_img[Ix,Iy] = G[Ix,Iy]        
            End = True
        else:
            preG[:,:] = G[:,:]
            G = cv2.filter2D(G, -1, H)

    return ori_img

def object_counter(img):
    End_cnt = False
    num_objects = 0
    new_img = np.zeros(img.shape) 
    new_img[:,:] = img[:,:] 
    while not End_cnt:
        x, y = np.where(new_img > 50)
        if x.size > 0:
            new_img = counter(new_img, x[0], y[0])
            num_objects += 1
            print('Object count:',num_objects)
        else:
            End_cnt = True
            break

    return num_objects

def laws_method(img):
    law = np.zeros((9,3,3)) 
    law[0] = np.array(([ 1, 2, 1], 
                       [ 2, 4, 2], 
                       [ 1, 2, 1]), np.float32) / 36
    law[1] = np.array(([1, 0, -1], 
                       [2, 0, -2], 
                       [1, 0, -1]), np.float32) / 12
    law[2] = np.array(([-1, 2,-1], 
                       [-2, 4,-2], 
                       [-1, 2,-1]), np.float32) / 12
    law[3] = np.array(([-1,-2,-1], 
                       [ 0, 0, 0], 
                       [ 1, 2, 1]), np.float32) / 12
    law[4] = np.array(([ 1, 0,-1], 
                       [ 0, 0, 0], 
                       [-1, 0, 1]), np.float32) / 4
    law[5] = np.array(([-1, 2,-1], 
                       [ 0, 0, 0], 
                       [ 1,-2, 1]), np.float32) / 4
    law[6] = np.array(([-1,-2,-1], 
                       [ 2, 4, 2], 
                       [-1,-2,-1]), np.float32) / 12
    law[7] = np.array(([-1, 0, 1], 
                       [ 2, 0,-2], 
                       [-1, 0, 1]), np.float32) / 4
    law[8] = np.array(([ 1,-2, 1], 
                       [-2, 4,-2], 
                       [ 1,-2, 1]), np.float32) / 4
     
    energy_win = np.ones((13,13))
    feature_vector_set = np.zeros((9,img.shape[0],img.shape[1]))
    
    for i in range(9):
        feature_vector_set[i] = cv2.filter2D(img, -1, law[i])
        Mean = cv2.filter2D(feature_vector_set[i], -1, energy_win)/169
        feature_vector_set[i] = (feature_vector_set[i] - Mean)**2
        feature_vector_set[i] = cv2.filter2D(feature_vector_set[i], -1, energy_win)
        feature_vector_set[i] = feature_vector_set[i]**(1/2) 
    return feature_vector_set

def k_means(img):
    pass

def better_classifier(img):
    pass

def flower_replacer(img):
    pass
