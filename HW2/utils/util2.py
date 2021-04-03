import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt 

def Row_grad():
    pass

def Col_grad():
    pass

def Edge_Det_1st(img, K = 2):
    #Default Sobel
    Kx = np.array(([-1, 0, 1], [-K, 0, K], [-1, 0, 1]), np.float32)
    Ky = np.array(([1, K, 1], [0, 0, 0], [-1, -K, -1]), np.float32)
                
    Ix = cv2.filter2D(img,-1, Kx)
    Iy = cv2.filter2D(img,-1, Ky)
    G = np.hypot(Ix, Iy)
    G = (G/G.max()*255)

    theta = np.arctan2(Iy, Ix)
                                            
    return (G, theta)

def Edge_Det_2st(img):
    pass

def Edge_Det_Canny(img):
    pass

def Edge_Cris(img):
    pass



