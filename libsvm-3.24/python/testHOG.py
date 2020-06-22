import os
import cv2
from PIL import Image
import numpy as np
import argparse
import random
from svmutil import *
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

if __name__ == '__main__':

    filepath = "/home/dzd/dzd/labwork/face/yaleBExtData"
    for root, dirs, files in os.walk(filepath):
        if root == "/home/dzd/dzd/labwork/face/yaleBExtData":
            continue
        for file in files:
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            if(img.size!=32256):
                img.resize(192,168)
#            img.resize(48,42)
            break
        break
    winSize = (168,192)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9;
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins);
#    hog = cv2.HOGDescriptor()
    hog_descriptor = hog.compute(img)
    hog_data = []
    for i in range(len(hog_descriptor)):
        hog_data.append(hog_descriptor[i][0])
    print(hog_data)
    print(len(hog_data))
