#pr homework libsvm

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
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import io

def gaindata(filepath):
    label = []
    data = []
    winSize = (168,192)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9;
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins);
#    hog = cv2.HOGDescriptor()
    for root, dirs, files in os.walk(filepath):
        print(root)
        datasingle = []
        if root == "/home/dzd/dzd/labwork/face/yaleBExtData":
            continue
        for file in files:
            hog_data = []
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            label.append(int(root[-2:]))
            if(img.size!=32256):
                img.resize(192,168)
#            img.resize(48,42)
            hog_descriptor = hog.compute(img)
            for i in range(len(hog_descriptor)):
                hog_data.append(hog_descriptor[i][0])
#            img.resize(1,48*42)
#            datasingle.append(img[0]/255.0)
            datasingle.append(hog_data)
        data.append(datasingle)
    return label, data

def fivefold_crossvalidation(data, label, c, i):
    param = '-s 0 -t 0 -h 0 -c ' + str(c)
    param = svm_parameter(str(param))
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    acc = 0
    pca = PCA(n_components = 0.95)
    for train_index, test_index in kf.split(data, label):
        x, x_t = data[train_index], data[test_index]
        y, y_t = label[train_index].tolist(), label[test_index]
        x = pca.fit_transform(x)
        x_t = pca.transform(x_t)
        prob  = svm_problem(y, x)
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(y_t, x_t, m)
        acc = acc + p_acc[0]
        print("c_index : " + str(i) + "   train index : " + str(train_index))
    acc = acc/5
    print("The acc is " + str(acc))
    time.sleep(1)
    return acc

def find_best_c(data, label):
    Cs = np.logspace(-5, 5, 50, base = 2)
    print(Cs)
    best_acc = 0
    for i in range(len(Cs)):
        acctemp = fivefold_crossvalidation(data, label, Cs[i], i)
        if(best_acc<acctemp):
            best_acc = acctemp
            best_c = Cs[i]
    return best_c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', default=7, type=int, help='number of total epochs to run in stage one')
    parser.add_argument('--c', default=1, type=int, help='number of total epochs to run in stage one')

    opt = parser.parse_args()
    p = opt.p

    filepath = "/home/dzd/dzd/labwork/face/yaleBExtData"
    label, data = gaindata(filepath)
    
#    plt.figure(figsize=(10, 5))
#    plt.subplots_adjust(hspace=0.1)
#    for n in range(50):
#        plt.subplot(5, 10, n+1)
#        plt.imshow(train_x[n].reshape(48,42),cmap='gray')
##        plt.imshow(train_x[n])
#        plt.axis('off')
#    _ = plt.suptitle("face Example")
#    plt.show()

    im = io.imread('/home/dzd/dzd/labwork/face/yaleBExtData/yaleB01/yaleB01_P00A-005E-10.pgm',as_grey=True)
    normalised_blocks, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8), visualise=True)
    io.imshow(hog_image)
    
    
#    
#    print(len(data))
#    print(len(data[0]))
#    print(len(data[0][0]))
    accuracy = 0
    
    #5-fold-crossvalidation
    datafull = []
    for i in range(38):
        datafull = datafull + data[i]

    datafull = np.array(datafull)
    labelnp = np.array(label)
#    best_c = 1.8899752566172132#2.9719885782738964#,  1.8899752566172132#find_best_c(datafull, labelnp)
#    best_c = find_best_c(datafull, labelnp)
#    best_c = 0.055028641778457996 #n = 200
    best_c = 0.06339039918545794 #0.95 52

    param = '-s 0 -t 0 -h 0 -c ' + str(best_c)
    param = svm_parameter(str(param))

    for j in range(10):
        x = []
        y = []
        x_t = []
        y_t = []

        for i in range(38):
            random.shuffle(data[i])
            x = x + data[i][0:p]
            y = y + label[i*65:i*65+p]
            x_t = x_t + data[i][p:]
            y_t = y_t + label[i*65+p: i*65+65]

        pca = PCA(n_components = 0.95)
#        print(x.isnull().any)
#        print(np.isnan(x).any())
        x = pca.fit_transform(x)
        print("feature num : " + str(len(data[0][0])))
        print("train data num : " + str(len(x)))
        print("variance percent : " + str(sum(pca.explained_variance_ratio_)))
        print("n_components : "+str(pca.n_components_))
        x_t = pca.transform(x_t)
        prob  = svm_problem(y, x)
        m = svm_train(prob, param)

        t_begin = time.time()
        p_label, p_acc, p_val = svm_predict(y_t, x_t, m)
        t_end = time.time()
        accuracy = accuracy + p_acc[0]

    #macc = svm_train(label, datafull, '-s 0 -t 0 -h 0 -v 5')
    print("The best C is " + str(best_c))
    print("The time to predict a image is " + str((t_end-t_begin)/(2470-p*38)) + " seconds\n")
    print("The average accuracy is " + str(accuracy/10) + "!")


