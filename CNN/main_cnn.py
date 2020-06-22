import os
os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, datasets, models
import os
import cv2
from PIL import Image
import numpy as np
import argparse
import random
import time
from sklearn.model_selection import StratifiedKFold

def gaindata(filepath):
    label = []
    data = []
    for root, dirs, files in os.walk(filepath):
        datasingle = []
        if root == "/home/dzd/dzd/labwork/face/yaleBExtData":
            continue
        for file in files:
            img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
            if ( int(root[-2:]) <= 13 ):
                label.append(int(root[-2:])-1)
            else:
                label.append(int(root[-2:])-2)
            if(img.size!=32256):
                img.resize(192,168)
#            img.resize(48,42)
            datasingle.append(img)
#            print(datasingle)
#            print('a')
        data.append(datasingle)
    return label, data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', default=7, type=int, help='number of total epochs to run in stage one')
    opt = parser.parse_args()
    p = opt.p

    filepath = "/home/dzd/dzd/labwork/face/yaleBExtData"
    
#    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
#    
#    print(type(train_x))
#    print(train_y)
#    print('train_x shape:', train_x.shape, 'test_x shape:', test_x.shape)    
#    # (50000, 32, 32, 3), (10000, 32, 32, 3)
#    print('train_y shape:', train_y.shape, 'test_y shape:', test_y.shape)
    

    (label,data) = gaindata(filepath)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    mydataset = []
    for i in range(38):
        random.shuffle(data[i])
        for j in range(p):
            mydataset.append([data[i][j] , label[i*65+p]])
#        train_x = train_x + data[i][0:p]
#        train_y = train_y + label[i*65:i*65+p]
        test_x = test_x + data[i][p:]
        test_y = test_y + label[i*65+p: i*65+65]
        
#    print(mydataset)
    random.shuffle(mydataset)
    train_x = []
    train_y = []
    for i in range(len(mydataset)):
        train_x.append(mydataset[i][0])
        train_y.append(mydataset[i][1])
        
    print(np.array(train_x).shape)
    train_x = np.array(train_x).reshape(len(train_x),192,168,1)
    test_x = np.array(test_x).reshape(len(test_x), 192,168,1)
    train_y = np.array(train_y).reshape(len(train_y), 1)
    test_y = np.array(test_y).reshape(len(test_y), 1)
    
#    print(type(train_x))
#    plt.figure(figsize=(5, 3))
#    plt.subplots_adjust(hspace=0.1)
#    for n in range(15):
#        plt.subplot(3, 5, n+1)
##        plt.imshow(train_x[n].reshape(192,168),cmap='gray')
#        plt.imshow(train_x[n])
#        plt.axis('off')
#    _ = plt.suptitle("face Example")
#    plt.show()

    train_x, test_x = train_x / 255.0, test_x / 255.0
    print('train_x shape:', train_x.shape, 'test_x shape:', test_x.shape)
    print('train_y shape:', train_y.shape, 'test_y shape:', test_y.shape)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(192,168, 1)))
#    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(38, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=32, epochs=20)

    test_loss, test_acc = model.evaluate(test_x, test_y)
    test_acc 
    
    
    
    print("Saving model to disk \n")
    mp = "/home/dzd/dzd/labwork/face/CNN/model/CNN_model2.h5"
    model.save(mp)
