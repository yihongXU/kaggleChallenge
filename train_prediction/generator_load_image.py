import cv2
import os
import numpy as np
from keras.utils import np_utils
import random

# image generator for loading image data #

train_dir = "/home/yxu/Downloads/data/train/"
valid_dir = "/home/yxu/Downloads/data/validation/"
img_rows, img_cols = 299, 299 # Resolution of inputs

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


def generator_dataset(path, batch_size, num_y):

    while 1:
        labels = []
        imgs_addr = []
        for i in range(num_y):
            tmp = listdir_fullpath(path + str(i) + '/')
            imgs_addr += tmp
            labels += [i]*len(tmp)
        x_train = np.zeros([batch_size, img_rows, img_cols, 3],dtype=np.float32)
        y_train = np.zeros([batch_size],dtype=np.uint8)
        num_batch = int(1.*len(labels)/batch_size)
        print num_batch
        for batch in range(num_batch):
            for j in range(batch_size):
                # randomly choose and read image
                i = np.random.choice(range(len(labels)), 1)
                x_train[j] = cv2.resize(cv2.cvtColor(cv2.imread(imgs_addr.pop(i)),cv2.COLOR_BGR2RGB), (img_rows, img_cols))
                y_train[j] = int(labels.pop(i))
            y_train_matrix = np_utils.to_categorical(y_train, num_y)
            yield (x_train, y_train_matrix)

"""
lbls = generator_dataset(train_dir, 20,120)
while 1:
    x,l = lbls.next()
    print l[0].dtype
    cv2.imshow("img", x[0].astype(np.uint8))
    print ('label',l[0,:])
    cv2.waitKey(1)
    
"""


