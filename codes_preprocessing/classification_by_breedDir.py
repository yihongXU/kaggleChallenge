import csv
import os, sys
import cv2
from shutil import copyfile
path_csv = "where is your file labels.csv"
path_savedata = "path to save images and their directories"
path_images = "where are your dogs' images"


# make dogs images directory
if not os.path.exists(path_savedata):
    os.makedirs(path_savedata)

with open(path_csv, 'rb') as f:
    reader = csv.reader(f)
    for line_idx, row in enumerate(reader):  # row = list of format [id, breed]
        if line_idx == 0:  # first line is the header
            continue
        else:
            if not os.path.exists(path_savedata+'/'+row[1]):  # creat directroy for each class (breed)
                os.makedirs(path_savedata+'/'+row[1])

            img = cv2.imread(path_images+row[0]+'.jpg')  # return !!!BGR!! format image numpy int array
            cv2.imshow('image underprocess',img)
            cv2.waitKey(1)
            copyfile(path_images+row[0]+'.jpg', path_savedata+'/'+row[1]+'/'+row[0]+'.jpg')
    print ("classification of %5d images by labels, done! " %line_idx)
