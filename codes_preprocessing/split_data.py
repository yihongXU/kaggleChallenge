import numpy as np
import csv
import os
from shutil import copyfile

# split data into trainset, validation set and test set by a predefined ratio in parameters. #

#  parameters
validation_ratio = 10 # ration of validation set in percentage
test_ratio = 10 # ration of test set in percentage
data_addr = "/home/yxu/Downloads/Images/"
output_dir = "/home/yxu/Downloads/data/"


# read dirs
dir_list = next(os.walk(data_addr))
labels = []
headers = []

# make directorys
if not os.path.exists(output_dir+"validation"):
    os.makedirs(output_dir+"validation")

if not os.path.exists(output_dir+"test"):
    os.makedirs(output_dir+"test")

for i, sub_dir in enumerate(dir_list[1]):  # iteration for every subset (breed)
    label = {sub_dir:i}
    labels.append(label)
    headers.append(sub_dir)
    img_dir = dir_list[0]+sub_dir  # image/label/
    files = os.listdir(img_dir)

    num = len(files)
    print ('num',num)
    num_test = int(num/100. * test_ratio)
    num_valid = int(num/100. * validation_ratio)
    print num_valid
    print num_test
    test_set = []
    validation_set = []

    # random choose images to test set and validation set
    print min(num_test, num_valid)
    for j in range(min(num_test, num_valid)):
        index = np.random.choice(range(len(files)),1)
        test_set.append(files.pop(index))
        index = np.random.choice(range(len(files)),1)
        validation_set.append(files.pop(index))

    for j in range(num_test-len(test_set)):
        index = np.random.choice(range(len(files)),1)
        test_set.append(files.pop(index))

    for j in range(num_valid-len(validation_set)):
        index = np.random.choice(range(len(files)),1)
        validation_set.append(files.pop(index))

    # create sub dirs
    if not os.path.exists(output_dir + "validation/" + str(i)):
        os.makedirs(output_dir + "validation/" + str(i))

    if not os.path.exists(output_dir + "test/" + str(i)):
        os.makedirs(output_dir + "test/" + str(i))

    if not os.path.exists(output_dir + "train/" + str(i)):
        os.makedirs(output_dir + "train/" + str(i))

    # save images to directory

    for fname in test_set:
        copyfile(img_dir+'/'+fname, output_dir + "test/" + str(i) + '/' + fname)

    for fname in validation_set:
        copyfile(img_dir+'/'+fname, output_dir + "validation/" + str(i) + '/' + fname)

    for fname in files:
        copyfile(img_dir+'/'+fname, output_dir + "train/" + str(i) + '/' + fname)


with open(output_dir+'label.csv', 'w') as out_f:
    f_csv = csv.DictWriter(out_f, headers)
    f_csv.writeheader()
    f_csv.writerows(labels)

out_f.close()
