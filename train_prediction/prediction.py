# -*- coding: utf-8 -*-

# predict probabilities of each class and output csv file for kaggle submission. #
#
from keras.layers import Input, Dense, Dropout
import numpy as np
import h5py
import csv
import cv2
from inception_v4 import inception_v4_base
from generator_load_image import preprocess_input
import os
from keras.models import Sequential


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def prediction(model_path, imgs_dir, classes_dir,submission_csv):
    """

    :param model_path: model.h5, keras model containing pretrained weights
    :param imgs_dir: test sets dir
    :param classes_dir: csv file indicating ordering of classes
    :param submission_csv: csv file indicating format of submission
    :return: dict predictions, list headers (used to make predictions_submission.csv
    """
    # read header (labels) csv
    headers = []
    with open(classes_dir, 'rb') as in_f:
        reader = csv.reader(in_f)
        for line_idx, row in enumerate(reader):  # row = list of format [id, breed]
            headers = row
            break
    in_f.close()

    #load model
    inputs = Input((299, 299, 3))
    """
    # Make inception base
    net = inception_v4_base(inputs)


    # Final pooling and prediction
    net_ft = AveragePooling2D((8,8), padding='valid')(net)
    net_ft = Dropout(0.0)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(units=len(headers), activation='softmax')(net_ft)
    model = Model(inputs, predictions_ft, name='inception_v4')
    model.load_weights(model_path)
    
    model = Sequential()
    model.add(AveragePooling2D((8,8), padding='valid',input_shape=(8, 8, 1536)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(len(headers), activation='softmax'))
    """
    model = Sequential()

    model.add(Dropout(0.2,input_shape=(1536,)))
    model.add(Dense(output_dim=120, activation='softmax'))
    model.load_weights(model_path)
    submission = []
    with open(submission_csv, 'rb') as in_f:
        reader = csv.reader(in_f)
        for line_idx, row in enumerate(reader):  # row = list of format [id, breed]
            submission = row
            break
    in_f.close()

    # verification label names = submission names
    for lname in headers:
        if lname not in submission:
            print(lname+" is not a valid name!!!!!")
            break

    # prediction
    predictions = []
    for k, img_addr in enumerate(imgs_dir):#bottlenecks
        """
        img = cv2.resize(cv2.cvtColor(cv2.imread(img_addr), cv2.COLOR_BGR2RGB), (299, 299)).astype(np.float32)
        tmp = np.zeros((1,299,299,3),dtype=np.float32)
        tmp[0] = preprocess_input(img)
        """
        _, file = os.path.split(img_addr)
        tmp = np.load(open(test_path+file[:-4]+'.npy'))
        print tmp.shape
        pdt = model.predict(tmp,batch_size=1)[0]
        pdt = pdt.tolist()
        print max(pdt)
        predt = {key:value for key, value in zip(headers, pdt)}
        print file[:-4]
        predt["id"] = file[:-4]
        print predt
        predictions.append(predt)

    return submission,predictions

test_path = "./test_res_incep/"
imgs_dir = listdir_fullpath(test_path)
out_csv = "/home/yxu/kaggle/submission_20epochs_2w_resinception_nodropout.csv"
submission, predictions = prediction("bottleneck_fc_model_res_Inception_nodrop20.h5",imgs_dir,"/home/yxu/Downloads/data/label.csv","/home/yxu/kaggle/sample_submission.csv")
print submission
print predictions
with open(out_csv, 'w') as out_f:
    f_csv = csv.DictWriter(out_f, submission)
    f_csv.writeheader()
    f_csv.writerows(predictions)
out_f.close()


