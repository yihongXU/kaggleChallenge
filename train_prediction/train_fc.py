# -*- coding: utf-8 -*-
# transfer learning frozen feature extraction layers and train a softmax classifier #
from keras.models import Sequential
from keras.optimizers import SGD,Adam,RMSprop
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from generator_load_image import generator_dataset
from sklearn.metrics import log_loss
import numpy as np
import h5py
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score
import cv2

train_dir = "/home/yxu/Downloads/data/train/"
valid_dir = "/home/yxu/Downloads/data/validation/"
test_dir = "/home/yxu/Downloads/data/test/"


def model(img_rows, img_cols, color_type=1, num_classes=None, dropout_keep_prob=0.2):
    '''
    from
    Inception V4 Model for Keras

    Model Schema is based on
    https://github.com/kentsommer/keras-inceptionV4

    ImageNet Pretrained Weights
    Theano: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5
    TensorFlow: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    '''

     # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model

    model = Sequential()

    model.add(Dropout(dropout_keep_prob,input_shape=(1536,)))
    model.add(Dense(output_dim=num_classes, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon= 0.1)
    #rmsprop = RMSprop()
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 299, 299 # Resolution of inputs
    channel = 3
    num_classes = 120
    batch_size = 20
    nb_epoch = 50

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    #X_train, Y_train,X_valid, Y_valid= load_cifar10_data(img_rows, img_cols)

    train_generator = generator_dataset(train_dir,batch_size,num_classes)
    valdiation_generator = generator_dataset(valid_dir,batch_size,num_classes)

    # Load our model
    model = model(img_rows, img_cols, channel, num_classes, dropout_keep_prob=0.0)

     # Start Fine-tuning
    #model.fit(X_train, Y_train,
    #          batch_size=batch_size,
    #          nb_epoch=nb_epoch,
    #          shuffle=True,
    #          verbose=1,
    #          validation_data=(X_valid, Y_valid),
    #          )

    #model.fit_generator(train_generator,
    #          steps_per_epoch=980, # int(1.*len(labels)/batch_size),
    #          nb_epoch=nb_epoch,
    #          shuffle=False,
    #          verbose=1,
    #          validation_data=valdiation_generator,
    #          validation_steps=48,
    #          )
    # here's a more "manual" example
    for e in range(nb_epoch):
        print('Epoch', e)
        batches_train = 0
        while 1:
            if batches_train%100==0 :
                print '**************************** '+ str(batches_train)+' ****************************'
            x_train = np.load(open('res_incep/bottleneck_features_resincp_x_' + str(batches_train) + '.npy'))
            y_train = np.load(open('res_incep/bottleneck_features_resincp_y_' + str(batches_train) + '.npy'))
            model.fit(x_train, y_train, batch_size=batch_size,epochs=1,verbose=0)
            """
            x_batch = np.load(open('xception_inceptionv3_data/bottleneck_features_xcep_x_'+str(batches_train)+'.npy'))
            y_batch = np.load(open('xception_inceptionv3_data/bottleneck_features_xcep_y_' + str(batches_train) + '.npy'))
            model.fit(x_batch, y_batch,batch_size=batch_size,epochs=1,verbose=0)
            """
            batches_train += 1
            if batches_train > 932:
                loss_incp_res = []
                batch_val = 0
                while 1:
                    X_valid = np.load(open('res_incep/bottleneck_features_val_resincp_x_' + str(batch_val) + '.npy'))
                    Y_valid = np.load(open('res_incep/bottleneck_features_val_resincp_y_' + str(batch_val) + '.npy'))
                    # Make predictions
                    y_predt = model.predict(X_valid,batch_size=batch_size)
                    loss_incp_res.append(log_loss(Y_valid,y_predt))
                    """
                    X_valid = np.load(open('xception_inceptionv3_data/bottleneck_features_val_v3_x_' + str(batch_val) + '.npy'))
                    Y_valid = np.load(open('xception_inceptionv3_data/bottleneck_features_val_v3_y_' + str(batch_val) + '.npy'))
                    # Make predictions
                    y_predt = model.predict(X_valid, batch_size=batch_size)
                    loss_v3 += log_loss(Y_valid, y_predt)
                    """
                    batch_val +=1
                    if batch_val > 48 : break
                print ("log loss_inception_res: ", loss_incp_res)
                print ("mean log loss inception_res: ", np.mean(np.array(loss_incp_res)))
                break

        if e%10 == 0:
            model.save('bottleneck_fc_model_res_Inception_nodrop'+str(e)+'.h5')

    model.save('bottleneck_fc_model_res_Inception_nodrop.h5')
