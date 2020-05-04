###################################################################################################

#System
import os
import shutil

#Base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from skimage.io import imread, imshow, imsave

#Keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose
from keras.layers import Dropout,BatchNormalization, Concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger

#Preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical

#Models
from keras.applications.vgg16 import VGG16

#GPU
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
K.set_session(sess)

#Seed
seed = 99
np.random.seed(seed)
random.seed(seed)

###################################################################################################

main_data_dir = "../data/ADE20K_encoded/"

train_dir = main_data_dir + "train/"
val_dir = main_data_dir + "val/"

img_train_dir = train_dir + "img/"
mask_train_dir = train_dir + "mask/"

img_val_dir = val_dir + "img/"
mask_val_dir = val_dir + "mask/"

callbacks_dir = "../checkpoints/"
callbacks_dir_name = "model11"

###################################################################################################

train_size = len(os.listdir(path = train_dir + "img/"))
val_size = len(os.listdir(path = val_dir + "img/"))
print("Train size: " + str(train_size))
print("Val size: " + str(val_size))

###################################################################################################

img_shape = 256
batch_size = 4
num_classes = 32

epoch_num = 50
train_coef = 0.1 # доля объектов тренировочной выборки на каждой эпохе
learning_rate = 0.0001

###################################################################################################

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smooth) / 
            (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
            
def jaccard_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((intersection + smooth) / 
            (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))
            
###################################################################################################

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
    
def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)
    
###################################################################################################

def data_gen(img_dir, mask_dir, num_classes, batch_size):
    img_folder = img_dir
    mask_folder = mask_dir
    num_classes = num_classes
    
    img_list = os.listdir(img_folder)
    random.shuffle(img_list)
    img_dir_size = len(img_list)
    
    for i in range(len(img_list)):
        img_list[i] = img_list[i].split(".")[0] #отделяем имя от формата
        
    counter = 0
    while (True):
        img = np.zeros((batch_size, img_shape, img_shape, 3)).astype('float')
        mask = np.zeros((batch_size, img_shape, img_shape, num_classes)).astype("uint8")

        for i in range(counter, counter + batch_size):  

            train_img = cv2.imread(img_folder + '/' + img_list[i] + ".jpg") / 255.
            train_img =  cv2.resize(train_img, (img_shape, img_shape))

            img[i - counter] = train_img 

            train_mask = cv2.imread(mask_folder + '/' + img_list[i] + ".png", cv2.IMREAD_GRAYSCALE)
            train_mask = cv2.resize(train_mask, (img_shape, img_shape), interpolation = cv2.INTER_NEAREST)
            train_mask = train_mask.reshape(img_shape, img_shape, 1)
            train_mask = to_categorical(train_mask, num_classes=num_classes)
            
            mask[i - counter] = train_mask

        counter += batch_size
        
        if (counter + batch_size >= img_dir_size):
            counter = 0
            random.shuffle(img_list)
                  
        yield img, mask
        
train_gen = data_gen(img_train_dir,mask_train_dir, num_classes=num_classes, batch_size=batch_size)
val_gen = data_gen(img_val_dir,mask_val_dir, num_classes=num_classes, batch_size=batch_size)

###################################################################################################

def get_model(img_shape, num_classes):
    block0_input = Input(shape=(img_shape, img_shape, 3))

    block1_conv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(block0_input)
    block1_conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(block1_conv1)
    block1_conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")(block1_conv2)
    block1_pool1 = MaxPool2D(2)(block1_conv3)

    block2_conv1 = Conv2D(128, (3, 3), padding="same", activation="relu")(block1_pool1)
    block2_conv2 = Conv2D(128, (3, 3), padding="same", activation="relu")(block2_conv1)
    block2_conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(block2_conv2)
    block2_pool1 = MaxPool2D(2)(block2_conv3)

    block3_conv1 = Conv2D(256, (3, 3), padding="same", activation="relu")(block2_pool1)
    block3_conv2 = Conv2D(256, (3, 3), padding="same", activation="relu")(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3), padding="same", activation="relu")(block3_conv2)
    block3_pool1 = MaxPool2D(2)(block3_conv3)

    block4_conv1 = Conv2D(512, (3, 3), padding="same", activation="relu")(block3_pool1)
    block4_conv2 = Conv2D(512, (3, 3), padding="same", activation="relu")(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3), padding="same", activation="relu")(block4_conv2)
    block4_upsa1 = UpSampling2D(2, interpolation="bilinear")(block4_conv3)
    
    block5_conc1 = Concatenate()([block3_conv3, block4_upsa1])
    block5_conv1 = Conv2D(256, (3, 3), padding="same", activation="relu")(block5_conc1)
    block5_conv2 = Conv2D(256, (3, 3), padding="same", activation="relu")(block5_conv1)
    block5_conv3 = Conv2D(256, (3, 3), padding="same", activation="relu")(block5_conv2)
    block5_upsa1 = UpSampling2D(2, interpolation="bilinear")(block5_conv3)

    block6_conc1 = Concatenate()([block2_conv3, block5_upsa1])
    block6_conv1 = Conv2D(128, (3, 3), padding="same", activation="relu")(block6_conc1)
    block6_conv2 = Conv2D(128, (3, 3), padding="same", activation="relu")(block6_conv1)
    block6_conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(block6_conv2)
    block6_upsa1 = UpSampling2D(2, interpolation="bilinear")(block6_conv3)

    block7_conc1 = Concatenate()([block1_conv3, block6_upsa1])
    block7_conv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(block7_conc1)
    block7_conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(block7_conv1)
    block7_conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")(block7_conv2)
    
    block8_output = Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(block7_conv3)

    return Model(inputs=block0_input, outputs=block8_output)

model = get_model(None, num_classes)

###################################################################################################

def get_callbacks(dir_name, callbacks_dir="checkpoints/"):
    dir_path = callbacks_dir + dir_name  + "/"
    os.mkdir(dir_path)
    
    #лучшие веса
    best_w = ModelCheckpoint(dir_path + "best_w.h5", 
                             monitor="val_loss",
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1
                            )

    #последние веса
    last_w = ModelCheckpoint(dir_path + "last_w.h5",
                             monitor="val_loss",
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=1
                            )
    
    #сохраняет историю обучения
    logger = CSVLogger(dir_path + "logger.csv",
                       append=True)

    return [best_w, last_w, logger]
    
###################################################################################################

model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss=jaccard_loss, 
              metrics=["accuracy", dice_coef, jaccard_coef])
    
model.fit_generator(train_gen, 
                    epochs=epoch_num,
                    steps_per_epoch=int(train_coef*train_size)//batch_size,
                    validation_data=val_gen, 
                    validation_steps=val_size//batch_size,
                    verbose=1,
                    callbacks=get_callbacks(callbacks_dir_name, callbacks_dir)
                    )
                    
###################################################################################################          
