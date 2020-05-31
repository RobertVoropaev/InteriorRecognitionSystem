# Created by RobertVoropaev 05.2020

############################################ Arg default #############################################


class def_config:

    main_data_dir = "../data/ADE20K_encoded/"
    callbacks_dir = "../callbacks/"

    img_shape = 256
    classes_num = 31

    batch_size = 16
    epoch_num = 100
    train_coef = 1
    learning_rate = 0.0001

    last_activation = "sigmoid"
    loss_function = "categorical_crossentropy"

    gpu_memory_limit = 0.8
    cpu_threads_num = 4

    callbacks_monitor = "val_jaccard_coef"
    callbacks_data_format = "%m.%d_%H-%M"
    file_name = "DefName"
    
    is_load = False
    argparse_is_on = True
    
    
############################################ Argparse #############################################


if def_config.argparse_is_on:
    import argparse

    parser = argparse.ArgumentParser('UNet model')

    ### Dirs

    parser.add_argument('-md', '--main_data_dir', type=str, required=False,
                        default=def_config.main_data_dir,
                        help="Главная папка с папками train и val")

    parser.add_argument('-cd', '--callbacks_dir', type=str, required=False,
                        default=def_config.callbacks_dir,
                        help="Папка, в которую будут записываться данные каждой модели")

    ### Input

    parser.add_argument('-is', '--img_shape', type=int, required=False,
                        default=def_config.img_shape,
                        help="Размер входного слоя сети")

    parser.add_argument('-cn', '--classes_num', type=int, required=False,
                        default=def_config.classes_num,
                        help="Количество классов = количество каналов выходного слоя")

    ### Train

    parser.add_argument('-bs', '--batch_size', type=int, required=False,
                        default=def_config.batch_size,
                        help="Количество объектов в каждом batch на обучении и валидации")

    parser.add_argument('-en', '--epoch_num', type=int, required=False,
                        default=def_config.epoch_num,
                        help="Количество эпох обучения")

    parser.add_argument('-tc', '--train_coef', type=float, required=False,
                        default=def_config.train_coef,
                        help="Доля объектов обучающей выборки, которые будут использоваться в одной эпохе")

    parser.add_argument('-lr', '--learning_rate', type=float, required=False,
                        default=def_config.learning_rate,
                        help="Скорость обучения модели")

    ### Output

    parser.add_argument('-la', '--last_activation', type=str, required=False,
                        default=def_config.last_activation,
                        help="Функция активации выходного слоя")

    parser.add_argument('-lf', '--loss_function', type=str, required=False,
                        default=def_config.loss_function,
                        help="Функция потерь")

    ### Memory limit

    parser.add_argument('-gl', '--gpu_memory_limit', type=float, required=False,
                        default=def_config.gpu_memory_limit,
                        help="Максимальная доля выделенной GPU памяти")

    parser.add_argument('-ct', '--cpu_threads_num', type=int, required=False,
                        default=def_config.cpu_threads_num,
                        help="Максимальное количество потоков CPU")

    ### Callbacks settings

    parser.add_argument('-cm', '--callbacks_monitor', type=str, required=False,
                        default=def_config.callbacks_monitor,
                        help="Метрика сохранения лучшего callback'а")

    ### Load

    parser.add_argument('-il', '--is_load', action="store_true", required=False,
                        help="Флаг загрузки предобученной модели")

    parser.add_argument('-wp', '--weight_path', type=str, required=False,
                        help="Путь до файла .h5 с весами")

    args = parser.parse_args()

else:
    args = def_config()


############################################ Import ###############################################


# System
import os
import sys
import datetime

# Base
import numpy as np
import cv2
import random

# Keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose
from keras.layers import Dropout, BatchNormalization, Concatenate
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger

# Preprocessing
from keras.utils import Sequence, to_categorical
from keras.utils.vis_utils import plot_model

# Backend
import tensorflow as tf
from keras import backend as K
from tensorflow.python.client import device_lib

# Seed
seed = 99
np.random.seed(seed)
random.seed(seed)


############################################ Session limit ###########################################


config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=args.cpu_threads_num,
                        inter_op_parallelism_threads=args.cpu_threads_num)
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_limit
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# GPU check list
GPU_list = [x for x in device_lib.list_local_devices() 
            if x.device_type == 'GPU' or x.device_type == "GPU"]
print(GPU_list)

if not tf.test.is_gpu_available():
    raise OSError("GPU not found")


############################################ Path #################################################


main_data_dir = args.main_data_dir

train_dir = main_data_dir + "train/"
val_dir = main_data_dir + "val/"

img_train_dir = train_dir + "img/"
mask_train_dir = train_dir + "mask/"

img_val_dir = val_dir + "img/"
mask_val_dir = val_dir + "mask/"

# Callbacks

file_name = sys.argv[0].split(".")[-2]
callbacks_dir = args.callbacks_dir

try:
    os.mkdir(callbacks_dir)
except OSError:
    pass

now = datetime.datetime.now()
callbacks_dir_name = file_name + now.strftime("_" + def_config.callbacks_data_format) + "/"

callbacks_full_dir = callbacks_dir + callbacks_dir_name
try:
    os.mkdir(callbacks_full_dir)
except OSError:
    pass


############################################ Size #################################################


train_size = len(os.listdir(path=train_dir + "img/"))
val_size = len(os.listdir(path=val_dir + "img/"))
print("Train size: " + str(train_size))
print("Val size: " + str(val_size))


############################################ Config ###############################################


img_shape = args.img_shape
batch_size = args.batch_size
classes_num = args.classes_num

epoch_num = args.epoch_num
train_coef = args.train_coef
learning_rate = args.learning_rate

loss_function = args.loss_function
last_activation = args.last_activation

is_load = args.is_load
if is_load:
	weight_path = args.weight_path
else:
	weight_path = None


with open(callbacks_dir + callbacks_dir_name + "/" + "config.txt", "w") as f:
    if def_config.argparse_is_on:
        args_str = str(args).lstrip("Namespace(").rstrip(')')
        args_arr = args_str.split(", ")
        f.write("\n".join(args_arr))
    else:
        f.write("Def_config\n")

callbacks_monitor = args.callbacks_monitor

############################################ Metric ##############################################


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


############################################ Loss #################################################

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)


############################################ Generator ############################################

def data_gen(img_dir, mask_dir, classes_num, batch_size):
    img_folder = img_dir
    mask_folder = mask_dir

    img_list = os.listdir(img_folder)
    random.shuffle(img_list)
    img_dir_size = len(img_list)

    for i in range(len(img_list)):
        img_list[i] = img_list[i].split(".")[0]  # отделяем имя от формата

    counter = 0
    while (True):
        img = np.zeros((batch_size, img_shape, img_shape, 3)).astype('float')
        mask = np.zeros((batch_size, img_shape, img_shape, classes_num)).astype("uint8")

        for i in range(counter, counter + batch_size):
            train_img = cv2.imread(img_folder + '/' + img_list[i] + ".jpg") / 255.
            train_img = cv2.resize(train_img, (img_shape, img_shape))

            img[i - counter] = train_img

            train_mask = cv2.imread(mask_folder + '/' + img_list[i] + ".png", cv2.IMREAD_GRAYSCALE)
            train_mask = cv2.resize(train_mask, (img_shape, img_shape), interpolation=cv2.INTER_NEAREST)
            train_mask = train_mask.reshape(img_shape, img_shape, 1)
            train_mask = to_categorical(train_mask, num_classes=classes_num)

            mask[i - counter] = train_mask

        counter += batch_size

        if counter + batch_size >= img_dir_size:
            counter = 0
            random.shuffle(img_list)

        yield img, mask


train_gen = data_gen(img_train_dir, mask_train_dir, classes_num=classes_num, batch_size=batch_size)
val_gen = data_gen(img_val_dir, mask_val_dir, classes_num=classes_num, batch_size=batch_size)


############################################ Model ################################################

def get_model(img_shape, classes_num, last_activation):
    model_name = "Unet0_model"
    
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

    block8_output = Conv2D(classes_num, (1, 1), padding="same", activation=last_activation)(block7_conv3)

    return Model(inputs=block0_input, outputs=block8_output), model_name


############################################ Callbacks ############################################

def get_callbacks(dir_path, callbacks_monitor):
    # лучшие веса
    best_w = ModelCheckpoint(dir_path + "best_w.h5",
                             monitor=callbacks_monitor,
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto',
                             period=1
                             )

    # последние веса
    last_w = ModelCheckpoint(dir_path + "last_w.h5",
                             monitor=callbacks_monitor,
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='auto',
                             period=1
                             )

    # сохраняет историю обучения
    logger = CSVLogger(dir_path + "logger.csv",
                       append=False)

    return [best_w, last_w, logger]


############################################ Compile #################################################

if last_activation != 'sigmoid' and last_activation != 'softmax':
    raise ValueError("Incorrect last activation :" + last_activation)

model, model_name = get_model(None, classes_num, last_activation)

plot_model(model=model, to_file=callbacks_full_dir + model_name + ".png", show_shapes=True, dpi=200)


if is_load:
    if not weight_path:
        raise ValueError("Don't load weight_path")
    model.load_weights(weight_path)

if loss_function == 'categorical_crossentropy':
    pass
elif loss_function == 'dice_loss':
    loss_function = dice_loss
elif loss_function == 'jaccard_loss':
    loss_function = jaccard_loss
else:
    raise ValueError("Incorrect loss function :" + loss_function)

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss=loss_function,
              metrics=["accuracy", dice_coef, jaccard_coef])

############################################ Fit ##################################################

model.fit_generator(train_gen,
                    epochs=epoch_num,
                    steps_per_epoch=int(train_coef * train_size) // batch_size,
                    validation_data=val_gen,
                    validation_steps=val_size // batch_size,
                    verbose=1,
                    callbacks=get_callbacks(callbacks_full_dir, callbacks_monitor)
                    )

###################################################################################################
