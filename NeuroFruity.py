import tensorflow as tf
import pathlib
import numpy as np
from joblib import Parallel, delayed
import cv2
import math

import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import UpSampling2D
import os


tf.get_logger().setLevel('ERROR')
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
np.set_printoptions(threshold=np.inf)

# Установка политики смешанной точности
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# __init__
epochs = 100
batch_size = 2
img_width = 640
img_height = 640


checkpoint = ModelCheckpoint("ChechPoints", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks = [checkpoint]


# Подготовка данных Preproceccing изображений (вырезает середину с пропорциями)
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: {img_path}")
        return None
    original_height, original_width = img.shape[:2]
    k1 = img_height / original_height
    k2 = img_width / original_width
    if k1 > k2:
        img_resized = cv2.resize(img, (math.ceil(original_width * k1), img_height), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = cv2.resize(img, (img_width, math.ceil(original_height * k2)), interpolation=cv2.INTER_LINEAR)
    img_resized = np.array(img_resized)
    if k1 > k2:
        img_resized_2 = img_resized[:, int((img_resized.shape[1] / 2) - (img_width / 2)):int(
            (img_resized.shape[1] / 2) + (img_width / 2)), :]
    else:
        img_resized_2 = img_resized[int((img_resized.shape[0] / 2) - (img_height / 2)):int(
            (img_resized.shape[0] / 2) + (img_height / 2)), :]
    img_resized_2 = cv2.cvtColor(img_resized_2, cv2.COLOR_BGR2RGB)
    return img_resized_2


# Выполнения операций параллельно
directory = "J:/da/Vit/data_sets/NeuroFruity_v_2"
dataset_dir = pathlib.Path(directory)
img_paths_fruits = list(dataset_dir.glob("*/*.jpg"))
input_data = np.array((Parallel(n_jobs=-1)(delayed(process_image)(str(img_path)) for img_path in img_paths_fruits)), dtype=np.uint8)
num_png = []
for root, dirs, files in os.walk(directory):
    png_count = sum(1 for file in files if file.endswith('.jpg'))
    num_png.append(png_count)
y_fruits = []
for i in range(1, len(num_png)):
    y_fruits.extend([i-1] * num_png[i])
output_data = np.array(y_fruits, dtype=np.uint8)

print('Вход:', input_data.shape)
print('Выход:', output_data.shape)

# Соединение входа и выхода
dataset = tf.data.Dataset.from_tensor_slices((input_data, (output_data, output_data, output_data)))
val_size = int(0.2 * len(input_data))
dataset = dataset.shuffle(buffer_size=len(input_data))
train_dataset = dataset.skip(val_size).batch(batch_size)
val_dataset = dataset.take(val_size).batch(batch_size)

# cache
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# Архитектура
#n = 3
w = 1
r = 1


def conv(x, filters, kernel_size, strides, padding):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = swish(x)
    return x


def Conv_layers(x, shortcut, filters_2):
    if shortcut:
        original_x = x

    x = conv(x, filters_2, kernel_size=3, strides=1, padding='same')
    x = conv(x, filters_2, kernel_size=3, strides=1, padding='same')

    if shortcut:
        x = Add()([x, original_x])
    return x


def main_block(x, filters, n, shortcut):
    x = conv(x, filters, kernel_size=1, strides=1, padding='same')
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    l1 = Conv_layers(x1, shortcut, filters//2)
    l2 = Conv_layers(x1, shortcut, filters//2)
    l3 = Conv_layers(x1, shortcut, filters//2)

    if n == 6:
        l4 = Conv_layers(x1, shortcut, filters//2)
        l5 = Conv_layers(x1, shortcut, filters//2)
        l6 = Conv_layers(x1, shortcut, filters//2)

    if n == 6:
        x = tf.concat([x1, x2, l1, l2, l3, l4, l5, l6], axis=-1)
    else:
        x = tf.concat([x1, x2, l1, l2, l3], axis=-1)
    x = conv(x, filters, kernel_size=1, strides=1, padding='same')
    return x


def detect_conv_hard(x, filters, strides):
    x1 = conv(x, filters=filters, kernel_size=3, strides=strides, padding='same')
    x1 = main_block(x1, filters=filters, shortcut=False, n=3)
    return x1


def detect_conv_lite(x, filters, strides):
    x1 = conv(x, filters=filters, kernel_size=3, strides=strides, padding='same')
    return x1


def detect(x):
    x1 = Flatten()(x)
    x1 = Dense(1000)(x1)
    x1 = ReLU()(x1)
    x1 = Dense(1000)(x1)
    x1 = ReLU()(x1)
    x1 = Dense(1000)(x1)
    x1 = Dense(len(num_png) - 1, activation='softmax')(x1)
    return x1


input_layer = Input(shape=(img_height, img_width, 3))
x = preprocessing.Rescaling(1. / 255)(input_layer)

#x = preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width))(x)

x = conv(x, filters=64*w, kernel_size=3, strides=2, padding='same')    # Input: 512x512x3
x = conv(x, filters=128*w, kernel_size=3, strides=2, padding='same')   # Input: 256x256x64*w

x = main_block(x, filters=128*w, shortcut=True, n=3)                                       # Input: 128x128x128*w
x = conv(x, filters=256*w, kernel_size=3, strides=2, padding='same')   # Input: 128x128x128*w

x = main_block(x, filters=256*w, shortcut=True, n=6)                                       # Input: 64x64*256*w

layer1_1 = x

x = conv(x, filters=512*w, kernel_size=3, strides=2, padding='same')    # Input: 64x64x256

x = main_block(x, 256*w, shortcut=True, n=6)                                    # Input: 32x32x512

layer1_2 = x

x = conv(x, filters=512*w*r, kernel_size=3, strides=2, padding='same')      # Input: 32x32x512

x = main_block(x, 512*w*r, shortcut=True, n=3)                                     # Input: 16x16x512

x = conv(x, filters=512*w*r, kernel_size=1, strides=1, padding='same')

copy_x1 = x

x = MaxPooling2D()(x)

copy_x2 = tf.image.resize(x, [20, 20])

x = MaxPooling2D()(x)

copy_x3 = tf.image.resize(x, [20, 20])

x = MaxPooling2D()(x)

x = tf.image.resize(x, [20, 20])

x = Add()([copy_x1, copy_x2, copy_x3, x])

layer2_3 = x

x = UpSampling2D(size=(2, 2))(x)

x = tf.concat([x, layer1_2], axis=-1)

x = main_block(x, 512*w, shortcut=False, n=3)

layer2_2 = x

x = UpSampling2D(size=(2, 2))(x)

x = tf.concat([x, layer1_1], axis=-1)

x = main_block(x, 256*w, shortcut=False, n=3)

detect1 = x

x = conv(x, filters=256*w, kernel_size=3, strides=2, padding='same')

x = tf.concat([x, layer2_2], axis=-1)

x = main_block(x, 512*w, shortcut=False, n=3)

detect2 = x

x = conv(x, filters=512*w, kernel_size=3, strides=2, padding='same')

x = tf.concat([x, layer2_3], axis=-1)

x = main_block(x, 512*w*r, shortcut=False, n=3)

detect3 = x

detect3 = detect_conv_hard(detect3, 256, strides=1)
detect3 = detect_conv_lite(detect3, 152, strides=2)
detect3 = detect_conv_lite(detect3, 48, strides=1)
detect3 = detect(detect3)

detect2 = detect_conv_hard(detect2, 256, strides=1)
detect2 = detect_conv_lite(detect2, 152, strides=2)
detect2 = detect_conv_lite(detect2, 48, strides=2)
detect2 = detect(detect2)

detect1 = detect_conv_hard(detect1, 128, strides=2)
detect1 = detect_conv_lite(detect1, 88, strides=2)
detect1 = detect_conv_lite(detect1, 48, strides=2)
detect1 = detect(detect1)

model = Model(inputs=input_layer, outputs=(detect3, detect2, detect1))

model.compile(
    optimizer=Adam(learning_rate=0.0004),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
)

#model.save("C:/Users/Vit/PycharmProjects/PythonProject1/Neuro/Temp")

# visualize
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.title('Training and Validation loss')
plt.show()

































