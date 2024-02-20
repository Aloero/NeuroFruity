import cv2
from PIL import ImageGrab
import math

import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import SeparableConv2D
from keras.callbacks import ModelCheckpoint

tf.get_logger().setLevel('ERROR')
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

img_width = 640
img_height = 640
classes = 6
area = (1920//2-img_height//2, 1080//2-img_width//2, 1920//2+img_height//2, 1080//2+img_width//2)

model = tf.keras.models.load_model("ChechPoints")


def graphics(from_num, to_num):
    if np.argmax(prediction[from_num:to_num]) % 6 == 0:
        text = "Apple"
        color = (255, 0, 0)
    if np.argmax(prediction[from_num:to_num]) % 6 == 1:
        text = "Bananas"
        color = (0, 255, 0)
    if np.argmax(prediction[from_num:to_num]) % 6 == 2:
        text = "Carrot"
        color = (0, 0, 255)
    if np.argmax(prediction[from_num:to_num]) % 6 == 3:
        text = "Nothing"
        color = (255, 0, 255)
    if np.argmax(prediction[from_num:to_num]) % 6 == 4:
        text = "Pear"
        color = (0, 255, 255)
    if np.argmax(prediction[from_num:to_num]) % 6 == 5:
        text = "Strawberry"
        color = (255, 255, 255)
    cv2.putText(screen_arr,
                str(round(max(prediction[from_num:to_num]), 3)),
                (10 + from_num*15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, from_num*0.041666666666 + 0.5, color, 2)
    cv2.putText(screen_arr,
                text,
                (10 + from_num*15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, from_num*0.041666666666 + 0.5, color, 2)


def process_image(img):
    if img is None:
        print(f"Error: Unable to read image at path {img_path}")
        return None
    if len(img.shape[:2]) != 2:
        print(f"Error: Unable to read image at path {img_path}")
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
    return img_resized_2


cap = cv2.VideoCapture(0)
while True:
    ret, screen = cap.read()                                                    # Читает в BGR
    #screen = ImageGrab.grab(area)                                              # Читает RGB
    #screen = cv2.imread("J:/da/Vit/data_sets/NeuroFruity_v_2/Pear/720.jpg")    # Читает в BGR
    screen_arr = np.array(screen)
    screen_arr = process_image(screen_arr)
    if screen_arr is None:
        continue
    screen_arr = cv2.cvtColor(screen_arr, cv2.COLOR_BGR2RGB)                   # Пероворачивает цвета
    screen_arr = cv2.resize(screen_arr, (img_height, img_width))
    screen_arr_dim = np.expand_dims(screen_arr, axis=0)
    prediction = model.predict(screen_arr_dim, verbose=0)
    prediction = np.reshape(prediction, 18)
    print(prediction)

    graphics(0, 6)
    graphics(6, 12)
    graphics(12, 18)

    screen_arr = cv2.cvtColor(screen_arr, cv2.COLOR_BGR2RGB)
    cv2.imshow('Screen Capture', screen_arr)                                     # Принимает BGR
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()