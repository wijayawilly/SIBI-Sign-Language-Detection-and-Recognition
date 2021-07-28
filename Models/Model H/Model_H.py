from tensorflow. keras import models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import keras as k
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers
import pickle
import cv2
from PIL import Image
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.utils import to_categorical
from scipy import linalg
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
# from keras.applications.vgg16 import preprocess_input
# from keras import regularizers


# untuk memaksimalkan penggunaan GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


data_vary = 'C:\\Users\\wijay\\Desktop\\a\\SKRIPSI\\CODE\\Dataset\\Dataset_Finale'

num_classes = 29
epoch = 100

data=[]
labels=[]


test_data=[]
test_labels=[]



categories = ["A", "B", "C", "D", "Del" ,"E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Nothing", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]

# Reading Training data
for category in categories:
    path = os.path.join(data_vary, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img = load_img(img_path, target_size=(100, 100), color_mode='rgb')
        img = img_to_array(img)
        data.append(img)
        labels.append(category)


le = LabelEncoder()
labels = le.fit_transform(labels)

data = np.array(data) / 255.0
# data = np.array(data)


labels = np.array(labels)

# print(data)



(train_images, val_images, train_labels, val_labels) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42, shuffle=True)

train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)


num_filters=64
ac='relu'
adm=Adam(lr=0.0001)

model = models.Sequential()
model.add(layers.Conv2D(num_filters, (3, 3),activation=ac,input_shape=(100, 100, 3), padding='same'))

model.add(layers.Conv2D(num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(2*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.Conv2D(2*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(4*num_filters, (3, 3), activation=ac, padding='same'))
model.add(layers.Conv2D(4*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.Conv2D(4*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(8*num_filters, (3, 3), activation=ac, padding='same',))
model.add(layers.Conv2D(8*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.Conv2D(8*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(8*num_filters, (3, 3), activation=ac, padding='same'))
model.add(layers.Conv2D(8*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.Conv2D(8*num_filters, (3, 3),activation=ac, padding='same'))
model.add(layers.MaxPooling2D(pool_size= (2,2), strides=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation=ac))
model.add(layers.BatchNormalization())
model.add(layers.Dense(units= num_classes, activation='softmax'))



model.compile(loss="categorical_crossentropy", optimizer=adm, metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights = True, verbose=1)
csv_logger = CSVLogger('training.log', separator=',', append=False)
history = model.fit(train_images, train_labels, batch_size=128,epochs=epoch, validation_data=(val_images,val_labels), callbacks=[es,csv_logger], verbose=1)
model.save("model_h.h5")
# model.summary()

plt.style.use("ggplot")
print(history.history)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.show()


plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.show()


