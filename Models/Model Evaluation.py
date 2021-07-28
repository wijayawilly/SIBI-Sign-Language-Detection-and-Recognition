from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from numpy.core.fromnumeric import size
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

# untuk memaksimalkan penggunaan GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


test_dir = 'C:\\Users\\wijay\\Desktop\\a\\SKRIPSI\\Dataset\\test_set_with_bg'
# test_out = "C:\\Users\\wijay\\Desktop\\a\\Project Skripsi\\Dataset_Skripsi\\SIBI_dataset\\test_set_outside"
test_all_black = "C:\\Users\\wijay\\Desktop\\a\\SKRIPSI\\Dataset\\test_all_black"
test_all_black2 = "C:\\Users\\wijay\\Desktop\\a\\Project Skripsi\\Dataset_Skripsi\\Dataset\\train_set_processed\\test_all_black_2"
# test_split = "C:\\Users\\wijay\\Desktop\\a\\Project Skripsi\\Dataset_Skripsi\\train_set\\train_set_processed\\test_set"
# test_lain = "C:\\Users\\wijay\\Desktop\\a\\Project Skripsi\\Dataset_Skripsi\\archive\\asl_alphabet_val"
num_classes = 29

"""Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

Arguments
---------
confusion_matrix: numpy.ndarray
    The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
    Similarly constructed ndarrays can also be used.
class_names: list
    An ordered list of class names, in the order they index the given confusion matrix.
figsize: tuple
    A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
    the second determining the vertical size. Defaults to (10,7).
fontsize: int
    Font size for axes labels. Defaults to 14.
    
Returns
-------
matplotlib.figure.Figure
    The resulting confusion matrix figure
"""


test_data=[]
test_labels=[]

def print_confusion_matrix(confusion_matrix, class_names, figsize = (15,15), fontsize=15):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth', size=15)
    plt.xlabel('Prediction', size=15)
    plt.show()



categories = ["A", "B", "C", "D", "Del" ,"E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Nothing", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]

for category in categories:
    path = os.path.join(test_all_black, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img = load_img(img_path, target_size=(100, 100), color_mode='rgb')
        img = img_to_array(img)
        test_data.append(img)
        test_labels.append(category)

tests = np.array(test_data) / 255.0

# print(tests)


le = LabelEncoder()
tests_labels = le.fit_transform(test_labels)
oh_test_labels = to_categorical(tests_labels, num_classes)

# LOADING THE MODEL AND PREDICT
model = load_model("C:\\Users\\wijay\\Desktop\\a\\SKRIPSI\\CODE\\Dataset Final Result Revised\\Model G\\model_g.h5")

# model.summary()
# model = load_model("C:\\Users\\wijay\\Desktop\\Dataset final result\\Dense 256 DO and BN (Current Best)\\model.h5")


pred_list=[]

prediction = model.predict(tests)

eval = model.evaluate(tests, oh_test_labels)
print(eval)


for i in range(len(prediction)):
    pred_class = np.argmax(prediction[i])
    if pred_class == 0:
        pred_class = "A"
    elif pred_class == 1:
        pred_class = "B"
    elif pred_class == 2:
        pred_class = "C"
    elif pred_class == 3:
        pred_class = "D"
    elif pred_class == 4:
        pred_class = "Del"
    elif pred_class == 5:
        pred_class = "E"
    elif pred_class == 6:
        pred_class = "F"
    elif pred_class == 7:
        pred_class = "G"
    elif pred_class == 8:
        pred_class = "H"
    elif pred_class == 9:
        pred_class = "I"
    elif pred_class == 10:
        pred_class = "J"
    elif pred_class == 11:
        pred_class = "K"
    elif pred_class == 12:
        pred_class = "L"
    elif pred_class == 13:
        pred_class = "M"
    elif pred_class == 14:
        pred_class = "N"
    elif pred_class == 15:
        pred_class = "Nothing"
    elif pred_class == 16:
        pred_class = "O"
    elif pred_class == 17:
        pred_class = "P"
    elif pred_class == 18:
        pred_class = "Q"
    elif pred_class == 19:
        pred_class = "R"
    elif pred_class == 20:
        pred_class = "S"
    elif pred_class == 21:
        pred_class = "Space"
    elif pred_class == 22:
        pred_class = "T"
    elif pred_class == 23:
        pred_class = "U"
    elif pred_class == 24:
        pred_class = "V"
    elif pred_class == 25:
        pred_class = "W"
    elif pred_class == 26:
        pred_class = "X"
    elif pred_class == 27:
        pred_class = "Y"
    elif pred_class == 28:
        pred_class = "Z"
    pred_list.append(pred_class)


pred_list = np.array(pred_list)
# print(pred_list)
test_labels = np.array(test_labels)

cm = confusion_matrix(test_labels,pred_list)
print_confusion_matrix(cm,categories)

print(classification_report(test_labels, pred_list))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(test_labels, pred_list, average='macro')
print('Precision: %f' % precision)

recall = recall_score(test_labels, pred_list, average='macro')
print('Recall: %f' % recall)
