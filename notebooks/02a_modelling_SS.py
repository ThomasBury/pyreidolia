import numpy as np 
import pandas as pd
import seaborn as sns
import scicomap as sc
import matplotlib as mpl
import yaml
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
import os

#To get a progress bar for long loops:
from tqdm.notebook import trange, tqdm
from time import sleep

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

# Custom package for the project, save all the functions into appropriate sub-packages
from pyreidolia.plot import set_my_plt_style, plot_cloud, plot_rnd_cloud, draw_label_only
from pyreidolia.mask import bounding_box, rle_to_mask, get_binary_mask_sum, mask_to_rle
from pyreidolia.img import get_resolution_sharpness

#%load_ext autoreload
#%autoreload 1

#%aimport pyreidolia

####### 
###############################
def string_print(df):
    return print(df.to_string().replace('\n', '\n\t'))

# Where is my yaml ? "C:/Users/xtbury/Documents/Projects/Pyreidolia/paths.yml"

paths_yml = input("where is the paths.yml config file?")
with open(paths_yml, "r") as ymlfile:
    path_dic = yaml.load(ymlfile, Loader=yaml.FullLoader)

pprint(path_dic)

train_csv_path = path_dic['data']['docs'] + 'train.csv'
train_pq_path = path_dic['data']['docs'] + "train_info_clean.parquet"
train_data = path_dic['data']['train'] 
test_data = path_dic['data']['test'] 
report_path = path_dic['reports']


############################################
X_train2 = np.load(path_dic['data']['docs'] +'X_train2.npy')
print(">> X_train2 memory size: %.2f Gb."
      % (X_train2.nbytes*1.0*10**(-9)))

row_px_data, col_px_data = 700, 467
print("(row_px_data, col_px_data):", row_px_data,",", col_px_data)
row_px_target, col_px_target = 420, 280
print("(row_px_target, col_px_target):", row_px_target,",", col_px_target)

y_train2_SS_Fish = np.empty((0, row_px_target*col_px_target))

for section_number in range(0,41+1):
    #Load from local memory:
    target_section = np.load(path_dic['data']['docs'] +'y_train2_SS_Fish_'+str(section_number)+'.npy')
    #Add to main array:
    y_train2_SS_Fish = np.append(y_train2_SS_Fish, target_section, axis=0)

print(">> y_train2_SS_Fish memory size: %.2f Gb."
      % (y_train2_SS_Fish.nbytes*1.0*10**(-9)))

print(">> y_train2_SS_Fish shape: "+ str(y_train2_SS_Fish.shape)+".")

X_train2 = X_train2.reshape((-1, row_px_data, col_px_data, 1))
print('New shape of X_train2:', X_train2.shape)


######################

inputs=Input(shape = (700, 467, 1), name = "Input")

conv_1 = Conv2D(filters = 30,
                kernel_size = (5, 5),
                padding = 'valid',
                input_shape = (700, 467, 1),
                activation = 'relu')

max_pool_1 = MaxPooling2D(pool_size = (2, 2))

conv_2 = Conv2D(filters = 16,                    
                kernel_size = (3, 3),          
                padding = 'valid',             
                activation = 'relu')

max_pool_2 = MaxPooling2D(pool_size = (2, 2))

flatten = Flatten()

dropout = Dropout(rate = 0.2)

dense_1 = Dense(units = 128,
                activation = 'relu')

dense_2 = Dense(units = 500,activation = 'relu')
dense_3 = Dense(units = 100,activation = 'relu')
dense_4 = Dense(units = 100,activation = 'relu')
dense_5 = Dense(units = 100,activation = 'relu')
dense_6 = Dense(units = 100,activation = 'relu')

dense_7 = Dense(units = row_px_target* col_px_target,
                activation = 'softmax')

############################

x=conv_1(inputs)
x=max_pool_1(x)
x=conv_2(x)
x=max_pool_2(x)

x=dropout(x)
x=flatten(x)
x=dense_1(x)
x=dense_2(x)
x=dense_3(x)
x=dense_4(x)
x=dense_5(x)
x=dense_6(x)
outputs=dense_7(x)

lenet = Model(inputs = inputs, outputs = outputs)


#############################

# Compilation
lenet.compile(loss='BinaryCrossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#############################

training_history_lenet = lenet.fit(X_train2, y_train2_SS_Fish,
                                   validation_split = 0.2,
                                   epochs = 5,
                                   batch_size = 50)














