# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:16:13 2020

@author: yuri crotti
"""

# import TensorFlow tf.keras
import tensorflow as tf
from tensorflow import keras

# Libraries aux
import numpy as np
import pathlib

import utils as utl
import ia_model as iam
import prediction as pred

print(tf.__version__)
AUTOTUNE =  tf.data.experimental.AUTOTUNE

# download data_set flowes photos / if you already have the directory, you can comment this line
datadir = keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos',untar=True)
datadir = pathlib.Path(datadir)

# count imagem flowers / if you already have the directory, you can comment this line
image_count = len(list(datadir.glob("*/*.jpg")))
print("Numero de Imagens : {}".format(image_count))

# get class names/ if you already have the directory, you can comment this line
CLASS_NAME = np.array([item.name for item in datadir.glob('*') if item.name != 'LICENSE.txt'])
print(CLASS_NAME)

# create_directories by flowers / if you already have the directory, you can comment this line
utl.create_directories()

# create train and test folders / if you already have the directory (train,test), you can comment this line
utl.create_test_train_folder(datadir,CLASS_NAME)

#define model 
model = iam.define_model()

# run model
iam.run(model)

# get imgs roses for predict example
roses = list(datadir.glob('roses/*'))

# run predict 
pred.run_example(roses[2])