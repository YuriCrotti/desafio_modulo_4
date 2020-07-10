# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:02:20 2020

@author: yuri crotti
"""

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example(img_example):
	# load the image
	img = load_image(img_example)
	# predict the class
	model = load_model('model.h5')
	result = model.predict(img)
	print(result[0])
 
