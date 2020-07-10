# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:49:38 2020

@author: yuric
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224 ,3)))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Conv2D(32, 3, padding='same', activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(64, 3, padding='same', activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	model.summary()
	return model

def run(model):

	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
    # simple early stopping
	es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
	# prepare iterators
	train_it = datagen.flow_from_directory(batch_size=64,
                                                           directory='dataset/train/',
                                                           shuffle=True,
                                                           target_size=(224, 224),
                                                           class_mode='categorical')
	test_it = datagen.flow_from_directory(batch_size=64,
                                                           directory='dataset/test/',
                                                           shuffle=True,
                                                           target_size=(224, 224),
                                                           class_mode='categorical')
	# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=30, verbose=1)
	# evaluate model
	model.save('model.h5')
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
    
    
# plot diagnostic learning curves
def summarize_diagnostics(history):
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')