# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:28:53 2020

@author: yuric
"""
import os 
from shutil import copyfile, copy,rmtree
import random

dataset_home = 'dataset/'

def create_directories ():
    
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
    	# create label subdirectories
    	labeldirs = ['roses/', 'dandelion/','daisy/','tulips/','sunflowers/']
    	for labldir in labeldirs:
    		newdir = dataset_home + subdir + labldir
    		os.makedirs(newdir, exist_ok=True)
            
    print('Create directories OK...')
    
def create_test_train_folder(data_dir,class_names):
    random.seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25
    # copy training dataset images into subdirectories
    src_directory = data_dir
    for name in (class_names):
    	for file in data_dir.glob(name+'/*'):
    		src = str(file)
    		dst_dir = 'train/'
    		rand = random.random()
    		#print(rand)
    		if rand < val_ratio:
    			 dst_dir = 'test/'
    			 dst = dataset_home + dst_dir  + name 
    			 copy(src, dst)
    		
    		dst = dataset_home + dst_dir  + name 
    		copy(src, dst)
    print("Create train and test folders")
    