"""
Utility functions for loading the fingernail imaging data
Navid Fallahinia - 06/16/2020
BioRobotics Lab
"""

import glob 
import os  
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image  
import PIL 

def load_data_withIdx(subjIdx_list, filenames, test_ratio = 0.2):
    """
    load the training dataset from images in subject folders and store them 
    in the train folder

    Inputs:
    - subjIdx_list: list of integers for subject indecies [1,17]
    """
    dataset_path = filenames['data']
    train_path = filenames['train']
    test_path = filenames['test']
    eval_path =  filenames['eval']
    raw_image_lists = []
    aligned_image_lists = []
    print('[INFO] Data Processing started ... ')

    for subIdx in subjIdx_list:
        subj_path = dataset_path+'/subj_'+f'{subIdx:02d}'
        print('\tProccessing images for subject_' +f'{subIdx:02d}'+ ' ...', end =" ")

        if(not os.path.isdir(subj_path)):
            print(' no file exists for subject_' +f'{subIdx:02d}'+ ' !!')
            continue

        raw_list = sorted(glob.glob(subj_path + '/raw_images/*.jpg'))
        aligned_list = sorted(glob.glob(subj_path + '/aligned_images/*.jpg'))   

        assert len(raw_list) == len(aligned_list) , "data size mismatch! raw_img:{0}, alig_img:{1}".format(len(raw_list) ,len(aligned_list)) 
               
        raw_image_lists += raw_list
        aligned_image_lists += aligned_list
        print(' Done!')

    data_to_write_train = train_test_split_data(aligned_image_lists, raw_image_lists, test_ratio)
    data_to_write_test = train_test_split_data(data_to_write_train[1], data_to_write_train[3], 0.5)
    print('[INFO] Processing Done! ')
    print("[INFO] Number of train data :{0:4d}, Number of eval data :{1:4d}, Number of test data :{2:4d} "
                . format(len(data_to_write_train[0]), len(data_to_write_test[0]),  len(data_to_write_test[1]))) 
    # data write part 
    write_data(data_to_write_train, data_to_write_test, train_path, eval_path ,test_path)

def train_test_split_data(aligned_image_lists, raw_image_lists, test_ratio ,validation = True ):
    """ spliting data into 3 different set """

    assert len(aligned_image_lists) == len(raw_image_lists), "images have different size"
    mask = list(range(len(aligned_image_lists)))
    mask_train, mask_test = train_test_split(mask, test_size= test_ratio, shuffle=True)

    aligned_lists_train = [aligned_image_lists[i] for i in mask_train]
    aligned_lists_test = [aligned_image_lists[i] for i in mask_test]

    raw_lists_train = [raw_image_lists[i] for i in mask_train]   
    raw_lists_test = [raw_image_lists[i] for i in mask_test] 

    return [aligned_lists_train, aligned_lists_test, raw_lists_train, raw_lists_test]

def write_data(data_to_write_train, data_to_write_test, train_path, eval_path ,test_path):
    """ write datat to a txt file """
    aligned_lists_train = data_to_write_train[0]
    raw_lists_train = data_to_write_train[2]

    aligned_lists_eval = data_to_write_test[0]
    raw_lists_eval = data_to_write_test[2]

    aligned_lists_test = data_to_write_test[1]
    raw_lists_test = data_to_write_test[3]

    filelist = list([train_path, eval_path, test_path])

    for file in filelist:
        aligned_path = os.path.join(file, 'aligned_image')
        raw_path = os.path.join(file, 'raw_image')
        os.mkdir(aligned_path)
        os.mkdir(raw_path)

    # raw image data
    for Idx, train_raw in enumerate(raw_lists_train):
        img = Image.open(train_raw)
        img.save(train_path+'/raw_image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);       
    print('\tTrain raw images saved! ')

    for Idx, eval_raw in enumerate(raw_lists_eval):
        img = Image.open(eval_raw)
        img.save(eval_path+'/raw_image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);  
    print('\tEval raw images saved! ')

    for Idx, test_raw in enumerate(raw_lists_test):
        img = Image.open(test_raw)
        img.save(test_path+'/raw_image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);  
    print('\tTest raw images saved! ')

    # aligned image data
    for Idx, train_aligned in enumerate(aligned_lists_train):
        img = Image.open(train_aligned)
        img.save(train_path+'/aligned_image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);       
    print('\tTrain aligned images saved! ')

    for Idx, eval_aligned in enumerate(aligned_lists_eval):
        img = Image.open(eval_aligned)
        img.save(eval_path+'/aligned_image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);  
    print('\tEval aligned images saved! ')

    for Idx, test_aligned in enumerate(aligned_lists_test):
        img = Image.open(test_aligned)
        img.save(test_path+'/aligned_image/img_'+f'{Idx:04d}.jpg')
        if Idx%100 == 0:
            print('\t%d images are saved'% Idx);  
    print('\tTest aligned images saved! ')