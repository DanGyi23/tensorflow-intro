import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import os
import cv2                
import numpy as np
from random import shuffle
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_DIR = '/Users/dangyi/Downloads/dogs-vs-cats/train'
TEST_DIR = '/Users/dangyi/Downloads/dogs-vs-cats/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
        word_label = img.split('.')[-3]
        if word_label == 'cat': return [1,0]
        elif word_label == 'dog': return [0,1]

def create_train_data():
        training_data = []
        for img in tqdm(os.listdir(TRAIN_DIR)):
                label = label_img(img)
                path = os.path.join(TRAIN_DIR,img)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                training_data.append([np.array(img),np.array(label)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

def process_test_data():
        testing_data = []
        for img in tqdm(os.listdir(TEST_DIR)):
                path = os.path.join(TEST_DIR,img)
                img_num = img.split('.')[0]
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                testing_data.append([np.array(img), img_num])
                
        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        return testing_data