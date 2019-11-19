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