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