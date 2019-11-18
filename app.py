import tensorflow_datasets as tfds
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="tf_flowers", 
                                                            with_info=True,
                                                            split=list(splits),                                       
                                                            as_supervised=True)

def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, (255, 255))
        return image, label
        
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

def augment_data(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
        return image, label

train = train.map(augment_data)

def shuffle_and_batch(dataset):
        ds = dataset.apply( tf.data.experimental.shuffle_and_repeat(buffer_size=218))
        ds = ds.batch(32)
        ds = ds.prefetch(buffer_size=218)

shuffle_and_batch(train)
shuffle_and_batch(validation)
shuffle_and_batch(test)