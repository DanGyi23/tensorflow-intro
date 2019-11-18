import tensorflow_datasets as tfds

SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="tf_flowers", 
                                                            with_info=True,
                                                            split=list(splits),                                       
                                                            as_supervised=True)

def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return image, label
        
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)