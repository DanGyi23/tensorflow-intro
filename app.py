import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 32
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)
(raw_train, raw_validation, raw_test), metadata = tfds.load(name="cats_vs_dogs", 
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
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=218)

shuffle_and_batch(train)
shuffle_and_batch(validation)
shuffle_and_batch(test)

def create_model():
        img_inputs = keras.Input(shape=(128, 128, 3))
        conv_1 = keras.layers.Conv2D(32, (3,3), activation=relu)(img_inputs)
        maxpool_1 = keras.layers.Maxpooling2D((2,2))(conv_1)
        conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1)
        maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
        conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2)
        flatten = keras.layers.Flatten()(conv_3)
        dense_1 = keras.layers.Dense(64, activation='relu')(flatten)
        output = keras.layers.Dense(metadata.features['label'].num_classes, activation='softmax')(dense_1)

        model = keras.Model(inputs=img_inputs, outputs=output)

        return model

num_train, num_val, num_test = (
metadata.splits['train'].num_examples * weight/10 
for weight in SPLIT_WEIGHTS
)

steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = round(num_val)//BATCH_SIZE

def train_model(model):
        model.compile(optimizer='adam',
                loss='sparse_categorical_crossenthropy',
                metrics='accuracy')

        tensoboard_callback = keras.callbacks.Tensorboard(log_dir=log_dir, histogram_freq=1)
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
        os.makedirs('training_checkpoints/', exist_ok=True)
        early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)

        history = model.fit(
                train.repeat(),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation.repeat(),
                validation_steps=validation_steps,
                callbacks=[tensorboard_callback,
                        model_checkpoint_callback,
                        early_stopping_checkpoint]
                )

        return history

base_model = keras.applications.InceptionV3(input_shape=(128, 128, 3),
                                        include_top=False, 
                                        weights='imagenet')

def build_model():
        model = keras.Sequential([
                base_model,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(metadata.features['label'].num_classes,
                activation='softmax')
        ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        return model

inception_model = build_model()

