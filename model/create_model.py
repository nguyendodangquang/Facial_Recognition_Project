import os
import tensorflow as tf

# from tensorflow import keras

from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, BatchNormalization, Conv2D, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

path = 'C:/Users/Dang Quang/Facial_Recognition/image'
num_label = len(os.listdir(path))

train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   rotation_range = 10,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   shear_range=10,
                                   validation_split=0.2)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    path,
    target_size = (150, 150),
    shuffle = True,
    seed = 42,
    batch_size = 16,
    class_mode='categorical',
    subset='training')

validation_generator = val_datagen.flow_from_directory(
    path,
    target_size = (150, 150),
    shuffle=False,
    seed = 42,
    batch_size = 16,
    class_mode='categorical',
    subset='validation')

def model_maker():
  mobilenet = MobileNetV2(input_shape = (150, 150, 3), include_top = False)
  mobilenet.trainable = False

  model = tf.keras.models.Sequential(
      [
       mobilenet,
       GlobalAveragePooling2D(),
#        Dense(1024),
#        Activation('relu'),
#        Dropout(0.3),
#        BatchNormalization(),
          
    #    Dense(512),
    #    Activation('relu'),
    #    Dropout(0.3),
    #    BatchNormalization(),
       Dense(512),
       Activation('relu'),
       Dropout(0.5),
       BatchNormalization(),
       Dense(num_label, activation = "softmax")
      ]
  )
  return model


earlystopping_callback = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=5, verbose=1)

checkpoint_callback = ModelCheckpoint(filepath="./model/face_detect_model.h5",
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

model = model_maker()
model.compile(loss = "categorical_crossentropy",
            metrics = ['accuracy'],
            optimizer = Adam())

history = model.fit(train_generator, validation_data = validation_generator, epochs = 30, callbacks = [checkpoint_callback, earlystopping_callback])