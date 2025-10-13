from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, \
                                    Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_train(path):
    df = pd.read_csv(f'{path}/labels.csv')
    #datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.25)    
    datagen = ImageDataGenerator(validation_split=0.25)    
    train_datagen_flow = datagen.flow_from_dataframe(
            df, directory=f'{path}final_files/',
            x_col='file_name', y_col='real_age',
            target_size=(150, 150),
            batch_size=16,
            class_mode='raw',
            subset='training',
            seed=12345)
    return train_datagen_flow

def load_test(path):
    df = pd.read_csv(f'{path}/labels.csv')
    #datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.25)    
    datagen = ImageDataGenerator(validation_split=0.25)    
    val_datagen_flow = datagen.flow_from_dataframe(
        df, directory=f'{path}final_files/',
        x_col='file_name', y_col='real_age',
        target_size=(150, 150),
        batch_size=16,
        class_mode='raw',
        subset='validation',
        seed=12345)
    return val_datagen_flow

def create_model(input_shape):

    backbone = ResNet50(input_shape=input_shape,#(150, 150, 3),
                    #weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    weights='imagenet', 
                    include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(20, activation='relu'))

    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', 
            metrics=['mae']) 

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=5,
                steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    return model 