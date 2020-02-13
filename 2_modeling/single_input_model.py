#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
import helper_functions as hp
import seaborn as sns
import pickle
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import backend as K
import sys
sys.path.insert(1, '../2_model_pipeline')
from model_helper import *
from sklearn.metrics import roc_curve, auc

# # Set Parameters and directories to save models

number_of_images = 'all'
EPOCHS = 10
# Test train split, plus the features that are used to binarize the data
X_train, X_val, y_train_bin, y_val_bin, features = train_test_split_custom(number_of_images)

name_of_model = 'model1_sample_sizeall_epoch50_dense2_losswbc'

base_path = '/home/ubuntu/efs/models/'

# Directories for chackpoint
checkpoint_path = base_path + 'Checkpoints/' + name_of_model + '.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# For training history
training_history_path = base_path + 'Training_history/' + name_of_model + '.pickle'

# For model saving once training has ended
saved_model_path = base_path + 'Saved_models/' + name_of_model + '.h5'

# setup the VGG
vgg = VGG19(include_top=True, weights='imagenet')
transfer_layer = vgg.get_layer('block5_pool')
conv_model = Model(inputs=vgg.input,outputs=transfer_layer.output)
conv_model.trainable = False
num_label = y_train_bin.shape[1]

def create_model():
    # Start a new Keras Sequential model.
    model = Sequential()

    # Add the convolutional part of the VGG16 model from above.
    model.add(conv_model)

    # Flatten the output of the VGG16 model because it is from a
    # convolutional layer.
    model.add(Flatten())

    # Add a dense (aka. fully-connected) layer.
    # This is for combining features that the VGG16 model has
    # recognized in the image.
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(num_label, activation='sigmoid'))

    # Settings
    LR = 1e-5
    optimizer = Adam(lr=LR)
    loss = weighted_bce
    metrics = [accuracy_on_one, accuracy_on_zero]

    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=metrics)

    return model


new_model = create_model()


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# create the datasets
train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)


# Fit the new model and start training

history = new_model.fit(train_ds,
                    epochs=EPOCHS,
                    steps_per_epoch=100,
                    validation_steps=2,
                    validation_data=create_dataset(X_val, y_val_bin),
                    callbacks=[cp_callback])


# Save the training history
pickle.dump(history.history, open(training_history_path, 'wb'))

# Save the model
new_model.save(saved_model_path)
