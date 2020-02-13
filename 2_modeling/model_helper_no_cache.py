#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image


import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Lambda

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunk of 1024 observations


def train_test_split_custom_2(sample_size, random_state=44):
  
    df_clean = pd.read_csv('../1_cleaning/metadata_cleaned2.csv')
    
    df_clean['features'] = df_clean['features'].apply(eval)
    
    df_small = df_clean if sample_size == 'all' else df_clean.sample(sample_size)

    X_train, X_val, y_train, y_val = train_test_split(df_small['image_path'], df_small['features'], test_size=0.2, random_state=random_state)
    
    X_train2, X_val2, y_train2, y_val2 = train_test_split(df_small['hierarchy_2'], df_small['features'], test_size=0.2, random_state=random_state)
    
    
    # Fit the multi-label binarizer on the training set
    mlb = MultiLabelBinarizer()
    mlb.fit(df_clean.features)

    # transform the targets of the training and test sets
    y_train_bin = mlb.transform(y_train)
    y_val_bin = mlb.transform(y_val)
    
    X_train2, X_val2 = one_hot_encode(X_train2, X_val2)
    
    return X_train, X_val, X_train2, X_val2, y_train_bin, y_val_bin, mlb.classes_

def train_test_split_custom(sample_size, csv = '../1_cleaning/metadata_cleaned.csv', random_state=44):
  
    df_clean = pd.read_csv(csv)
    
    df_clean['features'] = df_clean['features'].apply(eval)
    
    df_small = df_clean if sample_size == 'all' else df_clean.sample(sample_size)

    X_train, X_val, y_train, y_val = train_test_split(df_small['image_path'], df_small['features'], test_size=0.2, random_state=random_state)

    # Fit the multi-label binarizer on the training set
    mlb = MultiLabelBinarizer()
    mlb.fit(df_clean.features)

    # transform the targets of the training and test sets
    y_train_bin = mlb.transform(y_train)
    y_val_bin = mlb.transform(y_val)
    
    return X_train, X_val, y_train_bin, y_val_bin, mlb.classes_

def one_hot_encode(X_train_hier, X_val_hier):
    
    ohe = sklearn.preprocessing.OneHotEncoder()
    
    categories = np.array(list(set(X_train_hier.to_list()+X_val_hier.to_list()))).reshape(-1,1)
    
    ohe.fit(categories)

    X_train_hier=ohe.transform(X_train_hier.astype(str).values.reshape(-1,1)).todense()
    
    X_val_hier=ohe.transform(X_val_hier.astype(str).values.reshape(-1,1)).todense()
            
    return X_train_hier, X_val_hier


# X_train2_sc=ohe.transform(X_train2.astype(str).values.reshape(-1,1)).todense()
# X_val2_sc=ohe.transform(X_val2.astype(str).values.reshape(-1,1)).todense()

def create_dataset_multi(filenames, hierarchy, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
        
        
        Tried from here: https://github.com/tensorflow/tensorflow/issues/20698
    """
    
    # Create input and output datasets
    dataset_input = tf.data.Dataset.from_tensor_slices((filenames, hierarchy))
    dataset_output = tf.data.Dataset.from_tensor_slices((labels))
    
    # Batch the dataset and map it in a batch, this is vecotrizing the process
    dataset_input = dataset_input.map(parse_function_multi, num_parallel_calls=AUTOTUNE)
    
    #Combine the inputs and the outputs
    dataset = tf.data.Dataset.zip((dataset_input, dataset_output))
    
    # Shuffle the dataset
    dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    
    dataset = dataset.batch(BATCH_SIZE)

    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

def parse_function_multi(path, hierarchy):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(path)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    
    return image_normalized, hierarchy


def create_dataset(filenames, labels):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset

    

def parse_function(path,label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(path)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label

# Custom functions for metrics
def accuracy_on_one(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        sum_true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        
        sum_all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        acc_on_one = sum_true_positives / (sum_all_positives + K.epsilon())
        return acc_on_one
    
    
def accuracy_on_zero(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        
        y_true_flipped = 1 - y_true
        y_pred_flipped = 1- y_pred
        
        sum_true_negatives = K.sum(K.round(K.clip( y_true_flipped * y_pred_flipped, 0, 1)))
        
        sum_all_negatives = K.sum(K.round(K.clip(y_true_flipped, 0, 1)))
        
        acc_on_zero = sum_true_negatives / (sum_all_negatives + K.epsilon())
        return acc_on_zero

    
def precision_on_1(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    
def print_layer_trainable(conv_model):
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

    
def weighted_bce(y_true, y_pred):
    """ Custom loss function """
    weight = 10
    y_true=K.cast(y_true, 'float32')
    y_pred=K.cast(y_pred, 'float32')
    weights = (y_true * weight) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce
