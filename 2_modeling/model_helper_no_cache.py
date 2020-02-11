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


def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Shuffle the dataset
    dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    
    # Batch the dataset and map it in a batch, this is vecotrizing the process
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE)

    # Fetch batches in the background while the model is training.
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
    
def plot_training_history_accuracy(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history['accuracy']
    loss = history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')
    
    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()
    
def doPrediction(n, features, X_val, y_val_bin, predictions):
    im = Image.open(X_val.iloc[n]).convert('RGB')
    imgplot = plt.imshow(im)
    plt.show()

    check_pred=pd.DataFrame(features)
    check_pred['Actual labels']=y_val_bin[n]
    check_pred['Predicted labels']=predictions[n]
    check_pred.columns = ['category', 'actual','pred']
    
    check_pred=check_pred[check_pred['actual']==1]
    
    check_pred=check_pred.sort_values(by='pred', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 10))
    #tidy = check_pred.melt(id_vars='category').rename(columns=str.title)
    
    clrs = ['grey' if (x < 0.5) else 'green' for x in check_pred['pred']]
    sns.barplot(x='category', y='pred', palette=clrs, data=check_pred, ax=ax1)
    ax1.set_title('Our predictions for the true features (target label=1)')
    ax1.set_ylabel('Prediction probabilities')
    sns.despine(fig)
    plt.xticks(rotation=90)
    

def plot_per_feature_error_on_zeros(predictions, actual, features):
    
    """return plot of boxplot of predictions when the feature is not present"""
    
    only_zeros = np.array([np.where(y_true == 0, pred, np.nan) for y_true, pred in zip(actual,
                                                                                  predictions)])
    pred_df = pd.DataFrame(only_zeros)
    
    # Name the classes 
    pred_df.columns = features

    # order the whole dataframe and return the column names
    order = pred_df.median().sort_values().keys()
    fig_dims = (16,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    
    sns.boxplot(x='variable', y='value', data=pred_df.melt(), whis=0, showfliers=False, order=order)
    
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    
    return fig, ax


    
def print_layer_trainable(conv_model):
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
        
        
def plot_training_history_recall_on_zero(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc0 = history['accuracy_on_zero']


    # Get it for the validation-set (we only use the test-set).
    val_acc0 = history['val_accuracy_on_zero']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc0, linestyle='-', color='b', label='Training Recall on Zero')

    # Plot it for the test-set.
    plt.plot(val_acc0, linestyle='--', color='r', label='Test Recall on Zero')


    # Plot title and legend.
    plt.title('Training and Test Recall on 0s(=features are non-present)')
    plt.ylim(ymin=0)
    plt.ylim(ymax=1)
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()
    
def plot_training_history_recall_on_one(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc1 = history['accuracy_on_one']


    # Get it for the validation-set (we only use the test-set).
    val_acc1 = history['val_accuracy_on_one']


    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc1, linestyle='-', color='b', label='Training Recall on 1')

    # Plot it for the test-set.
    plt.plot(val_acc1, linestyle='--', color='r', label='Test Recall on 1')

    # Plot title and legend.
    plt.title('Training and Test Recall on 1s(=features are present)')
    plt.ylim(ymin=0)
    plt.ylim(ymax=1)
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()

def plot_training_history_precision_on_one(history):
    # Get the classification precision and loss-value
    # for the training-set.
    prec = history['precision_on_1']


    # Get it for the validation-set (we only use the test-set).
    val_prec = history['val_precision_on_1']


    # Plot the precision for the training-set.
    plt.plot(prec, linestyle='-', color='b', label='Training Preicision on 1')

    # Plot it for the test-set.
    plt.plot(val_prec, linestyle='--', color='r', label='Test Precision on 1')

    # Plot title and legend.
    plt.title('Training and Test Precision on 1s(=features are present)')
    plt.ylim(ymin=0)
    plt.ylim(ymax=1)
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()
    
    
def plot_training_history_loss(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    loss = history['loss']

    # Get it for the validation-set (we only use the test-set).

    val_loss = history['val_loss']

    # Plot the accuracy and loss-values for the training-set.

    plt.plot(loss, 'o', color='b', label='Training Loss')
    
    # Plot it for the test-set.
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Loss (Binary cross-entropy)')
    plt.ylim(ymin=0)
    plt.ylim(ymax=1)
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()
    
def weighted_bce(y_true, y_pred):
    """ Custom loss function """
    weight = 10
    y_true=K.cast(y_true, 'float32')
    y_pred=K.cast(y_pred, 'float32')
    weights = (y_true * weight) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def grouped_roc(y_val, y_pred):
    """
    The macro average finds the ROC of every feature and then averages all the lines
    the micro roc flattens all the y_val and y_preds and makes one 
    ROC curve as if it was one image with x number of features
    """

    n_classes = y_val.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_pred[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_val.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute the macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points and sum them all together
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        if not np.isnan(np.interp(all_fpr, fpr[i], tpr[i])).any():
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # plots
    lw = 2
    fig, ax = plt.subplots()
    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    
    return fig, ax


def precision_recall_f1(y_val_bin, predictions_test):

    predictions_conv=copy.deepcopy(predictions_test)

    predictions_conv[predictions_conv >= 0.5] =1 
    predictions_conv[predictions_conv < 0.5] =0


    true_pos=np.logical_and(predictions_conv, y_val_bin)
    true_pos_df=pd.DataFrame(true_pos)
    true_pos_df.columns = features
    true_pos_per_feature=true_pos_df.sum(axis=0) 

    true_ones=pd.DataFrame(y_val_bin)
    true_ones.columns = features
    true_ones_per_feature=true_ones.sum(axis=0) 

    recall=true_pos_per_feature/true_ones_per_feature
    recall_sorted=recall.sort_values()

    precision_df=pd.DataFrame(predictions_conv)
    precision_df.columns = features
    all_1_predictions_per_feature=precision_df.sum(axis=0) 
    precision=true_pos_per_feature/all_1_predictions_per_feature


    combined=pd.concat([recall, precision], axis=1)
    combined.columns= ['recall', 'precision']
    #combined=combined.sort_values(by='recall')
    combined['features']=combined.index

    #F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    combined['f1']= 2*combined['recall']*combined['precision']/(recall+precision)
    combined=combined.sort_values(by='f1')

    combined_melt= combined.melt('features', var_name='cols', value_name='vals')

    fig_dims = (22,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    g = sns.scatterplot(x='features', y="vals", hue='cols', data=combined_melt, alpha=0.6)
    g.set_xlabel('Fashion features', fontsize=30)
    g.set_ylabel('Recall and precision ratios', fontsize=30)
    g.set_title('F1, Precision and recall per feature', fontsize=40)
    g.set_xticklabels([],rotation=90)
    
    return 


def plot_per_feature_error(predictions, actual, features, diff=False):
    
    """return plot of boxplot of predictions when the feature is active"""
    
    only_ones = np.array([np.where(y_true == 1, pred, np.nan) for y_true, pred in zip(actual,
                                                                                  predictions)])
    pred_df = pd.DataFrame(only_ones)
    
    # Name the classes 
    pred_df.columns = features

    # order the whole dataframe and return the column names
    order = pred_df.median().sort_values().keys()
    fig_dims = (22,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    
    melted=pred_df.melt()
    #melted_with_medians = melted.merge(medians_per_feature, on='variable', how='left')
    #melted_with_medians['avg_prediction']=melted_with_medians['median']>0.5
    #my_pal = {"g", "r"}
    #boxprops = dict(linestyle='-', linewidth=4, color='grey', alpha=.1)
    
    sns.boxplot(x='variable', y='value', data=pred_df.melt(), whis=0, showfliers=False, order=order, boxprops=dict(alpha=.1))
    #sns.boxplot(x='variable', y='value', boxprops=boxprops, data=melted_with_medians, whis=0, hue='avg_prediction', palette=my_pal, showfliers=False, order=order)
    #boxprops=dict(alpha=.3)

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    
    ax.set_xlabel('Fashion features', fontsize=30)
    ax.set_ylabel('Raw probabilities if the label is ON', fontsize=30)
    ax.set_title('Raw probabilities per feature if the label is ON', fontsize=40)
    ax.set(xticklabels=[])

    
    if diff:
        only_zeros = np.array([np.where(y_true == 0, pred, np.nan) for y_true, pred in zip(actual,
                                                                                  predictions)])
        pred_df2 = pd.DataFrame(only_zeros)
        pred_df2.columns = features
        
        pred_df=pred_df.reindex(pred_df.median().sort_values().index, axis=1)
        pred_df2=pred_df2.reindex(pred_df.median().sort_values().index, axis=1)
        
        
        median_ones=pred_df.median()
        median_zeros=pred_df2.median()
        diff=median_ones-median_zeros
        diff=pd.DataFrame(diff)

        sns.scatterplot(data=diff)

    return fig, ax
