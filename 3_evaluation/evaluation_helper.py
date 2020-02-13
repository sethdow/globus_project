import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import copy

#General
import matplotlib.pyplot as plt
from collections import Counter

# Machine learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans

def get_feature_counts():    
    df_clean = pd.read_csv('../1_cleaning/metadata_cleaned2.csv')

    df_clean['features'] = df_clean['features'].apply(eval)

    list_of_all_features = [item for l in df_clean.features for item in l]

    feature_df = pd.DataFrame.from_dict(Counter(list_of_all_features), orient='index').reset_index()

    feature_df.rename(columns={'index':'feature', 0:'count'}, inplace=True)

    feature_df.sort_values('count', inplace=True)
    
    return feature_df

def feature_frequency():
    """Fetch all the features and then return a series with frequency of each"""
    
    df_clean = pd.read_csv('../1_cleaning/metadata_cleaned2.csv')

    df_clean['features'] = df_clean['features'].apply(eval)

    list_of_all_features = [item for l in df_clean.features for item in l]

    feature_df = pd.DataFrame.from_dict(Counter(list_of_all_features), orient='index').reset_index()

    feature_df.rename(columns={'index':'feature', 0:'count'}, inplace=True)

    feature_df.sort_values('count', inplace=True)
    
    feature_df=get_feature_counts()

    feature_df['feature_frequency'] = np.array(feature_df['count']/np.max(feature_df['count']))

    feature_df_simple = feature_df.set_index(keys='feature').drop(labels=['count'], axis=1)

    return feature_df_simple['feature_frequency']

# Recall Functions

def sum_true_positives(predictions_binarized, y_val_bin, features):
    """ Create a series with the sum of true positives per feature"""

    # Count True positives
    true_pos = np.logical_and(predictions_binarized, y_val_bin)
    
    # Turn true positive count into a dataframe
    true_pos_df = pd.DataFrame(true_pos, columns=features)
    
    # Sum the number of true positives per feature
    true_pos_per_feature = true_pos_df.sum(axis=0) 
    
    return true_pos_per_feature

def sum_false_positives(predictions_binarized, y_val_bin, features):
    """ Create a series with the sum of false positives per feature"""

    # Count True positives
    false_pos = np.logical_and((~y_val_bin.astype('bool')).astype('int'), predictions_binarized)
    
    # Turn true positive count into a dataframe
    false_pos_df = pd.DataFrame(false_pos, columns=features)
    
    # Sum the number of true positives per feature
    false_pos_per_feature = false_pos_df.sum(axis=0)
    
    return false_pos_per_feature
    

def sum_false_negatives(predictions_binarized, y_val_bin, features):
    """ Create a series with the sum of true positives per feature"""

    # Count True positives
    false_neg = np.logical_and((~predictions_binarized.astype('bool')).astype('int'), y_val_bin)
    
    # Turn true positive count into a dataframe
    false_neg_df = pd.DataFrame(false_neg, columns=features)
    
    # Sum the number of true positives per feature
    false_neg_per_feature = false_neg_df.sum(axis=0) 
    
    return false_neg_per_feature

    
def recall_series(true_positives_per_feature, false_negatives):
    """ find the recall and return it as a series with features as labels"""
        
    # Count the recall using True positives and all positives
    recall = true_positives_per_feature/(positives_per_feature)
    
    return recall

def create_recall_precision_frequency_df(predictions_test, y_val_bin, features):
    
    # Turn the raw predictions to 1 or 0
    predictions_binarized=(predictions_test >= 0.5).astype('int')
    
    # Get the total TP per feature
    true_positives_per_feature = sum_true_positives(predictions_binarized, y_val_bin, features)

    # Get the FN per feature
    false_negatives = sum_false_negatives(predictions_binarized, y_val_bin, features)

    # Get the FP per feature
    false_positives = sum_false_positives(predictions_binarized, y_val_bin, features)

    # Calculate recall
    recall = true_positives_per_feature/(false_negatives + true_positives_per_feature)

    # Calculate precitions
    precision = true_positives_per_feature/(false_positives + true_positives_per_feature)

    # Get the frequency of each feature
    feature_freq = feature_frequency()

    # Create the feature data frame with the three measures
    combined=pd.concat([recall, precision, feature_freq], axis=1, sort = True)
    combined.columns= ['recall', 'precision', 'frequency']
    
    combined.dropna(inplace = True)
    
    return combined

def cluster_elbow(recall_precision_frequency,cluster_mark=4):
    # empty list for inertia values
    inertia = []

    for i in range(1,10):
        # instantiating a kmeans model with i clusters
        # init=kmeans++ is default
        kmeans = KMeans(n_clusters=i)

        # fitting the model to the data
        kmeans.fit(recall_precision_frequency)

        # appending the inertia of the model to the list
        inertia.append(kmeans.inertia_)

        # Knowing from the data that we have three samples distributions
        # let's save the inertia for the case k=4
        if i == cluster_mark:
            elbow = kmeans.inertia_
    # creating a list with the number of clusters
    number_of_clusters = range(1,10)
    plt.plot(number_of_clusters, inertia)
    plt.plot(cluster_mark, elbow, 'ro', label='Elbow')
    plt.legend()
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
def cluster_features(recall_precision_frequency, num_clusters=4):
    """Take a dataframe with evaluation features and cluster it, return a labeled df"""
    
    km4=KMeans(n_clusters=num_clusters)

    km4.fit(recall_precision_frequency)
    
    recall_precision_frequency['labels']=km4.labels_
    
    return recall_precision_frequency

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

def precision_recall_f1(y_val_bin, predictions_test, features):

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
    
    return fig,ax

def plot_per_feature_error(predictions, actual, features):
    
    """return boxplot of predictions when the feature is on"""
    
    # Get on the feature that exist in each image
    only_ones = np.array([np.where(y_true == 1, pred, np.nan) for y_true, pred in zip(actual,
                                                                                  predictions)])
    pred_df = pd.DataFrame(only_ones)
    
    # Name the classes 
    pred_df.columns = features

    # Order the whole dataframe and return the column names
    order = pred_df.median().sort_values().keys()
    
    # Set figure dimensions
    fig_dims = (22,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    
    melted=pred_df.melt()

    
    sns.boxplot(x='variable', 
                y='value', 
                data=pred_df.melt(), 
                color='green', 
                whis=0, 
                showfliers=False, 
                order=order, 
                boxprops=dict(alpha=.1))

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    
    ax.set_xlabel('Fashion features', fontsize=30)
    ax.set_ylabel('Raw probabilities if the label is ON', fontsize=30)
    ax.set_title('Raw probabilities per feature if the label is ON', fontsize=40)
    ax.set(xticklabels=[])


    return fig, ax

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

def plot_per_feature_both(predictions, actual, features):
    
    """return plot of boxplot of predictions when the feature is active"""
    
    only_ones = np.array([np.where(y_true == 1, pred, np.nan) for y_true, pred in zip(actual,
                                                                                  predictions)])
    only_zeros = np.array([np.where(y_true == 0, pred, np.nan) for y_true, pred in zip(actual,
                                                                                  predictions)])
    
    # Order by differences
    diff=np.nanmedian(only_ones, axis=0)-np.nanmedian(only_zeros, axis=0)
    pred_diff=pd.DataFrame(diff.reshape(1,-1))
    pred_diff.columns = features
    order_diff = pred_diff.median().sort_values().keys()
    
    pred_one = pd.DataFrame(only_ones)
    pred_zero = pd.DataFrame(only_zeros)
    
    # Name the classes 
    pred_one.columns = features
    pred_zero.columns = features


    # order the whole dataframe and return the column names
    order = pred_one.median().sort_values().keys()
    fig_dims = (22,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    
    sns.boxplot(x='variable', y='value', 
                data=pred_one.melt(), 
                whis=0, 
                showfliers=False, 
                order=order_diff, 
                ax=ax,
                linewidth=2.5, 
                color=sns.xkcd_rgb["seafoam"])
    
    sns.boxplot(x='variable', y='value', 
                data=pred_zero.melt(), 
                whis=0, 
                showfliers=False, 
                order=order_diff, 
                ax=ax, 
                color=sns.xkcd_rgb["off white"])
    
    ax.set_xlabel('Fashion features', fontsize=30)
    ax.set_ylabel('Model Success Rate', fontsize=30)
    ax.set_title('Model Prediction on features when present and absent', fontsize=40)
    ax.set_xticklabels([],rotation=90)
    return fig, ax

# Comparative evaluation
def grab_model_name(path):
    """Get the path name and parse out the name of the model"""
    return path.split('/')[-1].split('.')[0]

def get_column_or_return_nan(frame, column):
    """ either retrieve the the column or return NaN if it doesn't exist"""
    
    return frame.get(column, pd.Series(index=frame.index, name=column))

def f1_score(frame, recall_column='val_accuracy_on_one', precision_column='val_precision_on_1'):
    """Calculate the f1 score and return the column """
    return 2 * frame[recall_column] * frame[precision_column]/\
                 (frame[recall_column]+frame[precision_column])

def frame_extraction(frame, model_name):
    """ retrieve the relevant columns and do some processing and return the clean frame """
    
    # Define the columns that we want to return
    columns = ['epoch','val_precision_on_1', 'val_accuracy_on_zero', 'val_accuracy_on_one','model', 'f1']
    
    # Create the F1 score
    frame['f1'] = f1_score(frame)
    
    # Get each of the desired columns, return NaN if it doesn't exist
    for column in columns:
        frame[column] = get_column_or_return_nan(frame, column)
    
    
    # Fetch the model name
    frame['model'] = model_name
    
    # Get the columns used for comparison
    reduced = frame[columns]
    
    return reduced
