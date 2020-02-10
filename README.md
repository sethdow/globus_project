

# Title:

Propulsion Academy Final Project DSWD-2020-1

GLOBUS

Authors: Valeria Polozun and Seth Dow

## Project File Structure:

### Data

Storage of project data and logging is in the efs instance provided by Propulsion. It has the following structure

**root:** efs/models
file naming conventions: These are verbose, they generally specifiy whether the model was multiinput, how many images, epochs, weight of loss functions, dense layers, open layers in the VGG, as well as a the number of neurons in each dense layer. For more detail see the readme in that 2_modeling folder in the git repo

Saved_model: This is where the .h5 fully trained models reside.
Training_history: We saved .pickle and .csv files with the metrics in this folder. You can navigate to the 2_modeling directory to see which metrics were used and the naming convention for models. These files were saved after training.
Checkpoints: We also kept weights saved after each epoch of training in case of network failure. These were kept as .ckpt files. 

### Main Git Repo

0_exploration: In this folder we explore several things that impacted our understanding of the project. The multilabel ROC curve, which behaves in interesting ways when calculated for a multilabel model. The frequency of each feature. How the features in our data mapped to the current GLOBUS heirarchy. Then there are files for Seth and Valeria that explore more general things such as number of photos per feature

1_cleaning: This folder creates csvs that have two purposes:

    1. connect the model to the photos via valid file paths. 
    2. make a cutoff for the features that will be fed to the model
    
    All csv created here are used in the 2_modeling folder to feed the models. The original metadata_cleaned was done by the first group. We created metadata_cleaned2, which has features that occur >500 times. We also created metadata_cleaned3 which has all the features, none were removed.

**ONLY ONE MODEL CAN BE LOADED INTO MEMORY AT A TIME. TO TRAIN OR DO EVALUATIONS YOU MUST SHUTDOWN ALL COMPETING NOTEBOOKS**

2_modeling: Contains the multi-input and single input model notebooks. Each is setup to train the model and set certain parameters. There are many ways to save the model. We prefer saving the whole model as .h5 files to use for future deployment.

3_evaluation: This folder has an Evaluation template that can be used to evaluate any model. It has tweo templates, one for the multiinput model and one for the single input model. You need to reproduce the test and train set in order to evaluate the model which is why the random state is set for the functions that produce that split. There are several ways to recreate the models to make predictions. It is easiest to use the .h5 files and then make prediction.


4_post_processing: The majority of the files in this folder are for mapping the model predictions to the GLOBUS attribute branches. We were given their attributes in the file categories&attributes.xlsx. Walkthrough the **Features mapped to Globus attribute groups** for an understanding of the process. 
