# Summary of all the files and directories
1. Modeling-refactored: the main script for training models
2. Checkpoints: Directory where weights of models are stored after each epoch in case of crash
3. Saved_model: Directory where each model(including architecture, weights, and optimizer configuration) will be saved
4. Training_history: Directory where the metrics for training different models will be stored

# Naming conventions
It is assumed that file names signify deviations from the base model. The basic model architecture is VGG19 cutoff at the last convolution layer followed by the following: flatten, dense, dropout with .2, the dense with the number of neurons as the number of classes. 

### Files of each type will be named like the following:

model#_sample_size#_epoch_numbers#_number_dense_trainable_layers_loss_function

- sample_size: the number of photos taken to traing the model
- epoch_number: nu,mber of epochs this model has been trained
- loss_function: which loss function was used to train the model
- number_dense: the number of dense later 

# Notes


'model1_sample_sizeall_epoch50_dense2_losswbc' was trained on 30.1.20, I setup a screen session, started a notebook and started running it, then logged out. I was not able to log back in until the next morning. The weights had been saved but length of training is unknown.

'model2_sample_sizeall_epoch50_dense2_losswbc' was trained on 31.1.20, it died at batch 657, which means that it made it to about 20k photos before dying. 