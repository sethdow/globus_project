# Files

Each file that starts with Evaluation is an Evaluation of a particular model. The name that follows is the model it is evaluating.

evaluation_helper.py is where all the functions are written that are used to visualize the results

predictions.p: Used as a temp storage for predictions so we didn't have to keep using the .predict() method, which is very slow. 

# Procedure

To Evaluate a new model copy the Evaluation Template of the type of model that you are evaluating, either single or multiinput. The setup is different for each so it matters which you choose. Then change the relevant variables in the file to load the correct model and make predictions. 