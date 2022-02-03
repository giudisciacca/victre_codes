README FILE
Suite for the classification of results from DOT.

Two main python scripts are present, SVM.py classifies data by means of Support Vector Machines and logistic regression. Classify.py implements a small neural network fro classification with tensorflow.

Each of the scripts requires 3 mat files as input. In the form {NAME}.mat for the training set,{NAME}_v_t.mat for the validation set and {NAME}_t.mat for the test set.

Such mat files will contain a vector of 16 features x N samples. The name of the mat file conttaining the training set will need to be hard coded  in the variable "loadname".





