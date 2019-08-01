# Innovation-Week-2019

DRNet is a deep neural network model for the detection and classification of severity of diabetic retinopathy from images of the eye. It is a submission for the Sydney Innovation Challenge 2019 (https://www.kaggle.com/c/innovation-challenge-2019/overview).

The model is based on fine tuning of pre-trained Densenet and Inception-ResNet-V2 models using the Artemis cluster. Over 90,000 training images were used, supplementing the original training data with the 2015 and 2019 public kaggle competitions.

The majority of training was done using the train.py and train_inception.py scripts, using model parameters defined in densenet.py and inception.py. A variety of modifications to the network architecture, model size, hyperparameters and loss functions were pre-defined as variants (each making a few modifications to the default structure) and the variant number was input as an integer to the training script. This allowed easy testing of a variety of models.

Several loss functions were tested to find the best performance for this classification task which was assessed using the Quadratic Weighted Kappa metric. Binary crossentropy worked well in the general case when the output was converted into a multi-task variant (ie. detecting each category of severity independently, with a positive result being if the actual category was equal or lower). A continuous approximation of the kappa function was also tested as well as correntropy and cauchy loss which are robust against noisy inputs.

Prediction can performed using the predict_from_file.py script by inputting the path to the folder of images as a command line argument. The second argument specifies the name of the output file. If this file already exists the first column is expected to be the list of file names to test, otherwise all image files in the input path will be tested and output in alphabetical order.
