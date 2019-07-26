import importlib
import sys
import os
import numpy as np
import pandas as pd
from utils import save_predictions, kappa_loss, ordinal_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from keras.models import load_model

if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    model_type = 'densenet'
if len(sys.argv) > 2:
    model_variant = int(sys.argv[2])
else:
    model_variant = 1
if len(sys.argv) > 3:
    repetition = int(sys.argv[2])
else:
    repetition = None

model_script = 'model_definitions.{}'.format(model_type)
modelDefinition = importlib.import_module(model_script)
modelClass = modelDefinition.getModelVariant(model_variant)

if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
    verbose = 1
else:
    data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
    verbose = 0
model_path = 'models'
model_name = "{}-{:03}{}".format( model_type, model_variant, "" if repetition is None else "_r{:02}".format(repetition) )

model = load_model(os.path.join(model_path, model_name) + '_best.h5', custom_objects = {'kappa_loss': kappa_loss, 'ordinal_loss': ordinal_loss})

### Load data
folders = ['Train/x_mix.npy', '2015_data/Train/x_mix.npy']
xs = [np.load(os.path.join(data_folder, folder)) for folder in folders]
x = np.vstack(xs)
x_test = np.load(os.path.join(data_folder, 'Test/test_x_mix.npy'))

if modelClass.last_activation == "softmax":
    folders = ['Train/y_mix.npy', '2015_data/Train/y_mix.npy']
    ys = [np.load(os.path.join(data_folder, folder)) for folder in folders]
    y = np.vstack(ys)
    classes = np.argmax(y, axis = 1)
else:
    folders = ['Train/y_multi_mix.npy', '2015_data/Train/y_multi_mix.npy']
    ys = [np.load(os.path.join(data_folder, folder)) for folder in folders]
    y = np.vstack(ys)
    classes = y.astype(int).sum(axis = 1) - 1

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = classes)

#####
if modelClass.last_activation == "softmax":
    y_test = model.predict(x_test, verbose = verbose)
    y_test = np.argmax(y_test, axis = 1)
else:
    y_test = model.predict(x_test, verbose = verbose) > 0.5
    y_test = y_test.astype(int).sum(axis=1) - 1

file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values

save_predictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))

#####
if modelClass.last_activation == "softmax":
    y_val = np.argmax(y_val, axis = 1)
    y_pred = model.predict(x_val, verbose = verbose)
    y_pred = np.argmax(y_pred, axis = 1)
else:
    y_val = y_val.sum(axis = 1) - 1
    y_pred = model.predict(x_val, verbose = verbose) > 0.5
    y_pred = y_pred.astype(int).sum(axis=1) - 1
kappa = cohen_kappa_score(y_val, y_pred, weights = 'quadratic')
print(kappa)