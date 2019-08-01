import importlib
import sys
import os
import numpy as np
import pandas as pd
from utils import save_predictions, save_probabilities, kappa_loss, ordinal_loss, cauchy_loss, correntropy_loss
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

model = load_model(os.path.join(model_path, model_name) + '_best.h5', custom_objects = {'kappa_loss': kappa_loss, 'ordinal_loss': ordinal_loss, 'cauchy_loss': cauchy_loss, 'correntropy_loss': correntropy_loss})
fill_type = 'mix'

### Load data


def predict_training(folder, save_name, save = True):
    x = np.load(os.path.join(data_folder, folder, 'x_{}.npy'.format(fill_type)))

    if modelClass.last_activation == "softmax":
        y = np.load(os.path.join(data_folder, folder, 'y_{}.npy'.format(fill_type)))
        classes = np.argmax(y, axis = 1)
    else:
        y = np.load(os.path.join(data_folder, folder, 'y_multi_{}.npy'.format(fill_type)))
        classes = y.astype(int).sum(axis = 1) - 1

    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = classes)

    #####
    if modelClass.last_activation == "softmax":
        y_pred_raw = model.predict(x, verbose = verbose)
        y_pred = np.argmax(y_pred_raw, axis = 1)
    else:
        y_pred_raw = model.predict(x, verbose = verbose)
        y_pred = (y_pred_raw > 0.5).astype(int).sum(axis=1) - 1
    kappa = cohen_kappa_score(classes, y_pred, weights = 'quadratic')
    print(kappa)
    if save:
        save_probabilities(y_pred_raw, y_pred, model_name, save_name)

#####
def predict_test():
    x_test = np.load(os.path.join(data_folder, 'Test/test_x_{}.npy'.format(fill_type)))
    if modelClass.last_activation == "softmax":
        y_test_raw = model.predict(x_test, verbose = verbose)
        y_test = np.argmax(y_test_raw, axis = 1)
    else:
        y_test_raw = model.predict(x_test, verbose = verbose)
        y_test = (y_test_raw > 0.5).astype(int).sum(axis=1) - 1
    file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values
    # save_predictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))
    save_probabilities(y_test_raw, y_test, model_name, 'test')


# predict_test()
# predict_training('Train/', 'train')
# predict_training('2015_data/Train/', '2015_train')
# predict_training('2015_data/Test/', '2015_test')
# predict_training('aptos2019_data/Train/', 'aptos')
predict_training('Train/299/', 'train')
predict_training('2015_data/Train/299/', '2015_train')
predict_training('2015_data/Test/299/', '2015_test')
predict_training('2015_data/Test/299a/', '2015_test')
predict_training('aptos2019_data/Train/299/', 'aptos')