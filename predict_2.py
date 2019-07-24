import importlib
import sys
import os
import numpy as np
import pandas as pd
from utils import savePredictions
from keras.models import load_model

if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    model_type = 'densenet'

if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
else:
    data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
model_path = 'models'
model_name = model_type

model = load_model(os.path.join(model_path, model_name) + '_best.h5')

x_test = np.load(os.path.join(data_folder, 'Test/test_x.npy'))

#####
y_test = model.predict(x_test) > 0.5
y_test = y_test.astype(int).sum(axis=1) - 1

file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values

savePredictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))