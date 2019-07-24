from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import importlib
import sys
import os
import numpy as np
import pandas as pd
from utils import savePredictions
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.models import load_model

if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    model_type = 'densenet'


model_script = 'model_definitions.{}'.format(model_type)
modelDefinition = importlib.import_module(model_script)
model = modelDefinition.build_model()
IMAGE_SIZE = modelDefinition.IMAGE_SIZE

if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
else:
    data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
model_path = 'models'  # For saving
model_name = model_type

training_examples = 11314
validation_examples = 2831
test_examples = 999

EPOCHS = 30
BATCH_SIZE = 16

### Load data
x_train = np.load(os.path.join(data_folder, 'Train/training_x.npy'))
y_train = np.load(os.path.join(data_folder, 'Train/training_y_multi.npy'))
x_val = np.load(os.path.join(data_folder, 'Train/val_x.npy'))
y_val = np.load(os.path.join(data_folder, 'Train/val_y_multi.npy'))
x_test = np.load(os.path.join(data_folder, 'Test/test_x.npy'))

train_datagen = modelDefinition.train_datagen

train_generator = train_datagen.flow(x_train, y_train, batch_size = BATCH_SIZE)


###
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save(os.path.join(model_path, model_name) + '_best.h5')

        return

logger = CSVLogger(os.path.join(model_path, model_name) + '-History.csv', separator = ',', append = True)
os.makedirs(os.path.join(model_path, model_name + '-Checkpoint'), exist_ok = True)  # Create folder if not present
checkpoint = ModelCheckpoint(os.path.join(model_path, model_name + '-Checkpoint', model_name) + '-Checkpoint-{epoch:03d}.h5')
# early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, min_delta = 1e-4)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4)
kappa_metrics = Metrics()
# callbacks_list = [logger, checkpoint, early_stop, reduce_lr, kappa_metrics]
callbacks_list = [logger, checkpoint, reduce_lr, kappa_metrics]

STEP_SIZE_TRAIN = x_train.shape[0] // train_generator.batch_size
history = model.fit_generator(
                              train_generator,
                              steps_per_epoch = STEP_SIZE_TRAIN,
                              epochs = EPOCHS,
                              validation_data = (x_val, y_val),
                              callbacks = callbacks_list,
                              class_weight = modelDefinition.class_weight,
                              workers = 4
                             )


# model.save(os.path.join(model_path, model_name) + '.h5')

model = load_model(os.path.join(model_path, model_name) + '_best.h5')

#####
y_test = model.predict(x_test) > 0.5
y_test = y_test.astype(int).sum(axis=1) - 1

file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None).values

savePredictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))

pd.DataFrame(kappa_metrics.val_kappas).to_csv(os.path.join(model_path, model_name) + '-kappa.csv', header = False)