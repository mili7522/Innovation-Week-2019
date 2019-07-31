from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import importlib
import sys
import os
import numpy as np
import pandas as pd
from utils import save_predictions, save_summary, kappa_loss, ordinal_loss, cauchy_loss, correntropy_loss
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
    repetition = int(sys.argv[3])
else:
    repetition = None


model_script = 'model_definitions.{}'.format(model_type)
modelDefinition = importlib.import_module(model_script)
modelClass = modelDefinition.getModelVariant(model_variant)
model = modelClass.build_model()

if os.path.exists('/media/mike/Files/'):
    data_folder = '/media/mike/Files/Data and Results/innovation-challenge-2019/'
    verbose = 1
    train_folders = ['aptos2019_data/Train/299/', 'Train/299/']
else:
    data_folder = '/project/rc2d/Mike/InnovationWeek/Data/'
    verbose = 2
    train_folders = ['2015_data/Train/299/', '2015_data/Test/299/', '2015_data/Test/299a/', 'aptos2019_data/Train/299/']
val_folder = 'Train/299/'
model_path = 'models'  # For saving
model_name = "{}-{:03}{}".format( model_type, model_variant, "" if repetition is None else "_r{:02}".format(repetition) )
print("\n======================")
print("Training", model_name, flush = True)
print("======================\n")

# training_examples = 11314
# validation_examples = 2831
# test_examples = 999

EPOCHS = 30
BATCH_SIZE = 16
fill_type = 'mix'

### Load data
xs = [np.load(os.path.join(data_folder, folder + 'x_{}.npy'.format(fill_type)), mmap_mode = 'r') for folder in train_folders]
x = np.vstack(xs)

if modelClass.last_activation == "softmax":
    ys = [np.load(os.path.join(data_folder, folder + 'y_{}.npy'.format(fill_type))) for folder in train_folders]
    y = np.vstack(ys)
    classes = np.argmax(y, axis = 1)
else:
    ys = [np.load(os.path.join(data_folder, folder + 'y_multi_{}.npy'.format(fill_type))) for folder in train_folders]
    y = np.vstack(ys)
    classes = y.astype(int).sum(axis = 1) - 1

# Load val data
X_val = np.load(os.path.join(data_folder, val_folder + 'x_{}.npy'.format(fill_type)), mmap_mode = 'r')
if modelClass.last_activation == "softmax":
    y_val = np.load(os.path.join(data_folder, val_folder + 'y_{}.npy'.format(fill_type)))
    y_val = np.argmax(y_val, axis = 1)
else:
    y_val = np.load(os.path.join(data_folder, val_folder + 'y_multi_{}.npy'.format(fill_type)))
    y_val = y_val.astype(int).sum(axis = 1) - 1

# Load test data
x_test = np.load(os.path.join(data_folder, 'Test/299/test_x_{}.npy'.format(fill_type)))
file_list = pd.read_csv(os.path.join(data_folder, 'Test/test_files.csv'), header = None, squeeze = True).values
os.makedirs('predictions', exist_ok = True)


def own_train_generator(x, y, train_datagen):
    while True:
        shuffled_ids = np.random.permutation(list(range(len(y))))
        for start in range(0, len(shuffled_ids), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(shuffled_ids))
            batch_len = end - start
            x_batch = np.empty([batch_len] + list(x.shape[1:]))
            y_batch = np.empty([batch_len] + list(y.shape[1:]))
            ids_train_batch = shuffled_ids[start:end]
            for i, idx in enumerate(ids_train_batch):
                orig_image = x[idx].copy()
                transform = train_datagen.get_random_transform(orig_image.shape)
                new_image = train_datagen.apply_transform(orig_image, transform)
                x_batch[i] = new_image
                y_batch[i] = y[idx]
            yield x_batch, y_batch


###
best_kappa = -np.inf
val_kappas = []
best_kappa_epoch = None
class Metrics(Callback):
    # def on_train_begin(self, logs={}):
        # self.val_kappas = []
        # self.best_kappa = -np.inf
        # self.best_kappa_epoch = None

    def on_epoch_end(self, epoch, logs={}):
        global best_kappa, best_kappa_epoch, val_kappas
        if modelClass.last_activation == "softmax":
            y_pred = self.model.predict(X_val)
            y_pred = np.argmax(y_pred, axis = 1)
        else:
            y_pred = self.model.predict(X_val) > 0.5
            y_pred = y_pred.astype(int).sum(axis = 1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa > best_kappa:
            best_kappa = _val_kappa
            best_kappa_epoch = epoch
            print("Validation Kappa has improved. Predicting and saving model.")
            if modelClass.last_activation == "softmax":
                y_test = self.model.predict(x_test, verbose = (verbose - 2) * -1 )
                y_test = np.argmax(y_test, axis = 1)
            else:
                y_test = self.model.predict(x_test, verbose = (verbose - 2) * -1 ) > 0.5
                y_test = y_test.astype(int).sum(axis = 1) - 1
            
            save_predictions(y_test, file_list, save_name = 'predictions/{}.csv'.format(model_name))
            pd.DataFrame(val_kappas).to_csv(os.path.join(model_path, 'history', model_name) + '-kappa.csv', header = False)

            self.model.save(os.path.join(model_path, model_name) + '_best.h5')

        return

logger = CSVLogger(os.path.join(model_path, 'history', model_name) + '-History.csv', separator = ',', append = True)
os.makedirs(os.path.join(model_path, 'history'), exist_ok = True)
# os.makedirs(os.path.join(model_path, model_name + '-Checkpoint'), exist_ok = True)  # Create folder if not present
# checkpoint = ModelCheckpoint(os.path.join(model_path, model_name + '-Checkpoint', model_name) + '-Checkpoint-{epoch:03d}.h5')
# early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, min_delta = 1e-4)
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3, verbose = 1, min_delta = 1e-4)
kappa_metrics = Metrics()
callbacks_list = [logger, reduce_lr, kappa_metrics]
# callbacks_list = [logger, reduce_lr]


train_datagen = modelClass.get_image_datagen()
STEP_SIZE_TRAIN = x.shape[0] // BATCH_SIZE
model.fit_generator(
                    own_train_generator(x, y, train_datagen),
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    epochs = EPOCHS // 4,
                    callbacks = callbacks_list,
                    class_weight = modelClass.class_weight,
                    workers = 1,
                    verbose = verbose
                   )

model.trainable = True

model.fit_generator(
                    own_train_generator(x, y, train_datagen),
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    initial_epoch = EPOCHS // 4,
                    epochs = EPOCHS,
                    callbacks = callbacks_list,
                    class_weight = modelClass.class_weight,
                    workers = 1,
                    verbose = verbose
                   )

# model.save(os.path.join(model_path, model_name) + '.h5')
# model = load_model(os.path.join(model_path, model_name) + '_best.h5', custom_objects = {'kappa_loss': kappa_loss, 'ordinal_loss': ordinal_loss})

#####
save_summary(model_name, best_kappa = best_kappa, epoch = best_kappa_epoch, filename = 'models/performance.csv')