from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import importlib
import sys
import os
from utils import savePredictions
from sklearn.metrics import cohen_kappa_score, accuracy_score

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


train_datagen = modelDefinition.train_datagen
test_datagen = modelDefinition.test_datagen

train_generator = train_datagen.flow_from_directory(
                                                    os.path.join(data_folder, 'Train/train'),
                                                    target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                    batch_size = BATCH_SIZE,
                                                    class_mode = 'categorical'
                                                   )

validation_generator = test_datagen.flow_from_directory(
                                                        os.path.join(data_folder, 'Train/val'),
                                                        target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                        batch_size = BATCH_SIZE,
                                                        class_mode = 'categorical'
                                                       )

test_generator = test_datagen.flow_from_directory(os.path.join(data_folder, 'Test/'),
                                                  target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                  class_mode = None,
                                                  batch_size = 1,
                                                  seed = 42,
                                                  shuffle = False
                                                 )

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
early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1, min_delta = 1e-4)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1, min_delta = 1e-4)
kappa_metrics = Metrics()
callbacks_list = [logger, checkpoint, early_stop, reduce_lr, kappa_metrics]

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
history = model.fit_generator(
                              train_generator,
                              steps_per_epoch = STEP_SIZE_TRAIN,
                              epochs = EPOCHS,
                              validation_data = validation_generator,
                              validation_steps = STEP_SIZE_VALID,
                              callbacks = callbacks_list,
                              class_weight = modelDefinition.class_weight,
                              workers = 8
                             )



# model.save(os.path.join(model_path, model_name) + '.h5')


model = load_model(os.path.join(model_path, model_name) + '_best.h5')

#####
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
predictions = model.predict_generator(
                                      test_generator,
                                      steps = STEP_SIZE_TEST,
                                      verbose = 1
                                     )

savePredictions(predictions, test_generator.filenames, save_name = 'predictions/{}.csv'.format(model_name))