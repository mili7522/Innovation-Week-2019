import pandas as pd
import numpy as np
import os



def save_predictions(predictions, filenames, save_name):
    filenames = [os.path.basename(x) for x in filenames]
    if predictions.dtype == int:
        category = predictions
    else:
        category = np.argmax(predictions, axis = 1)
    df = pd.DataFrame({"Id": filenames, "Expected": category})
    df.to_csv(save_name, index = None)

def save_summary(model_name, best_kappa, epoch, filename = 'models/performance.csv'):
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col = 0)
    else:
        df = pd.DataFrame(columns = ['Best Kappa', 'Epoch'])
    df.loc[model_name, 'Best Kappa'] = best_kappa
    df.loc[model_name, 'Epoch'] = epoch
    df['Epoch'] = df['Epoch'].astype(int)
    df.to_csv(filename)