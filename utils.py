import pandas as pd
import numpy as np
import os



def savePredictions(predictions, filenames, save_name):
    filenames = [os.path.basename(x) for x in filenames]
    if predictions.dtype == int:
        category = predictions
    else:
        category = np.argmax(predictions, axis = 1)
    df = pd.DataFrame({"Id": filenames, "Expected": category})
    df.to_csv(save_name, index = None)