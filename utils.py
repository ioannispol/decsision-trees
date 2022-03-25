"""This utils file inspired from scikit learn implementation of loading 
the naitive dtasets

"""

import csv
import pandas as pd
from matplotlib.pyplot import nipy_spectral
import numpy as np
import os
from pathlib import Path
from importlib import resources
from sklearn.utils import Bunch

DATA_FOLDER = Path("datasets")
TMP_FOLDER = Path("tmp")

def tmp_read(data_file_name):

    tmp_path = Path("tmp/out.csv")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    else:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
    
    return data_file_name.to_csv(tmp_path, index=False)

def fill_null(data_file_name, *, data_folder=DATA_FOLDER, fnull=False) -> object:
    """Helper function to fill the Null values in a csv with the mean

    Args:
        data_file_name (_type_): _description_
        fnull (bool, optional): _description_. Defaults to False.

    Returns:
        object: _description_
    """
    file_path = os.path.join(data_folder, data_file_name)
    with open(file_path) as csv_file:
        csv_file = pd.read_csv(csv_file)
        
        #fnull=None
        # TODO: make the new_header file agnostic
        new_header = ["","","","","","","3276","10", "class_0", "class_1"]

        if fnull:
            is_null = csv_file.isnull().any()
            
            if is_null.any() == True:
                df = csv_file.fillna(csv_file.mean())
                df2 = df.set_axis(new_header, axis=1, inplace=False)

                return df2
        else: 
            return csv_file

def load_csv(data_file_name, *, data_folder=TMP_FOLDER):

    path_to_tmp = os.path.join(TMP_FOLDER, data_file_name)

    with open(path_to_tmp) as csv_file:

        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    return data, target, target_names

def load_water(*, return_X_y=False):
    """Loads the watr dataset

    =================   =======================================
    Classes                          2      Potable/notPotable
    Samples per class               
    Samples total                  3276     
    Dimensionality                   9
    Features            real, positive
    =================   =======================================


    Args:
        return_X_y (bool, optional): _description_. Defaults to False.
        fnull (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    data_file_name = "out.csv"

    data, target, target_names = load_csv(
        data_file_name=data_file_name
    )

    feature_names = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]

   
    target_cols = [
        "target"
    ]
    

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        target_names=target_names,
        feature_names=feature_names,
        filename=data_file_name,
        data_folder=DATA_FOLDER,
    )

