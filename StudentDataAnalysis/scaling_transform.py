import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_data_sklearn(data: pd.DataFrame, columns: list = [], scaling_obj: object = None, fit: bool = True):
    data = data.reset_index(drop=True)
    if data.shape[0] == 0:
        raise Exception("Data is empty. Please check")
    if len(columns) == 0:
        raise Exception("Columns Not Given")
    for col in columns:
        if col not in data.columns:
            raise Exception(f"{col} not in data")
    if scaling_obj is None and not fit:
        raise Exception("There must be an object for just to transform")
    if scaling_obj is None:
        scaling_obj = StandardScaler()
        fit = True
        print("No Object Is given, doing forecful fit")
    if fit:
        scaled_df = scaling_obj.fit_transform(data[columns]) # type: ignore
        data = pd.concat([data.drop(columns=columns), pd.DataFrame(scaled_df, columns=columns)], axis=1).reset_index(drop=True)
    else:
        scaled_df = scaling_obj.transform(data[columns]) # type: ignore
        data = pd.concat([data.drop(columns=columns), pd.DataFrame(scaled_df, columns=columns)], axis=1).reset_index(drop=True)
    return data, scaling_obj

def descale_data(data: pd.DataFrame, columns: list = [], scaling_obj: object = None):
    data = data.reset_index(drop=True)
    if scaling_obj is None:
        print("Scaling Object Can't be None")
    for col in columns:
        if col not in data.columns:
            raise Exception(f"{col} not in data")
    descaled_df = scaling_obj.inverse_transform(data[columns]) # type: ignore
    data = pd.concat([data.drop(columns=columns), pd.DataFrame(descaled_df, columns=columns)], axis=1).reset_index(drop=True)
    return data