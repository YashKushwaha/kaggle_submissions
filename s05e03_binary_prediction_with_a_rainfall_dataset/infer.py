import sys
import os
from pathlib import Path
import pandas as pd

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

path = Path(__file__)
parent_folder = path.parent.parent.resolve()

from utils import *
sys.path.append(str(parent_folder))


from kaggle_utils import (load_kaggle_creds, download_competition_data,
        find_target_variable,
        find_id_variable, upload_submission)

if __name__ == '__main__':
    kaggle_cred_file = r'E:\env\KAGGLE_KEY\kaggle.json'
    load_kaggle_creds(kaggle_cred_file)

    os.chdir(path.parent.resolve())
    print('CWD -> ', os.getcwd())
    
    competition = 'playground-series-s5e3'
    download_competition_data(competition)

    data_folder = os.path.join(os.getcwd(), 'data')    

    target = find_target_variable(data_folder)[0]
    id_col = find_id_variable(data_folder)[0]

    df= load_data(data_folder, 'test')
    df = preprocess_data(df)
    df = feature_engineering(df)
    
    features_to_keep = feature_selection(df)

    X = df[features_to_keep]

    model_file = 'xgboost_model.pkl'
    model = joblib.load(model_file)

    y_pred_new = model.predict(X)

    df[target] = y_pred_new

    submission_file = 'submission.csv'

    submission_df = df[[id_col,target]]
    submission_file = 'first_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    
    message = 'first submission'
    upload_submission(competition, submission_file, message)
