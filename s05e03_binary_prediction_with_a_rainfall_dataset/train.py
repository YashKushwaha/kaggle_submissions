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
        find_id_variable)

if __name__ == '__main__':
    kaggle_cred_file = r'E:\env\KAGGLE_KEY\kaggle.json'
    load_kaggle_creds(kaggle_cred_file)

    os.chdir(path.parent.resolve())
    print('CWD -> ', os.getcwd())
    
    competition = 'playground-series-s5e3'
    download_competition_data(competition)

    data_folder = os.path.join(os.getcwd(), 'data')      
    df= load_data(data_folder, 'train')

    target = find_target_variable(data_folder)[0]
    id_col = find_id_variable(data_folder)[0]


    df = preprocess_data(df)
    df = feature_engineering(df)

    features_to_keep = feature_selection(df)

    X = df[features_to_keep]
    y = df[target]
    
    X_train, X_test, y_train, y_test = do_train_test_split(X, y)

        # 1. Initialize the model
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # or 'multi:softprob' for multi-class
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'  # Avoid warning in newer versions
    )

    # 2. Train the model
    model.fit(X_train, y_train)

    # 3. Predict on test set
    y_pred = model.predict(X_test)

    # 4. Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 5. Save the model to disk
    joblib.dump(model, 'xgboost_model.pkl')
    print("Model saved as 'xgboost_model.pkl'")