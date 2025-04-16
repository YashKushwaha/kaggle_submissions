import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_folder, data_type='train'):
    if data_type == 'train':
        file = os.path.join(data_folder, 'train.csv')
    else:
        file = os.path.join(data_folder, 'test.csv')
    
    df = pd.read_csv(file)
    
    return df

def preprocess_data(df):
    return df

def feature_engineering(df, stage='train'):
    return df

def feature_selection(df):
    features_to_keep = ['humidity', 'cloud', 'sunshine']
    return features_to_keep

def do_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test 

def train_model(X,y):
    pass