import os
import json
import zipfile
import subprocess
#from kaggle.api_client import ApiClient

import pandas as pd

def load_kaggle_creds(kaggle_cred_file):
    with open(kaggle_cred_file, "r") as f:
        credentials = json.load(f)
    for i,j in credentials.items():
        os.environ[f'KAGGLE_{i}'] = j 

def authentic_kaggle(kaggle_cred_file):
    
    load_kaggle_creds(kaggle_cred_file)
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return None

def download_competition_data(competition):
    downloaded_file = f'{competition}.zip'
    if not os.path.exists(downloaded_file):
        print('Downloading data files')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition)
    if os.path.exists('data') and os.listdir('data'):
        print('Data files already found')
    else:
        print('Unzipping data files')
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall('data')  # Extracts all files

def find_target_variable(data_folder):
    train_file = os.path.join(data_folder, 'train.csv')
    test_file = os.path.join(data_folder, 'test.csv')

    train_df = pd.read_csv(train_file, nrows=1)
    test_df = pd.read_csv(test_file, nrows=1)

    target_variable = [i for i in train_df.columns if i not in test_df.columns]
    return target_variable

def find_id_variable(data_folder):
    train_file = os.path.join(data_folder, 'train.csv')
    submission_file = os.path.join(data_folder, 'sample_submission.csv')

    train_df = pd.read_csv(train_file, nrows=1)
    submission_df = pd.read_csv(submission_file, nrows=1)

    id_variable_and_target_var = [i for i in train_df.columns if i in submission_df.columns]

    target_variable = find_target_variable(data_folder)
    id_variable = [i for i in id_variable_and_target_var if i not in target_variable]
    return id_variable

def upload_submission(competition_name, submission_file, message):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    command = f"kaggle competitions submit -c {competition_name} -f {submission_file} -m \"{message}\""
    subprocess.run(command, shell=True)

def view_leaderboard(competition_name):
    #from kaggle.api.kaggle_api_extended import KaggleApi
    #api = KaggleApi()
    #api.authenticate()
    command = f"kaggle competitions leaderboard {competition_name} -s >> leaderboard.txt"
    subprocess.run(command, shell=True)

def view_submissions(competition_name):
    command = f"kaggle competitions submissions -c {competition_name} >> leaderboard.txt"
    subprocess.run(command, shell=True)