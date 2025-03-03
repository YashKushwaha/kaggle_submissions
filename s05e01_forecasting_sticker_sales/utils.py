import os
import json
import zipfile
import subprocess
#from kaggle.api_client import ApiClient


def load_kaggle_creds(kaggle_cred_file):
    with open(kaggle_cred_file, "r") as f:
        credentials = json.load(f)
    for i,j in credentials.items():
        os.environ[f'KAGGLE_{i}'] = j 
        
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

def upload_submission(competition_name, submission_file, message):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    command = f"kaggle competitions submit -c {competition_name} -f {submission_file} -m \"{message}\""
    subprocess.run(command, shell=True)

def show_box_plot(data, categorical_cols, target_col):
    """Generates box plots for multiple categorical variables."""
    num_plots = len(categorical_cols)
    fig, axes = plt.subplots(num_plots, 1, figsize=(5, 5*num_plots), sharey=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for col, ax in zip(categorical_cols, axes):
        sns.boxplot(x=data[col], y=data[target_col], ax=ax, 
                    boxprops={'facecolor': 'lightgray', 'alpha': 0.6},  # Light fill color
                    medianprops={'color': 'red', 'linewidth': 2})  # Emphasize median

        ax.set_title(f"Boxplot of {target_col} vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        ax.tick_params(axis='x', rotation=45)  # Rotate labels for readability

    plt.tight_layout()
    plt.show()
    
class DataPipeline:
    def __init__(self,df):
        self.cat_cols = ['Brand', 'Material', 'Size','Laptop Compartment', 'Waterproof', 'Style', 'Color']
        self.int_cols =  ['Compartments']
        self.float_cols = ['Weight Capacity (kg)']
        self.df = df.copy()
        self.mode_dict = {}
        self.mean_dict = {}

        self.scaler = None
    def imputer(self):
        for col in self.cat_cols + self.int_cols:
            self.mode_dict[col] = self.df[col].mode()[0]
        for col in self.float_cols:
            self.mean_dict[col] = float(self.df[col].mean())
        return {**self.mode_dict, **self.mean_dict}        
    
    def allowed_values(self):
        values_seen = {}
        for col in self.cat_cols + self.int_cols:
            values_seen[col] = self.df[col].dropna().unique().tolist()
        return values_seen
    def fit(self):
        self.mode_dict = self.imputer()
        self.allowed_values = self.allowed_values()
        self.scaler
        return self
    
    def transform(self, df=None):
        df = df if df is not None else self.df
        for col in self.cat_cols + self.int_cols:
            df[col] = df[col].fillna(self.mode_dict[col])
            df.loc[~df[col].isin(self.allowed_values[col]), col] = self.mode_dict[col]

        for col in self.cat_cols:
            df[col] = df[col].astype('category') 
        for col in self.float_cols:
            df[col] = df[col].fillna(self.mean_dict[col])
        return df