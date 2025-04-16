import sys
from pathlib import Path


path = Path(__file__)
parent_folder = path.parent.parent.resolve()

from utils import *
sys.path.append(str(parent_folder))

from kaggle_utils import (view_leaderboard, authentic_kaggle, view_submissions)


competition = 'playground-series-s5e3'


kaggle_cred_file = r'E:\env\KAGGLE_KEY\kaggle.json'
authentic_kaggle(kaggle_cred_file)

view_submissions(competition)
view_leaderboard(competition)

