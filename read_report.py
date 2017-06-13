import pandas as pd
from glob import glob

filename_list = glob('./lvl1/report/*.csv')
frame = pd.DataFrame()
list_ = []
for f in filename_list:
    data = pd.read_csv(f)
    list_.append(data)
frame = pd.concat(list_)
print frame.describe()
print frame.sort_values(by='AUC', ascending=False)
