import pandas as pd
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_subjects', type=int, action='store', default=2)
parser.add_argument('-l', type=int, action='store', default=1)
args, unknown = parser.parse_known_args()
subjects = range(1, args.n_subjects + 1)
level = args.l

filename_list = glob('./lvl%d/report/val_*.csv'%level)
if len(filename_list) == 0:
    print 'There is no report'
else:
    frame = pd.DataFrame()
    list_ = []
    for f in filename_list:
        data = pd.read_csv(f)
        list_.append(data)
    frame = pd.concat(list_)
    print frame.describe()
    # print frame
    print frame.sort_values(by='AUC', ascending=False)
