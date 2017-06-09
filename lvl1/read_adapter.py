from glob import glob

DATA_PREPATH = '../data'
subjects_path_list = glob(DATA_PREPATH + '/*/')

def get_horizo_path(idx_subject, idx_sample):
    return subjects_path_list[idx_subject] + 'trialHO%d.csv'%idx_sample

if __name__ == '__main__':
    print get_horizo_path(0, 1)
