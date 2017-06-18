from glob import glob
import scipy.io as sio

DATA_PREPATH = '/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge/data/'
subjects_path_list = glob(DATA_PREPATH + '*/')

def get_all_horizon_path_from_the_subject(idx_subject):
    return subjects_path_list[idx_subject] + 'trialHO*.csv'

def get_all_vertical_path_from_the_subject(idx_subject):
    return subjects_path_list[idx_subject] + 'trialVE*.csv'

def get_horizo_path(idx_subject, idx_sample):
    return subjects_path_list[idx_subject] + 'trialHO%d.csv'%idx_sample

def get_horizo_velocity_path():
    return DATA_PREPATH + 'ControlSignalHO_Manual.mat'

def get_horizo_velocity():
    return sio.loadmat(get_horizo_velocity_path())['ControlSignalHO_Manual']

def get_vertic_velocity_path():
    return DATA_PREPATH + 'ControlSignalVE_Manual.mat'

def get_vertic_velocity():
    return sio.loadmat(get_vertic_velocity_path())['ControlSignalVE_Manual']

if __name__ == '__main__':
    print get_horizo_path(0, 1)
    print get_horizo_velocity().shape
    print get_vertic_velocity().shape
