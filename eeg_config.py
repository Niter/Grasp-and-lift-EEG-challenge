'''
This script contains most of the configuration parameters that applies to model
'''

# from read_adapter import DATA_PREPATH, subjects_path_list

CH_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
# START_TRAIN = 660

DIRECTION_CURSOR = 'HO'

IS_CLASSIFICATION = True
IS_REGRESSION = not IS_CLASSIFICATION

N_EVENTS = 3 if IS_CLASSIFICATION else 1

# subjects = range(1, len(subjects_path_list)) # 34 subjects in total
# subjects = range(1, 3)
