def give_context(n_units):
    context = '''imports:
    preprocessing.filterBank:
        - FilterBank
    sklearn.preprocessing:
        - StandardScaler
    sklearn.decomposition:
        - PCA

Meta:
    file: 'RNN_FBL_PCA_%d'
    cachePreprocessed: False

Preprocessing:
    - FilterBank:
        filters: [[1],[5],[10],[30]]
    - StandardScaler:
    - PCA:

Training:
    lr: 0.1
    decay: 1e-6
    momentum: 0.9
    delay: 256
    skip: 4
    parts_train: 2
    parts_test: 1
    smallEpochs: 4
    majorEpochs: 25
    checkEveryEpochs: 1
    subsample: 1

Architecture:
    - 'GRU':
            dropout: 0.5
            num_units: %d
            next_GRU: False
    - 'Dense':
            num_units: %d
    - 'Activation':
            type: 'relu'
    - 'Dropout':
            p: 0.7
    - 'Output':
    '''%(n_units, n_units, n_units)
    return context

Freqs = [4, 8, 16, 32, 64, 128, 256, 512]
for freq in Freqs:
    cont = give_context(freq)
    with open('./lvl1/models/RNN_FBL_PCA_%d.yml'%freq, 'w') as f:
        f.write(cont)
