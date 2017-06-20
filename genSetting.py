def give_context(model_name, n_units):
    context = '''imports:
    preprocessing.filterBank:
        - FilterBank
    sklearn.preprocessing:
        - StandardScaler
    sklearn.decomposition:
        - PCA

Meta:
    file: '%s'
    cachePreprocessed: False

Preprocessing:
    # - FilterBank:
    #     filters: [[1],[5],[10],[30]]
    - StandardScaler:
    # - PCA:


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
    - 'Dropout':
        p: 0.5
    - 'Dense':
        num_units: %d
    - 'Activation':
        type: 'relu'
    - 'Dropout':
        p: 0.7
    - 'Output':
    - 'Activation':
        type: 'tanh'
    '''%(model_name, n_units)
    return context

Freqs = [4, 8, 16, 32, 64, 128, 256, 512]
for freq in Freqs:
    level = 1
    model_name = 'NN_%d'%freq
    filename = './lvl%d/models/%s.yml'%(level, model_name)
    print 'Generating: %s'%(filename)
    cont = give_context(model_name, freq)
    with open(filename, 'w') as f:
        f.write(cont)
