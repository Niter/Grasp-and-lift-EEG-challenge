#!/usr/bin/env sh
# This scrip is to go through all models that are configured in lvl1/models
# And store the predictions in lvl1/val and the report in lvl1/reprot

module unload python
module load tensorflow
module load keras

keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge

# for filename in models/*.yml; do
filename="models/RNN_256_customDelay_allModels_ADAM_bags_model.yml"
echo "$filename"
if [[ "$filename" == *"bags_model"* ]]
then
    $keras_python genEns_BagsModels.py $filename val
elif [[ "$filename" == *"bags"* ]]
then
    # python genEns_BagsSubjects.py $filename $i
    echo "pass genEns_BagsSubjects.py"
else
    # python genEns.py $filename $i
    echo "pass genEns.py"
fi
# done
