#!/usr/bin/env sh
array=( val )

module unload python
module load tensorflow
module load keras

keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge

for i in "${array[@]}"
do
  for filename in models/*.yml; do
    echo "$filename"
    if [[ "$filename" == *"bags_model"* ]]
    then
      # $keras_python genEns_BagsModels.py $filename $i     
      echo "pass genEns_BagsModels.py"
    elif [[ "$filename" == *"bags"* ]]
    then
      # $keras_python genEns_BagsSubjects.py $filename $i
      echo "pass genEns_BagsSubjects.py"
    else
      $keras_python genEns.py $filename $i
      # echo "pass genEns.py"
    fi
  done
done
