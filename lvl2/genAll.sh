#!/usr/bin/env sh
array=( val )

module unload python
module load tensorflow
module load keras

keras_python=/opt/packages/keras/keras_2.0.4/kerasEnv/bin/python
workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge
n_subjects=12

cd $workdir
python genInfos.py --n_subjects=$n_subjects
cd $workdir/lvl2

for i in "${array[@]}"
do
  for filename in models/[xgbNN]*.yml; do
    echo "$filename"

    if [[ "$filename" == *"bags_model"* ]]
    then
      $keras_python genEns_BagsModels.py $filename $i --n_subjects=$n_subjects
      # echo "pass genEns_BagsModels.py"
    elif [[ "$filename" == *"bags"* ]]
    then
      $keras_python genEns_BagsSubjects.py $filename $i --n_subjects=$n_subjects
      # echo "pass genEns_BagsSubjects.py"
    else
      $keras_python genEns.py $filename $i --n_subjects=$n_subjects
      # echo "pass genEns.py"
    fi
  done
done
