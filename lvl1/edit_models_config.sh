#!/usr/bin/env sh
# To modify setting in a batch of models configure file

workdir=/home/lucien/eeg_mibk/Grasp-and-lift-EEG-challenge
cd $workdir/lvl1/models

perl -i -pe 's/window: \d+/window: 512/g' ./*
perl -i -pe 's/delay: \d+\//delay: 1024\//g' ./*
perl -i -pe 's/skip: \d+\//skip: 20\//g' ./*
 
# CNN_files=()
# while IFS=  read -r -d $'\0'; do
#     CNN_files+=("$REPLY")
# done < <(find . -name 'cnn_*.yml' -print0)
# echo 
# for f in "${CNN_files[@]}"; do
# done

# perl -pi.bak -e 's/delay: \d+/delay: 800/g' cnn_*.yml
# perl -pi.bak -e 's/skip: \d+/skip: 2/g' cnn_*.yml
