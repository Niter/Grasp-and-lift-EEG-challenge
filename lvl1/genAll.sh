#!/usr/bin/env sh
array=( val )
cd ..
python genInfos.py
cd lvl1
cd val
find . -type f -not -name 'placeholder' -print0 | xargs -0 rm --
cd ..
cd report
find . -type f -not -name 'placeholder' -print0 | xargs -0 rm --
cd ..
for i in "${array[@]}"
do
    # generate validation preds
    # cov models
    python genPreds.py models/CovAlex_500.yml $i &
    python genPreds.py models/CovAlex_500_1-15.yml $i &
    python genPreds.py models/CovAlex_500_20-35.yml $i &
    python genPreds.py models/CovAlex_500_70-150.yml $i &
    python genPreds.py models/CovAlex_250_35Hz.yml $i &
    python genPreds.py models/CovAlex_500_35Hz.yml $i &
    python genPreds.py models/CovERP_dist.yml $i &
    wait
    # PolynomialFeatures cov model
    python genPreds.py models/CovAlex_500_poly.yml $i &
    python genPreds.py models/CovAlex_500_1-15_poly.yml $i &
    python genPreds.py models/CovAlex_500_20-35_poly.yml $i &
    python genPreds.py models/CovAlex_500_70-150_poly.yml $i &
    python genPreds.py models/CovAlex_250_35Hz_poly.yml $i &
    python genPreds.py models/CovAlex_500_35Hz_poly.yml $i &
    python genPreds.py models/CovERP_dist_poly.yml $i &
    wait
    # rafal cov model 
    python genPreds.py models/CovRafal_256_35Hz.yml $i &
    python genPreds.py models/CovRafal_500_35Hz.yml $i &
    wait
    # aggregated cov model
    python genPreds.py models/CovAlex_All.yml $i &
    python genPreds.py models/CovAlex_old_All.yml $i &
    wait

    # Low pass EEG model
    python genPreds.py models/FBL.yml $i &
    python genPreds.py models/FBL_delay.yml $i &
    wait

    # Hybrid model (cov + FBL)
    python genPreds.py models/FBLC_256pts.yml $i python genPreds.py models/FBLCR_256.yml $i &
    wait

    # NN models
    python genPreds_RNN.py models/RNN_FB_delay4000.yml $i
    python genPreds_CNN_Tim.py models/cnn_script_2D_30Hz.yml $i
    python genPreds_CNN_Tim.py models/cnn_script_2D_30Hz_shorterDelay.yml $i
    python genPreds_CNN_Tim.py models/cnn_script_1D_30Hz.yml $i
    python genPreds_CNN_Tim.py models/cnn_script_1D_30Hz_shorterDelay.yml $i
    python genPreds_CNN_Tim.py models/cnn_script_1D_5Hz.yml $i
    python genPreds_CNN_Tim.py models/cnn_script_1D_7-30Hz.yml $i

done
