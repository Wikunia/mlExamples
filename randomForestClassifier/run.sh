#!/bin/bash

##############################################################
# demo data and ML prediction - predefined input
##############################################################

echo "Housekeeping, remove results from previous runs"
cd output
rm *.json
rm *.npy
cd ../results
rm *.csv
cd ..

echo "start 01_dataCleaning.py"
python exec_01_dataCleaning.py
echo "...01 success"
echo "start 02_bag_of_words.py"
python exec_02_bag_of_words.py
echo "...02 success"
echo "start 03_random_forest_classifier.py"
python exec_03_random_forest_classifier.py
echo "...03 success"
echo "start 04_prediction_manual_input.py"
python exec_04_prediction.py
echo "...04 success"


##############################################################
# demo data and ML prediction - manual input
##############################################################


### Possitive feedback
echo "Test: should be possitive !"
python exec_04_prediction_manual_input.py



#exec_01_normalizeImage.py --input "../numbers/handwritten/0_010.png"
