#!/bin/bash

##############################################################
# demo data and ML prediction - predefined input
##############################################################

echo "Housekeeping, remove results from previous runs"
rm -R output
mkdir output


echo "start 01_dataCleaning.py"
python exec_01_dataCleaning.py
echo "...01 success"
echo "start 02_word2vec_alg.py"
python exec_02_word2vec_alg.py
echo "...02 success"
echo "start 03_prediction.py"
python exec_03_prediction.py
echo "...03 success"
