#!/bin/bash

##############################################################
# Only visualize computed word2vec.
# Its needed call run01_compute_and_visualize.sh
# at least one time before.
##############################################################

echo " * only visualize computed word2vec "
python exec_04_w2v_main.py --searchWord1 "american" --searchWord2 "four" --onlyVisualize "true"
