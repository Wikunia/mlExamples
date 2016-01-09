#!/bin/bash

##############################################################
# compute word2vec and visualize results
##############################################################

echo "Housekeeping, remove results from previous runs"
rm -R output
mkdir output

echo " * compute word2vec and visualize results * "
python exec_04_w2v_main.py --searchWord1 "american" --searchWord2 "four" --onlyVisualize "false"
#python exec_04_w2v_main.py --searchWord1 "one" --searchWord2 "british" --onlyVisualize "false"
#python exec_04_w2v_main.py --searchWord1 "will" --searchWord2 "country" --onlyVisualize "false"
#python exec_04_w2v_main.py --searchWord1 "computer" --searchWord2 "technology" --onlyVisualize "false"
