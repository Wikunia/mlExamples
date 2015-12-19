#!/bin/bash

###############################
# demo data and ML prediction
###############################

### Success

#0 ( black number on gray background) 0,0,0,0,0,0,0,0,0
#python exec_01_normalizeImage.py --input "../numbers/handwritten/0_010.png"

#2 ( black number on gray background) 2,2,2,2,2,2,2,2
#python exec_01_normalizeImage.py --input "../numbers/handwritten/2_010.png"

#2 ( black number on gray background) 2,2,2,2,2,2,2,2
#python exec_01_normalizeImage.py --input "../numbers/handwritten/2_001.png"

#3 ( blue number on gray background) 3,3,3,3,3,3
#python exec_01_normalizeImage.py --input "../numbers/handwritten/3_001.png"

#4 ( blue number on gray background) 4,4,8,4,4,4,4,4,4
#python exec_01_normalizeImage.py --input "../numbers/handwritten/4_001.png"

#5 ( black number on gray background) 5,5,5,5,5,5,5
#python exec_01_normalizeImage.py --input "../numbers/handwritten/5_010.png"

#5 ( black number on gray background) 5,5,5,5,5,5
#python exec_01_normalizeImage.py --input "../numbers/handwritten/5_001.png"

#6 ( black number on gray background) 6,6,6,6,6
python exec_01_normalizeImage.py --input "../numbers/handwritten/6_010.png"

#8 ( black number on gray background) 8,2,8,8,8,8,8,2,8
#python exec_01_normalizeImage.py --input "../numbers/handwritten/8_010.png"

#8 ( black number on gray background) 8,8,8,8,8,8,8
#python exec_01_normalizeImage.py --input "../numbers/handwritten/8.png"

#8 ( gold number on white background) 8,8,8,8
#python exec_01_normalizeImage.py --input "../numbers/handwritten/8_001.png"


### Failed

#1 ( American number format, black number on gray background) 8,8,8,8,8,1,8
#python exec_01_normalizeImage.py --input "../numbers/handwritten/1_011.png"

#1 ( German number format, black number on gray background)
#python exec_01_normalizeImage.py --input "../numbers/handwritten/1_010.png"  #German 1

#4 ( black number on gray background) 8,8,8,8,8,8,9
#python exec_01_normalizeImage.py --input "../numbers/handwritten/4_010.png"
#4 ( black number on gray background) 9,9,9,9,9
#python exec_01_normalizeImage.py --input "../numbers/handwritten/4_011.png"

#6 ( blue number on gray background) 2,2,2,2,2,2,8,2,8,2
#python exec_01_normalizeImage.py --input "../numbers/handwritten/6_001.png"

#7 ( American number format, black number on gray background) 1,1,1,1,1,1,1
#python exec_01_normalizeImage.py --input "../numbers/handwritten/7_011.png"

#9 ( German number format, black number on gray background) 3,3,8,3,3,3,8,3,3
#python exec_01_normalizeImage.py --input "../numbers/handwritten/9_010.png"
#9 ( American number format, black number on gray background) 3,3,3,3,3,3,3,3,3
#python exec_01_normalizeImage.py --input "../numbers/handwritten/9_011.png"
#9 ( German number format, blue number on gray background) 3,3,3,3,3,3,3,3,3
#python exec_01_normalizeImage.py --input "../numbers/handwritten/9_001.png"



### Failed, but ok

#1 ( German number format, blue number on gray background) 9,9,9,9 # OK, MNIST is American number format. German number format was not trained.
#python exec_01_normalizeImage.py --input "../numbers/handwritten/1_002.png"

#3 ( black number on gray background) 0,8,8,0,8,0,0  # ok, the generated 28x28 picture is not great
#python exec_01_normalizeImage.py --input "../numbers/handwritten/3_010.png"

#3 ( black number on gray background) 5,3,5,5,3,3,3,3 # wrong input format black front color !
#python exec_01_normalizeImage.py --input "../numbers/handwritten/3.png"

#7 ( German number format, black number on gray background) 2,2,2,2,2 # OK, MNIST is American number format. German number format was not trained.
#python exec_01_normalizeImage.py --input "../numbers/handwritten/7_001.png"

#7 ( German number format, black number on gray background) 8,2,8,2,28 # OK, MNIST is American number format. German number format was not trained.
#python exec_01_normalizeImage.py --input "../numbers/handwritten/7_010.png"

#8 ( gold number with texture on green background) 8,8,8,8  # OK, the input format is not correct !!
#python exec_01_normalizeImage.py --input "../numbers/handwritten/8_002.png"


#################
# ML prediction
#################

python exec_02_tensorflow.py
