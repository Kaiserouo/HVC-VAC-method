#!/bin/bash
# Execute this while current working directory is in this folder
# to generate result image
python ../../main.py -c -i img_a.png img_b.png -s img_sec.png -o img_res_a.png img_res_b.png

# if you also want the bitwise-AND result, add `-a img_and.png` in argument list

# if you want to save the model and also do HVC, you can do either the following:
# ```
#   python ../../main -s img_sec.png -m test.pickle
#   python ../../main -c -i img_a.png img_b.png -l test.pickle -o img_res_a.png img_res_b.png
# ```
# The first step world take significantly longer than the second.
# Or you can combine the 2 step, that is allowed:
# ```
#   python ../../main.py -c -i img_a.png img_b.png -s img_sec.png \
#                        -o img_res_a.png img_res_b.png -m test.pickle
# ```