# !/bin/bash

# Convert MCTS.py to MCTS.pyx
python make_pyx.py

# Compile the MCTS.pyx file
python setup.py build_ext --inplace

echo ".pyx files built"