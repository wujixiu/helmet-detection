#!/bin/sh
./build/tools/caffe train -solver="examples/Hardhat/Pelee-RPA/solver.prototxt" \
-weights="models/Pelee/peleenet_inet_acc7243.caffemodel" \
--gpu 0 2>&1 | tee examples/Hardhat/Pelee/Pelee.log \

