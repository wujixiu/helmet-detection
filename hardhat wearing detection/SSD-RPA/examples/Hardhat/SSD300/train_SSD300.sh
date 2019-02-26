#!/bin/sh
./build/tools/caffe train -solver="examples/Hardhat/SSD300/solver.prototxt" \
-weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee examples/Hardhat/SSD300/SSD300_Hardhat.log

\
