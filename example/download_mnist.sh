#!/bin/bash -x
rm -rf mnist_data
mkdir -p mnist_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P mnist_data
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P mnist_data
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P mnist_data
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P mnist_data
gunzip mnist_data/*
