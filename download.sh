#!/bin/bash
mkdir -p data
cd data

wget http://statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar -xzvf training-parallel-europarl-v7.tgz
rm training-parallel-europarl-v7.tgz

wget http://statmt.org/wmt14/training-parallel-nc-v9.tgz
tar -xzvf training-parallel-nc-v9.tgz
rm training-parallel-nc-v9.tgz

cd ..
