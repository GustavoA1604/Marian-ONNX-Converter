#!/bin/bash

sudo apt-get update
sudo apt-get install -y cmake build-essential pkg-config libprotobuf-dev protobuf-compiler

git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig

wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
rm onnxruntime-linux-x64-1.16.3.tgz