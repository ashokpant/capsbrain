#!/usr/bin/env bash

#Load database links to data directory

unlink ./data/mnist
unlink ./data/fashion_mnist
unlink ./data/att_faces
unlink ./data/smallNORB
unlink ./data/cifar_10
unlink ./data/cifar_100



ln -s /home/ashok/Data/Datasets/mnist ./data/mnist
ln -s /home/ashok/Data/Datasets/fashion_mnist ./data/fashion_mnist
ln -s /home/ashok/Data/Datasets/face/att_faces ./data/att_faces
ln -s /home/ashok/Data/Datasets/smallNORB ./data/smallNORB
ln -s ~/Data/Datasets/cifar/cifar_10/fine/ ./data/cifar_10
ln -s ~/Data/Datasets/cifar/cifar_100/fine/ ./data/cifar_100