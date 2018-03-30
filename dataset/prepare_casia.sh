#!/bin/bash


#python tfrecord/tfrecord_writer.py --database_dir /data/datasets/CASIA-WebFace/Normalized_Faces/webface/100/ --image_size 64 --image_size 64 --tfrecord_file /data/datasets/CASIA-WebFace/casia
#python tfrecord_writer.py --dataset_dir data/mnist --tfrecord_file mnist --force True --image_size 64 --image_size 64 --test_ratio 0.1
python tfrecord/tfrecord_writer.py --dataset_dir data/att_faces/  --tfrecord_file att_faces --force True --image_size 32 --image_size 32 --test_ratio 0.2


#python tfrecord_writer.py --dataset_dir /data/datasets/CASIA-WebFace/Normalized_Faces/webface/100/ --tfrecord_file  /data/datasets/CASIA-WebFace/casia --force True --image_size 32 --image_size 32 --test_ratio 0.2 --max_classes 1000 --min_samples_per_class 1000
