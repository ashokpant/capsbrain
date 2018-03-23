#!/bin/bash


#python tfrecord/tfrecord_writer.py --database_dir /data/datasets/CASIA-WebFace/Normalized_Faces/webface/100/ --image_size [64, 64] --tfrecord_file /data/datasets/CASIA-WebFace/casia
python tfrecord/tfrecord_writer.py --dataset_dir data/att_faces/  --tfrecord_file att_faces

