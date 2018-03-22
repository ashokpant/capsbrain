#!/bin/bash

dir=./att_faces

if [ ! -d "$dir" ]; then
    mkdir $dir
    cd $dir
    wget "http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
    echo "Done fetching archive files. Extracting..."

    for archive in ./*.zip ; do
        echo "$archive"
        unzip "$archive"
    done

    echo "Done!"
else
    echo "$dir already exist!"
fi


echo "Creating train/test (20%) files"
chmod +x ./make_train_test.sh
sh ./make_train_test.sh ${dir}
echo "Done"
