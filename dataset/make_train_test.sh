#!/usr/bin/env bash
#Create a shuffeled train/test file with 0.2 test split. It scans all directories in the current directory and populates the files.

DIR=$1

if [ -z "${DIR}" ]; then
    echo "No directory to process. Please provide dataset directory. Exiting..."
    exit 1
fi

DIRECTORY=`echo "$(cd "$(dirname ${DIR})"; pwd)/$(basename ${DIR})"`
echo "Processing directory: ${DIRECTORY}"

rm ${DIRECTORY}/files.txt
rm ${DIRECTORY}/labels.txt
rm ${DIRECTORY}/test.txt
rm ${DIRECTORY}/train.txt

folders=$(ls ${DIRECTORY}) #total folders in the directory
k=0;
for f in ${folders}
do
	echo "Scanning : ${DIRECTORY}/$f"
	files=`(ls ${DIRECTORY}/${f}/)`
	for name in ${files}; do
		echo "${DIRECTORY}/$f/${name} $k" >> ${DIRECTORY}/files.txt
	done
	echo "$k ${f%/}" >> ${DIRECTORY}/labels.txt
	k=`expr ${k} + 1`
done

shuf ${DIRECTORY}/files.txt > ${DIRECTORY}/temp.txt
mv ${DIRECTORY}/temp.txt ${DIRECTORY}/files.txt


N=`wc -l < ${DIRECTORY}/files.txt`
split=0.2
split_N=`echo "($N * $split)" | bc`
split_N=${split_N%.*}
remain_N=`echo "($N - $split_N)" | bc`
remain_N_N=${remain_N%.*}
echo "Total=$N, Test=$split=$split_N, Train=$remain_N"

head -n ${remain_N} ${DIRECTORY}/files.txt > ${DIRECTORY}/train.txt
tail -n ${split_N} ${DIRECTORY}/files.txt > ${DIRECTORY}/test.txt

echo "Train file: ${DIRECTORY}/train.txt"
echo "Test file:  ${DIRECTORY}/test.txt"
echo "Total file: ${DIRECTORY}/file.txt"
echo "Label file: ${DIRECTORY}/labels.txt"
