#!/bin/sh
echo "This script will ask for some paths in order to prepare the data to be used to fine-tune a network.";

python fine_tuning_utils/partition_dataset.py -x -i ../input/images/ -r 0.1
echo "Generating tf train record."
python fine_tuning_utils/generate_tf_record.py -x ../input/images/train -l ../input/annotations/label_map.pbtxt -o ../input/annotations/train.record -i ../input/images/train -c ../input/annotations/train.csv
echo "Generating tf train record."
python fine_tuning_utils/generate_tf_record.py -x ../input/images/test -l ../input/annotations/label_map.pbtxt -o ../input/annotations/test.record -i ../input/images/test -c ../input/annotations/test.csv

