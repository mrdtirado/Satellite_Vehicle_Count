# Satellite_Vehicle_Count

This project utlized both Cars Overhead with Context, as well as Xview2018 Datasets to train object deteciton models focused on detecting vehicles. 

Requirements: Xview Dataset or Carsoverhead dataset
https://gdo152.llnl.gov/cowc/
http://xviewdataset.org/



Useage:
Run setup.sh 
To sym link tensorflow

Edit Config.py
To point to the location of your files, I have left the directories I used as an example

Run build_space_records.py 
in tensorflow models/research directory

This will create your records dataset for tensorflow API by chipping large images into multiple small images while keeping track of their respective bounding boxes.

This will allow you to use any tensorflow config located below
https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

along with its pretrained model
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

In order to use yolov3 you can follow the directions here
https://pjreddie.com/darknet/

Along with the easy to use COWC-M instructions that will create the correct label formats for yolov3
https://github.com/LLNL/cowc

