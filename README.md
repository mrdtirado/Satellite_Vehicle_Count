# Satellite_Vehicle_Count

This project utlized both Cars Overhead with Context, as well as Xview2018 Datasets to train object deteciton models focused on detecting vehicles. 

Requirements: Xview Dataset or Carsoverhead dataset
https://gdo152.llnl.gov/cowc/
http://xviewdataset.org/



Useage:
Run **setup.sh **
To sym link tensorflow

Edit **Config.py**
To point to the location of your files, I have left the directories I used as an example

Run**build_space_records.py**
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

If you want to utilize NAIP 1m reoslution data with two year national coverage.
Requirements: AWS, geolocations of desired areas, local verison copy of manifest, as well as every file in the index directories in respective NAIP storage. The script to download all the index files is also in aws_naip.utils
https://registry.opendata.aws/naip/


Run **aws_naip.utils** pointing at your directory of geolocations to grab the file names of all relevant satellite geotiffs.
This will save a pickle of that list, and will take some time to run.

Then run the main function of **naip_processing.py** pointing at the pickle file and a directory of shape files, to crop all the geotiffs to a desired square shape around the geolocation of interest and output geotiffs maintining the metadata.
