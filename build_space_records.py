# import the necessary packages
from config import space_config as config
from utils.tfannotation import TFAnnotation, chip_image, true_bounds
from utils.tfannotation import convertToJpeg, draw_bboxes
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import io
import json
import glob
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def main(_):
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")

    # loop over the classes
    for (k, v) in config.CLASSES.items():
            # construct the class information and write to file
            item = ("item {\n"
                            "\tid: " + str(v) + "\n"
                            "\tname: '" + k + "'\n"
                            "}\n")
            f.write(item)

    # close the output classes file
    f.close()

    # initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    labels = {}
    with open(config.XVIEW_CLASSES) as f:
        for row in csv.reader(f):
            labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

    with open(config.ANNOT_PATH) as f:
        data = json.load(f)

    # get a list of all true images
    imagePath = glob.glob(os.path.sep.join([config.BASE_PATH,"train_images/*.tif"]))
    # grab the filename
    verify_name = [path.split(os.path.sep)[-1] for path in imagePath]


        # loop over each object in the annotation json
    for row in range(len(data['features'])):
        # parse through
        startX, startY, endX, endY = [int(x) for x in data['features'][row]['properties']['bounds_imcoords'].split(',')]
        imagePath = data['features'][row]['properties']['image_id']
        label = data['features'][row]['properties']['type_id']
        
        if imagePath not in verify_name:
            continue
        if label in labels.keys():
            label = labels[label]
        else:
            continue
        # if we are not interested in the label, ignore it
        if label not in config.CLASSES:
            continue
        
            
        # build the path to the input image, then grab any other
        # bounding boxes + labels associated with the image, path,
        # labels and bounding box lists, repsectively
        p = os.path.sep.join([config.BASE_PATH,"train_images/{}".format(imagePath)])

        b = D.get(p, [])

        # build a tuple consisting of the label and bounding box,
        # then update the list and store it in the dictionary
        b.append((label, np.array([startX, startY, endX, endY])))
        D[p] = b

    # create training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
            test_size=config.TEST_SIZE, random_state=42)

    # initalize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
        ]
    for (dType, keys, outputPath) in datasets:
        # initalize the Tensorflow writer and initalize the total
        # number of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)

        # loop over all the keys in the current set
        for k in keys:
            print(k)
            image = cv2.imread(k)
            # grab the bounding boxes for each image
            bounds = [num[1] for num in D[k]]
            true_label = np.array([num[0] for num in D[k]])

            # re order bounding bboxes into (xMin, yMin, xMax, yMax) format
            ordered_bounds = true_bounds(bounds)
            
            print(type(image))
            # chip images into 500 x 500 tiles
            c_img, c_box, c_cls = chip_image(img = image,
                                             coords = ordered_bounds,
                                             classes= true_label,
                                             shape=(500,500))

            filename = k.split(os.path.sep)[-1]

            for i in range(c_img.shape[0]):
                # initialize the TensorFlow writer and initialize the total
                # number of examples written to file
                print("[INFO] processing '{}'...".format(dType))
                writer = tf.python_io.TFRecordWriter(outputPath)
                total = 0

                encoded = convertToJpeg(c_img[i])
                encoded = bytes(encoded)

                chip_name = '{}_{}'.format(i, filename)
                encoding = "jpeg"
                
                print(chip_name)

                # load the shape of each chip
                width = c_img[i].shape[0]
                height = c_img[i].shape[1]


                # initalize the annotation objet used to store
                # information regarding the bounding box + label
                tfAnnot = TFAnnotation()
                tfAnnot.image = encoded
                tfAnnot.encoding = encoding
                tfAnnot.filename = chip_name
                tfAnnot.width = width
                tfAnnot.height = height

                # loop over the bounding boxes and labels associated with each chip
                for j in range(len(c_box[i])):

                    # tensorflow expects your bounding boxes to be in
                    # the range [0, 1] so we scale them
                    if c_cls[i][j] == 0:
                        continue
                    xMin = c_box[i][j][0] / width
                    xMax = c_box[i][j][2] / width
                    yMin = c_box[i][j][1] / height
                    yMax = c_box[i][j][3] / height

                    tfAnnot.xMins.append(xMin)
                    tfAnnot.xMaxs.append(xMax)
                    tfAnnot.yMins.append(yMin)
                    tfAnnot.yMaxs.append(yMax)

                    tfAnnot.textLabels.append(np.str(c_cls[i][j]).encode("utf8"))
                    tfAnnot.classes.append(config.CLASSES[c_cls[i][j]])
                    tfAnnot.difficult.append(0)

                    # increment the total number of examples
                    total += 1

                    # encode the data point attributes using TensorFlow
                    # helper functions
                features = tf.train.Features(feature=tfAnnot.build())
                example = tf.train.Example(features=features)

                    # add the example to the writer
                writer.write(example.SerializeToString())

                # close the writer and print diagnostic information
            writer.close()
            print("[INFO] {} examples saved for '{}'".format(total, dType))


# check to see if the main thread should be started
if __name__ == "__main__":
    tf.app.run()

