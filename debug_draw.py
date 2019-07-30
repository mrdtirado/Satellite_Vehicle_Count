# import the necessary packages
from config import space_config as config
from utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import json
import glob
import csv
import numpy as np
import cv2


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

        # loop over each object in the annotation json
    for row in range(len(data['features'])):
        # parse through
        startX, startY, endX, endY = [int(x) for x in data['features'][row]['properties']['bounds_imcoords'].split(',')]
        imagePath = data['features'][row]['properties']['image_id']
        label = data['features'][row]['properties']['type_id']
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
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    # create training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()),
            test_size=config.TEST_SIZE, random_state=42)

    # initalize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
        ]
    
    # loop over the datasets
    for (dType, keys, outputPath) in datasets:
        # initalize the Tensorflow writer and initalize the total
        # number of examples written to file
        print("[INFO] processing '{}'...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0

        test_draw = ['/home/ubuntu/satellite_vehicle_count/train_images/2207.tif']

        # loop over all the keys in the current set
        for k in test_draw:
            # load the input image from disk as TensorFlow object
            encoded = tf.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)

            # load the image from disk again, this time as a PIL object
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            # initalize the annotation object used to store
            # information regarding the bounxing box + labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            image = cv2.imread(k)

            # loop over the bounding boxes + labels assocaited with the image
            for (label, (startX, startY, endX, endY)) in D[k]:
                # Tensorflow assumes all bounxing boxes are in the
                # range [0, 1] so we need to scale them
                coords = np.array([[startX, startY], [endX, endY]])

                xMin, xMax = coords[:, 0].min(), coords[:, 0].max()
                yMin, yMax = coords[:, 1].min(), coords[:, 1].max()
                
                # xMin = startX / w
                # xMax = endX / w
                # yMin = startY / h
                # yMax = startY / h

                

                # load the input image from disk and denormalize the
                # bounding box coordinates
                startX = int(xMin)
                startY = int(yMin)
                endX = int(xMax)
                endY = int(yMax)

                print("[INFO] Drawing Boxes on Images")
                # draw the bounding boxes on the image
                image = cv2.rectangle(image, (xMin, yMin), (xMax, yMax),
                               (0, 255, 0), 2)
                cv2.imwrite('/home/ubuntu/satellite_vehicle_count/{}'.format(
                        filename), image)

# check to see if the main thread should be started
if __name__ == "__main__":
    tf.app.run()
