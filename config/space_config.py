# import the necessary packages
import os

# intialize the base path for the Xview Dataset
BASE_PATH = "capstone"

# build the path to the annotations file

ANNOT_PATH = os.path.sep.join([BASE_PATH, "xView_train.geojson"])

# build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = os.path.sep.join([BASE_PATH,
    "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH,
    "records/testing.record"])
CLASSES_FILES = os.path.sep.join([BASE_PATH,
    "records/classes.pbtxt"])

# initalize the class labels dictionary
CLASSES = {
