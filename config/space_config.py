# import the necessary packages
import os

# initialize the base path for the Xview dataset
BASE_PATH = "/home/ubuntu/satellite_vehicle_count"

# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "xView_train.geojson"])

# build the path to the ouput training and testing record files,
# along with the class labels files
TRAIN_RECORD = os.path.sep.join([BASE_PATH,
    "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH,
    "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH,
    "records/classes.pbtxt"])

# initalize the test split size
TEST_SIZE = 0.25

# initalize the class labels dictionary
CLASSES = {"Passenger Vehicle":1,"Small Car":2, "Bus":3, "Pickup Truck":4,"Utility Truck":5,"Truck": 6, "Cargo Truck":7, "Truck w/Box":8, "Truck w/Flatbed" :9, "Building": 10, "Vehicle Lot":11, "Damaged Building": 12}
