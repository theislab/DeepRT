from keras import Input
import os
import numpy as np
import cv2
import glob
from model import get_deep_unet
from segmentations import Segmentations
from dicom_table import DicomTable
from map import Map
import matplotlib.pyplot as plt
params = {}

# configuration parameters
dicom_path = "./data"
params["segmentation_model_path"] = "./output/"

# set path where maps should be saved
params["thickness_map_path"] = "./thickness_maps"

# oct image dimensions -  should be same used in thickness segmentation
params["img_shape"] = (512, 512, 3)

# save oct and segmentation in dicom dir
params["save_octs"] = True
params["save_segmentations"] = True

# setting thickness map dim to 128 pixels
params["thickness_map_dim"] = 128

# set model paramaters
params["n_filters"] = 32
params["batchnorm"] = True
params["dropout"] = 0.2
params["is_training"] = False

# retrieve full paths to dicom files
dicom_paths = glob.glob(dicom_path+"/*/*.dcm")

# set path where segmenation algorithm is saved
save_model_path = os.path.join(params["segmentation_model_path"], "weights.hdf5")

# load model config and weights
model = get_deep_unet(params)
model.load_weights(save_model_path, by_name = True, skip_mismatch=True)

# iterate through dicom files and generate map for each
for dicom_path in dicom_paths:

    # get full dicom information
    dicom = DicomTable(dicom_path)

    # if arguments in dicom are faulty, None is returned
    if dicom.record_lookup is not None:

        # retrieve all segmentations
        segmentations = Segmentations(dicom, model, params)

        # set path in dicom dir to save octs
        dicom_dir = "/".join(dicom_path.split( "/" )[:-1])

        if params["save_segmentations"]:
            # save all segmentations in ./dicom_dir/segmentations/
            segmentations.save_segmentations(os.path.join(dicom_dir, dicom.record_id+"_segmentations"))

        if params["save_octs"]:
            # save octs in ./dicom_dir/octs/
            segmentations.save_octs(save_path = os.path.join(dicom_dir, dicom.record_id+"_octs"))

        # calculate the retinal thickenss map
        map_ = Map(dicom, segmentations.oct_segmentations, dicom_path)

        # initialize calculation of thickness map
        map_.depth_grid(interpolation = "linear")

        # plot thickness map
        map_.plot_thickness_map(os.path.join(dicom_dir, dicom.record_id+"_thickness_map.png"))

        # save thickness map
        np.save(os.path.join(params["thickness_map_path"],
                             dicom.record_id + ".npy"),
                             cv2.resize(map_.thickness_map,
                             (params["thickness_map_dim"], params["thickness_map_dim"]),
                             interpolation = cv2.INTER_NEAREST))
