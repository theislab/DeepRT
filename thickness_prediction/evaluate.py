from __future__ import print_function
from keras.optimizers import adam
import cv2
import matplotlib.pyplot as plt
import model as mt
import numpy as np
import pandas as pd
import os
import input as i
from utils import Params, TrainOps, Logging
from thickness_map_utils import ThicknessMapUtils

# load utils classes
params = Params("params.json")
params.model_directory = "./logs/4"

# create directory for saving predictions
if not os.path.isdir(params.model_directory + "/predictions"):
    os.makedirs(params.model_directory + "/predictions")

logging = Logging("./logs", params)
train_ops = TrainOps(params)

params.data_path = "./data"
params.architecture = "DeepRT"

model = mt.DeepRT(input_shape = params.img_shape,
                  enc_filters = params.enc_filters,
                  dec_filters = params.dec_filters)

'''Compile model'''
adam = adam(lr=params.learning_rate)

model.compile(optimizer=adam, loss=train_ops.custom_mae, metrics=[train_ops.custom_mae,train_ops.percentual_deviance])
model.summary()

'''train and save model'''
save_model_path = os.path.join(params.model_directory, "weights.hdf5")

'''Load models trained weights'''
model.load_weights(save_model_path)
mae_list = []

# get input generators and statistics
_, _, test_generator = i.get_generators(params)

# get test file names
test_file_names = pd.read_csv(os.path.join(params.data_path, "filenames/test_filenames_filtered.csv"))

# get thickness map utils
maputils = ThicknessMapUtils(None, None, None)
colormap = maputils.heidelberg_colormap()

# create dictionary log for all evaluation values
log = {"record_name":[], "mae": [],"mae_rel": [],"mae_thick": [],"mae_fovea_thick": [],"C0_value": [], "S1_value": [], "S2_value": [],
        "N1_value": [], "N2_value": [], "I1_value": [], "I2_value": [], "T1_value": [], "T2_value": []}

etdrs_names = ["C0_value", "S1_value", "S2_value", "N1_value", "N2_value", "I1_value", "I2_value", "T1_value", "T2_value"]

for rec_iter, record_id in enumerate(test_file_names["ids"].values.tolist()):

    log["record_name"] = record_id
    # get record
    record = test_generator.get_record(record_id)

    # Train model on data set
    prediction = model.predict(record[0].reshape(1, params.img_shape, params.img_shape, 3))

    predicted_thickness_map = prediction[0, :, :, 0]

    # rescale images
    predicted_thickness_map = predicted_thickness_map * 500.
    record[1] = record[1] * 500.

    mae = np.abs(np.mean(predicted_thickness_map - record[1] ))
    mae_rel = mae / np.mean(record[1])

    # append results to log
    log["mae"].append(mae)
    log["mae_rel"].append(mae_rel)

    # get ETDRS values
    etdrs_prediction = maputils.get_low_res_depth_grid_values(predicted_thickness_map)
    etdrs_label = maputils.get_low_res_depth_grid_values(record[1])

    # element wise difference
    etdrs_mae = np.abs(np.subtract(etdrs_label, etdrs_prediction)).tolist()

    # note 1 if record has thick retinal measurements
    if np.max(record[1]) >= 400:
        log["mae_thick"].append(1)
    else:
        log["mae_thick"].append(0)

    # get etdrs max values
    etdrs_max = maputils.get_low_res_depth_grid_maxvalues(record[1])

    # note 1 if record has thick fovea measurements
    if etdrs_max[0] >= 400:
        log["mae_fovea_thick"].append(1)
    else:
        log["mae_fovea_thick"].append(0)

    # add all etdrs values to log
    for iter_, etdrs_value in enumerate(etdrs_mae):
        # iterate from second position
        log[etdrs_names[iter_]].append(etdrs_value)

    # predict the examples
    plt.imsave(params.model_directory + "/predictions/" + record_id + "_label.png", record[1], cmap=colormap)
    plt.imsave(params.model_directory + "/predictions/" + record_id + "_prediction.png",
               np.nan_to_num(predicted_thickness_map),
               cmap=colormap)

    print("test mae is:", np.mean(mae_list))

# load examples with gold standard confirmed atrophy and/or edema
gold_standard_pd = pd.read_csv("./data/gold_standard/gold_standard.csv")

# gather results log
result_pd = pd.DataFrame(log)

# adding gold standard atrophy and edema cases
result_pd = pd.merge(result_pd, gold_standard_pd, how="left")

# saves all results necessary for paper figures in model directory
result_pd.to_csv(params.model_directory + "/results_log.csv")