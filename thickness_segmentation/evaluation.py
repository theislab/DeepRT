import os
import model as mt
from keras.optimizers import *
import pandas as pd
from utils import Params, Logging, TrainOps, Evaluation
from python_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

def return_data_fromcsv_files(params, dataset):
    train_file = os.path.join(params.data_dir, "file_names_complete", "train_new_old_mapping.csv")
    val_file = os.path.join(params.data_dir, "file_names_complete", "validation_new_old_mapping.csv")
    test_file = os.path.join(params.data_dir, "file_names_complete", "test_new_old_mapping.csv")

    train_files = pd.read_csv(train_file, index_col = False)
    val_files = pd.read_csv(val_file, index_col = False)
    test_files = pd.read_csv(test_file, index_col = False)

    if dataset == "topcon":
        train_files = train_files[train_files["old_id"].str.contains("p_")]
        val_files = val_files[val_files["old_id"].str.contains("p_")]
        test_files = test_files[test_files["old_id"].str.contains("p_")]
    if dataset == "spectralis":
        train_files = train_files[~train_files["old_id"].str.contains("p_")]
        val_files = val_files[~val_files["old_id"].str.contains("p_")]
        test_files = test_files[~test_files["old_id"].str.contains("p_")]

    partition = {"train": train_files["new_id"].tolist(),
                 "validation": val_files["new_id"].tolist(),
                 "test": test_files["new_id"].tolist()}
    return partition


# load utils classes
params = Params("params.json")
params.data_dir = "./data"
params.model_directory = "./logs/6"
params.continuing_training = False
params.batchnorm = True
params.is_training = True
params.shuffle = True
params.mode = "test"

# instantiate eval class
evalutaion = Evaluation(params)

# get train ops
trainops = TrainOps(params)

# get model
model = mt.get_deep_unet(params)

'''train and save model'''
save_model_path = os.path.join(params.model_directory, "weights.hdf5")
'''Load models trained weights'''
model.load_weights(save_model_path, by_name = True, skip_mismatch = True)

sets = ["test"]
device = ["spectralis"]

partition = return_data_fromcsv_files(params, dataset = device)

if not os.path.exists(os.path.join(params.model_directory, params.mode + "_predictions")):
    os.makedirs(os.path.join(params.model_directory, params.mode + "_predictions"))

scores = [[], []]

# evaluate all test images
for i in range(len(partition['test'])):
    # Generators
    test_generator = DataGenerator(partition['test'], params)

    image, label = test_generator.get_record(partition["test"][i])

    prediction = model.predict(image)

    # set prediction to classes
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1

    # evaluate
    js = jaccard_score(label.flatten(), prediction.flatten(), average='macro')

    # save image
    evalutaion.plot_examples([image[0, :, :, :], label[0, :, :, 0], prediction[0, :, :, 0]], name = partition["test"][i])

    scores[0].append(str(partition["test"][i]))
    scores[1].append(js)

# save results
result_pd = pd.DataFrame(scores).T
result_pd = result_pd.rename(columns={0:"id", 1:"jaccard"})
result_pd.to_csv(params.model_directory + "/results.csv")

