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


def figure1(data, model_dir):
    '''
    data is a list of lists containing train validation and test results for Spectralies and Topcon OCTs seperately
    [result_dict["train_spectralis"].tolist(),
    result_dict["train_topcon"].tolist(),
    result_dict["validation_spectralis"].tolist(),
    result_dict["validation_topcon"].tolist(),
    result_dict["test_spectralis"].tolist(),
    result_dict["test_topcon"].tolist()]
    '''

    plt.figure(figsize = (30, 10))
    fig7, ax7 = plt.subplots(figsize = (20, 10))

    bplot_ = ax7.boxplot(data, showmeans = False, patch_artist = True)

    colors = ["black", "darkred", "black", "darkred", "black", "darkred"]
    for patch, color in zip(bplot_['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(bplot_["medians"], color = "gold", linewidth = 3)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(
        axis = 'x',  # changes apply to the x-axis
        which = 'both',  # both major and minor ticks are affected
        bottom = False,  # ticks along the bottom edge are off
        top = False,  # ticks along the top edge are off
        labelbottom = False)

    # fill with colors
    # fill with colors
    plt.savefig(model_dir + "/segmentation_results.png", dpi = 450, transparent = True)


# load utils classes
params = Params("params.json")
params.data_dir = "./data"
params.model_directory = "./output"
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

# store predictions
if not os.path.exists(os.path.join(params.model_directory, params.mode + "_predictions")):
    os.makedirs(os.path.join(params.model_directory, params.mode + "_predictions"))


# store result files
if not os.path.exists(os.path.join(params.model_directory + "/results")):
    os.makedirs(os.path.join(params.model_directory + "/results"))

sets_ = ["test", "train", "validation"]

# spectralis OR topcon
devices = ["topcon","spectralis"]

for device in devices:

    # get file names for device
    partition = return_data_fromcsv_files(params, dataset = device)

    for set_ in sets_:

        # log lists
        scores = [[], []]

        # evaluate all test images
        for i in range(len(partition[set_])):
            # Generators
            test_generator = DataGenerator(partition[set_], params)

            image, label = test_generator.get_record(partition[set_][i])

            prediction = model.predict(image)

            # set prediction to classes
            prediction[prediction < 0.5] = 0
            prediction[prediction >= 0.5] = 1

            # evaluate
            js = jaccard_score(label.flatten(), prediction.flatten(), average='macro')

            # save image
            evalutaion.plot_examples([image[0, :, :, :], label[0, :, :, 0], prediction[0, :, :, 0]], name = partition[set_][i])

            scores[0].append(str(partition[set_][i]))
            scores[1].append(js)

        # save results
        result_pd = pd.DataFrame(scores).T
        result_pd = result_pd.rename(columns={0:"id", 1:"jaccard"})
        result_pd.to_csv(params.model_directory + "/results/results_{}_{}.csv".format(set_, device))


result_paths = ["results_test_spectralis.csv",
                "results_train_spectralis.csv",
                "results_validation_spectralis.csv",
                "results_test_topcon.csv",
                "results_train_topcon.csv",
                "results_validation_topcon.csv"]

result_dict = {}

for result_path in result_paths:
    result_dict[result_path.replace(".csv","")] = pd.read_csv(os.path.join(params.model_directory + "/results",
                                                                           result_path))["jaccard"]

data = [result_dict["results_test_spectralis"].tolist(),
        result_dict["results_train_spectralis"].tolist(),
        result_dict["results_validation_spectralis"].tolist(),
        result_dict["results_test_topcon"].tolist(),
        result_dict["results_train_topcon"].tolist(),
        result_dict["results_validation_topcon"].tolist()]

# create figure 1 panel a from paper and save in model directory
figure1(data, os.path.join(params.model_directory + "/results"))