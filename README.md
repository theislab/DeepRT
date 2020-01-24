# Self Supervised Retinal Thickness prediction using Deep Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The repository contains the code and partial data for for the manuscript: https://www.biorxiv.org/content/10.1101/861757v1

The repository contains 4 main parts.

* Thickness segmentation
* Thickness map calculation
* Thickness prediction
* SSL Kaggle

## Getting Started
To set up a working example, clone the github repository and install all software requirements listed in the requirements.txt. Main tools used are Python and Tensorflow. Installing within an Anaconda enviroment is recommended.

In order to run illustrative examples four downloads are required from: https://doi.org/10.5281/zenodo.3626020

Below are the four necessary file to download:

* DeepRT_light.tar.gz
* ssl_data_sample.tar.gz
* thickness_segmentation_data.tar.gz
* thickness_segmentation_model.tar.gz

1. Extract DeepRT_light.tar.gz and place the file in ~/ssl_kaggle/pretrained_weights directory.
2. Extract ssl_data_sample.tar.gz and put in the ~/ssl_kaggle as the data folder
3. Extract thickness_segmentation_data.tar.gz put in ~/thickness_segmentation
4. Extract thickness_segmentation_model and put in ~/thickness_map_calculation/output folder

Once repository is set up locally with all code and correctly placed data, follow below instructions for each step.

## Thickness segmentation

To train a new model with same parameters as in manuscript, run main.py. The model is then stored in the ~/thickness_segmentation/logs directory. 

To evaluate the model set the path to your model in the evaluation.py file (model_dir = ...). Then run evaluation.py.

Note: If changes are made to the model in main.py, same changes need to be propagated to the evaluation file such that the weights can load properly. 

**Data:**

Find all images and segmentation masks used in ./thickness_segmentation/data.

The program uses filenames to read the records, see ~/data/file_names_complete folder. The records are pre split between train, validation and test for easy validation.

* ~/data/file_names_complete/test_new_old_mapping.csv
* ~/data/file_names_complete/train_new_old_mapping.csv
* ~/data/file_names_complete/validation_new_old_mapping.csv

These csv files contain a column named "new_id" which correspond to the image ids present in the record folders.

Note: The algorithm is trained on OCT images from both Spectralis and Topcon devices. Which images were obtained with each device is present in the ~/data/file_names_spectralis and ~/data/file_names_topcon folders.

The Topcon images were obtained from the following publication: Menke, M. N., Dabov, S., Knecht, P. & Sturm, V. Reproducibility of retinal thickness measurements in patients with age-related macular degeneration using 3D Fourier-domain optical coherence tomography (OCT) (Topcon 3D-OCT 1000). Acta Ophthalmol. 89, 346–351 (2011)

**Additional training information:**

The main.py file automatically creates a model directory in the ~/thickness_segmentation/logs directory. For housekeeping, the program removed any directory in ~/thickness_segmentation/logs that does not contain a **weights.hdf5** file. 

Note: Any housekeeping features can be remove in utils.py file.

When evaluating the model, evaluation.py will mainly do two things.

* save plots of OCTs, grount truth annotations and predicted annotations masks in the ~/logs/model_dir/test_predictions folder.
* save .csv files with the jaccard index for train, test, validation as well as Topcon and Spectralis records seperately. 

These files are then loaded to create the boxplot from Figure 1a in manuscript.

## Thickness map calculation

To calculate high resolution thickness maps based on OCT DICOM exports store all DICOM files in the ~/thickness_map_calculation/data directory. This code repository contains two examples DICOM files, enabling excecution of code and instruction on how to store the DICOM files one wishes to convert into thickness maps. 
In ~/thickness_map_calculation/data one finds subfolders with the following names:

* PatientPseudoID_Laterality_StudyDate

Where Patient pseudo ID is a randomized ID given to a patient at the time of data export. Laterality referring to the Right or the Left eye and StudyDate being the date of the eye examination.

While these two DICOM files are real examples, they have been completetely anonymized and all patient specific information removed. 

In the ~/output directory, place the trained weights for the OCT segmentation algorithm obtained in the thickness segmentation task.

Also create a ~/thickness_maps directory for saving the calculated thickness maps in numpy format.

To calculate high resolution thickness maps for the DICOM files in the ~/data directory, configure the following parameters in the main.py file:

* img shape = (256, 256, 3)
* save_octs = True
* save_segmentations = True
* thickness_map_dim = 128

Note: img_shape should be the same OCT dimension used in the thickness segmentation task. save_octs and save_segmentations is by default set to true, this saves each individual OCT and tissue segmentation in the DICOM directory. Thickness map dim is set to 128 as determined sufficient for accurate thickness map regression.

As stated in manuscript, the quality of the ground truth thickness map can vary considerably depending on various factors. Most importantly, low quality OCT scans present in real world data can cause low quality examples. For this reason, in ~/quality_filter total variation is calculated for each record and saved in ~/quality_filter/tv_pd.csv. This csv can then be used to filter out to low quality thickness maps from training and evaluation. The value of the choosen threshold will affect the reported results of the thickness regressor. In this project the value 0.014 was choosen after visual inspection of several maps and their quality, on both sides of this threshold.
 
To generate the tv_ps.csv run log_tv.py and ensure that the path to the calculated thickness map is set directly in the log_tv.py.

With the example DICOM files provided all have a total variation below the threshold of 0.014 which corresponds to 4.19 micro meters as stated in the manuscript. 

## Thickness map prediction

In order to run the Thickness prediction model (DeepRT) place all your coregistered fundus images and thickness maps in ~/data/fundus and ~/data/thickness_maps as in data sample provide. Further in ~/data/filenames provide .csv files for train, validation and test samples as in test sample. 

Before training the model, configure params.json, here the default parameters are set as used in manuscript. "enc_filters" here regulates the size of the encoder. In the manuscript the results are achieved using "enc_filters" = 8 for the DeepRT light model. 

Once trained, find model in the ~/logs directory.

To evaluate the model and produce all the test statistics used for the plots in the manuscript, simply run evaluation.py with the params.model_directory set correctly in the python file. This program will generate all necessary test statistics in the results_log.pd file and save it in the specified model directory. Further, all labels and prediction are plotted and saved in the folder "predictions", also residing in the specified model directory.   

Note: in order to produce results on thickness prediction test errors for edema and atrophy cases, put the file gold_standard.csv in the ~/data/gold_standard directory, see test data for example format. Important here is that the file names are of the same format as in other examples.  

## Transfer learning onto the Kaggle Diabetic Retinopathy data set

In order to run the evaluation of transferring DeepRT weights vs. random and Imagenet initialization the first step is to download the data publicly available at: https://www.kaggle.com/c/diabetic-retinopathy-detection/data.

Note: here only the public test set is used for evaluation the algorithms. 

The second step is to pre-process the data as stated in the manuscript. To preprocess the image run preprocess.py in the ~/data directory. The repository includes 5 example images located in ~/data/unprocessed. After running the preprocess.py the output is saved in ~/data/processed.

Once all the downloaded images are preprocessed the data partitions can be created. An example is seen in the ~/data/512_3 directory. Here the test folder always contains the same image accross all partitions. In this example the 3 % partition is included in the repository where each image is resized to 512x512x3 pixels. In this repository the test folder contains the same images as the validation data, to keep the amount of data in the repository low.

Once all data is structured, run main.py specifying the model, weight initialization as well has hyperparameters. 

Note: Using grid search the optimal parameters for ResNet50 with Imagenet Init was found to be learning rate 0.0001 and momentum 0.99. While for DeepRT random and transfer used learning rate 0.001 and momentum 0.9.

Model directories are as usual logged in the ~/logs folder. To evaluate the model on the test data, specify the model directory and model of interest  in the evaluation.py file and run it. A result.txt file with all the main metrics will be saved in the model directory among others. 

### Prerequisites

Manuscript: https://www.biorxiv.org/content/10.1101/861757v1

## Acknowledgments

Thanks to all our collaborators at the LMU eyeclinic and all other collegues contributing with advice and experience. 

