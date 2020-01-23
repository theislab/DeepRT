from __future__ import print_function
from keras.optimizers import SGD
import input as i
from utils import *
import pandas as pd
import keras

#load utils classes
params = Params("params.json")
trainops = TrainOps(params)

'''mode to evalutate'''
model_dir = "./logs/3"

'''input pipeline'''
params.data_path = "./data/512_3"

'''Full resnet of DeepRT light'''
model = "DeepRT"

'''load model'''
if model == "DeepRT":
    model = m.resnet_v2(params=params, input_shape=params.img_shape, n=2,num_classes=5)
if model == "Resnet50":
    model = keras.applications.ResNet50(include_top=True, weights=None, input_tensor=None,
                                        input_shape=(512, 512, 3), pooling=None, classes=5)

# load the model
trainops.load_model_test(model, model_dir)

print("loaded trained model under configuration")

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=params.learning_rate, momentum=0.99),
              metrics=['accuracy'])

# print model structure
model.summary()



# get standard configured data generators
train_generator, valid_generator, test_generator = i.create_generators(params.data_path)

# get data number of samples for training
num_training_images, num_validation_images, num_test_images = i.get_data_statistics(params.data_path)

# get labels
lbls = test_generator.classes.tolist()

print("###################### inititing predictions and evaluations ######################")
pred = model.predict_generator(generator=test_generator,
                                     steps=int(num_test_images / (1)),
                                     verbose=1,
                                     use_multiprocessing=False,
                                     workers = 1)

#get predictions and labels in list format
preds = np.argmax(pred,axis=1).tolist()

pd.DataFrame(preds).to_csv(model_dir + "/predictions.csv")
pd.DataFrame(lbls).to_csv(model_dir + "/labels.csv")

#instantiate the evaluation class
evaluation = Evaluation(labels=lbls,
                        predictions=preds,
                        softmax_output=pred,
                        model_dir=model_dir,
                        filenames=test_generator.filenames,
                        params=params)

evaluation.write_plot_evaluation()
#evaluation.plot_examples()
