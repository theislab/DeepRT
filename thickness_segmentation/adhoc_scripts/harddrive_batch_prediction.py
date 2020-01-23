import tensorflow as tf
import numpy as np
import input_data as id
import model as m
from PIL import Image
import helper as h
import os
import depth_vector as dv
import datetime as dt
im_dir = '/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/augen_clinic_data/image_data'
#clinic train dir
clinic_im_dir = "/home/olle/PycharmProjects/segmentation_OCT/import_env/OCT_segmentation/data/train/X_clinic/"
clinic_seg_dir = "/home/olle/PycharmProjects/segmentation_OCT/import_env/OCT_segmentation/data/train/Y_clinic/"

logging_dir = '/home/olle/PycharmProjects/segmentation_OCT/logging_u_net_no_preprocc_dice/'
import scipy.misc

img_height = 160
img_width = 400
batch_size = 1
import random
#
#
# def iterate_through(im_dir):
#     num_images = 1000
#
#     k = 0
#     oct_image_dirs = []
#     patients = os.listdir(im_dir)
#     for pat in patients:
#         pat_dir = os.path.join(im_dir, pat)
#         studies = os.listdir(pat_dir)
#         for study in studies:
#             study_dir = os.path.join(pat_dir, study)
#             folders = os.listdir(study_dir)
#             for folder in folders:
#                 if folder == "Volume":
#                     volume_dir = os.path.join(study_dir, folder)
#                     image_types = os.listdir(volume_dir)
#                     for image_type in image_types:
#                         if image_type == "OCT":
#                             OCT_dir = os.path.join(volume_dir, image_type)
#                             lateralities = os.listdir(OCT_dir)
#                             for laterality in lateralities:
#                                 image_folder_dir = os.path.join(OCT_dir, laterality)
#                                 image_folders = os.listdir(image_folder_dir)
#                                 for image_folder in image_folders:
#                                     image_dir = os.path.join(image_folder_dir, image_folder)
#                                     images = os.listdir(image_dir)
#                                     for image in images:
#                                         #if "seg" in image:
#                                         #    print("seg in image")
#                                         if "seg" not in image:
#                                             oct_image_dirs.append(os.path.join(image_dir, image))
#                                             k += 1
#                                         if k % 1000 == 0:
#                                             print("Number of images predicted are {}".format(k))
#
#     ints = random.sample(range(1, len(oct_image_dirs)), num_images)
#     # patients[ints]
#     T = [oct_image_dirs[i] for i in ints]
#     print(len(T))
#     return T
#
#oct_image_dirs = iterate_through(im_dir)
#save file
#with open(logging_dir + "/oct_files_left_to_evaluate.txt", "w") as output:
#   output.write(str(oct_image_dirs))

print("DONE WITH THE LIST CREATION")
def unpadding_with_zeros(im, orig_shape, new_shape, batch_size):
    '''
    :param im:
    :param orig_shape:
    :param new_shape:
    :return:
    '''
    #    im = im.reshape(orig_shape)
    result = np.zeros(orig_shape)
    # print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((new_shape[0] - orig_shape[0]) / 2)  # 0 would be what you wanted
    y_offset = int((new_shape[1] - orig_shape[1]) / 2)  # 0 in your case
    #print(x_offset, y_offset)
    result = im[x_offset:im.shape[0] - x_offset, y_offset:im.shape[1] - y_offset]
    return (result)

#input placeholders
X = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name='X')
y = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name='y')

logits = m.u_net(X)
print("logits shape is {}".format(logits.get_shape()))

#create session
session = tf.Session()
#init global vars
init = tf.global_variables_initializer()
#preidction
prediction  = tf.argmax(tf.nn.softmax(logits), axis=1)
probability_map = tf.nn.softmax(logits)
#correct prediction
correct_prediction = tf.equal(prediction, tf.cast(tf.reshape(y,prediction.get_shape(),name=None), tf.int64))
#accuracy
accuracy_c = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Counter for total number of iterations performed so far.# Counte
total_iterations = 0
# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()
# Start the queue runners.
tf.train.start_queue_runners(sess=session)
#set logging to specific location
summary_writer = tf.summary.FileWriter(logging_dir, session.graph)
# Create a saver.
saver = tf.train.Saver(tf.global_variables())

step_start = 0
try:
    ####Trying to find last checkpoint file fore full final model exist###
    print("Trying to restore last checkpoint ...")
    save_dir = logging_dir
    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
    # get the step integer from restored path to start step from there
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    # create init op for the still unitilized variables
    init_new_vars_op = tf.variables_initializer(uninitialized_vars)
    session.run(init_new_vars_op)
except:
    # If all the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore any checkpoints. Initializing variables instead.")
    session.run(init)
###############################################BELOW THE OPTIMAL THRESHOLD OVER 100 IMAGES IS DECIDED
# batch_size = 30
# im_batch, labels_batch, im_displayed = id.get_clinic_train_data(clinic_im_dir, clinic_seg_dir, img_width, img_height, batch_size)
#
# best_threshold_rank = []
# worst_threshold_rank = []
# for i in range(0,batch_size):
#     print(i)
#     #get predictions, prob_mao and labels from tensorflow graph into numpy format
#     feed_dict_train = {X: im_batch[i].reshape(1,img_height,img_width,1), y: labels_batch[i].reshape(1,img_height,img_width,1)}
#     pred, cp, label, prob_map = session.run([prediction, correct_prediction, y, probability_map], feed_dict=feed_dict_train)
#     #derive the best thresholds for the validation set
#     best_threshold, worst_threshold = h.eval_plot(im_batch[i], labels_batch[i].reshape(img_height*img_width), pred
#                                                   , prob_map[:,1],logging_dir,im_displayed[i]
#                                                   ,img_width, img_height)
#     best_threshold_rank.append(best_threshold)
#     worst_threshold_rank.append(worst_threshold)
# #get the optimal weighted threshold
# new_weighted_threshold = h.get_best_weighted_threshold(best_threshold_rank, worst_threshold_rank)
# print("THE NEW WEIGHTED THRESHOLD IS {}".format(new_weighted_threshold))

###############################################BELOW THE PREDICTION FOR EACH 1000 IMAGES IS MADE
k = 0
#create a copy of the full list for bookkeeping
#print("The number of images quied are:{}".format(len(oct_image_dirs)))
#write all selected paths to logging file
#images_left_to_complete = list(oct_image_dirs)

#with open(logging_dir + "/oct_files_left_to_evaluate.txt", "w") as output:
#    output.write(str(images_left_to_complete))

completed_images = []
#bookkeeping
input_time = []
prediction_time = []
saving_time = []
number_of_faulty_images = 0
# #opening the images_left_to_complete left to predict on
# images_left_to_complete = open(logging_dir + "/oct_files_left_to_evaluate.txt", "r")
# images_left_to_complete = images_left_to_complete.read()
# images_left_to_complete = images_left_to_complete.replace("[","")
# images_left_to_complete = images_left_to_complete.replace("]","")
# images_left_to_complete = images_left_to_complete.replace("'","")
# images_left_to_complete = images_left_to_complete.replace(" /","/")
# images_left_to_complete = images_left_to_complete.split(",")
images_left_to_complete = ['/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/'
                           'augen_clinic_data/image_data/608/2013-06-27/Volume/OCT/R/11-19-41/13.tif']
print("The number of images_left_to_complete left to predict on is: {}".format(len(images_left_to_complete)))
while images_left_to_complete:
    try:
        #init image list
        image_list = []
        image_names = []
        main_dirs = []
        #select batch size of paths
        image_paths = images_left_to_complete[0:batch_size]
        #measure time to load data
        starting_data_loading = dt.datetime.now()
        for path in image_paths:
            #bookkeep the processed images
            completed_images.append(path)
            #remove selected paths from main list
            images_left_to_complete.remove(path)
            #get image name
            image_names.append(path.split("/")[-1])
            #get image main dir
            main_dirs.append("/".join(path.split("/")[0:-1]))
            #retrieve the image and save it to image list
            im, new_shape, orig_shape = id.get_clinic_data_hardrive(path, img_width, img_height)
            image_list.append(im)
        #end meaduring load time
        ending_data_loading = dt.datetime.now()
        # append the time it takes to list
        input_time.append(abs((ending_data_loading.microsecond - starting_data_loading.microsecond)) / 1e6)
        #create numpy array of image list
        image_array = np.asarray(image_list).reshape(batch_size, img_height, img_width, 1)
        starting_prediction = dt.datetime.now()
        #set the feed dict
        feed_dict_train = {X: image_array}
        #predict
        pred, prob_map = session.run([prediction, probability_map],
                                     feed_dict=feed_dict_train)
        ending_prediction = dt.datetime.now()
        prediction_time.append(abs((ending_prediction.microsecond - starting_prediction.microsecond)) / 1e6)
        # print("the network predicts {}".format(pred.shape))
        pred = pred.reshape(batch_size, 160, 400).astype(np.int32)

        im_batch = []
        #resize all images
        for i in range(0, pred.shape[0]):
            im_resized = np.asarray(Image.fromarray(pred[i]).resize([new_shape[1], new_shape[0]]))
            im_resized = unpadding_with_zeros(im_resized, orig_shape, new_shape, batch_size)
            im_batch.append(im_resized)
        im_batch = np.asarray(im_batch, dtype=np.int32).reshape(batch_size, orig_shape[0], orig_shape[1])
        #measure saving time
        starting_saving = dt.datetime.now()
        for i in range(0,pred.shape[0]):
            depth_vector = dv.get_depth_vector(im_batch[i])

            np.save(main_dirs[i] + "/DV_" + image_names[i], depth_vector)
            k += 1
            # save file of completed images
            with open(logging_dir + "/oct_files_left_to_evaluate.txt", "w") as output:
                output.write(str(images_left_to_complete))
            with open(logging_dir + "/evaluated_oct_files.txt", "w") as output:
                output.write(str(completed_images))
            # print to see the bookeping work
        ending_saving = dt.datetime.now()
        saving_time.append(abs((ending_saving.microsecond-starting_saving.microsecond))/1e6)
        print("loading time is {}, prediction_time is {}, saving time is {}".format(np.mean(input_time),
                                                                                    np.mean(prediction_time)
                                                                                    , np.mean(saving_time)))
    except:
        number_of_faulty_images += 1
        print("Number of faulty images are: {}".format(number_of_faulty_images))
            #unpad the zeros
print("loading time is {}, prediction_time is {}, saving time is {}".format(np.mean(input_time),np.mean(prediction_time)
                                                                       ,np.mean(saving_time)))
