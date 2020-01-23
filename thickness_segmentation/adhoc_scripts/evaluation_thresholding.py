import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import input_data as id
import model as m
import helper as h
import os
#write no pyc files

parser = m.parser

parser.add_argument('--train_dir', type=str, default='/home/olle/PycharmProjects/segmentation_OCT/logging_u_net_no_preprocc_dice_fourth_round/',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--pre_trained_dir', type=str, default='./output/pre_weights',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')
parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')
parser.add_argument('--weight_decay_rate', help='Weight decay rate',
                    type=float, default=0.0005)

FLAGS = parser.parse_args()

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128

#number of iterations to train
num_iterations = 100

#batch size
batch_size = 1
#training data set path
seg_dir_train = '/home/olle/PycharmProjects/segmentation_OCT/import_env/OCT_segmentation/data/train/Y/'
im_dir_train = '/home/olle/PycharmProjects/segmentation_OCT/import_env/OCT_segmentation/data/train/X/'
mask_dir_train = '/home/olle/PycharmProjects/segmentation_OCT/import_env/OCT_segmentation/data/train/masks/'
#validatiaon data set path
seg_dir = '/home/olle/PycharmProjects/segmentation_OCT/data/clinic_train/testing_data/test_labels/'
im_dir = '/home/olle/PycharmProjects/segmentation_OCT/data/clinic_train/testing_data/test_images/'
logging_dir = '/home/olle/PycharmProjects/segmentation_OCT/logging_u_net_no_preprocc_dice_fourth_round/'
result_logging = "/home/olle/PycharmProjects/segmentation_OCT/data/clinic_train/testing_data/"
clinic_data_dir = '/home/olle/PycharmProjects/segmentation_OCT/data/clinic_train/unannotated_test/'
#DATA DIMENSIONS

img_height = 160
img_width = 400
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_height * img_width

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_height, img_width)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes
num_classes = 2

#plot a few images to see if correct
'''
TODO: Fix the helper plotting function to plot images
'''
def get_best_weighted_threshold(best_thresholds, worst_thresholds):
    '''
    :param best_threshold_rank: the best thresholds per pictures, LIST
    :param worst_threshold_rank: the worst thresholds per picture, LIST
    :return: a scalar with the the best weighted threshold
    '''
    ###HERE we optimize thresholding###
    best_thresh_array = np.asarray(best_thresholds).astype(np.float64)
    worst_thresh_array = np.asarray(worst_thresholds).astype(np.float64)
    threshold_importance = np.subtract(best_thresh_array, worst_thresh_array)
    weights = np.abs(np.around(threshold_importance, decimals=2))
    weighted_thesholds = np.multiply(best_thresh_array, weights)
    new_weighted_threshold = np.divide(np.sum(weighted_thesholds), np.sum(weights))
    return(new_weighted_threshold)

#h.plot_images(X=images, y=labels, names=im_names)

#input placeholders
X = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name='X')
y = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name='y')
#border pixel mask
#pixel_mask = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name="pixel_mask")
#class weights of shape [2,1]
#weights = tf.constant([[0.30770313], [0.69229688]], dtype=tf.float32)
#model
logits = m.u_net(X)
print("logits shape is {}".format(logits.get_shape()))

#create session
session = tf.Session()
#init global vars
init = tf.global_variables_initializer()
#preidction
#here we derive the prediction based on threshold values 0.8
'''
threshold = 0.8
softmax_logits = tf.nn.softmax(logits)
gt_mask = tf.greater(softmax_logits, threshold)
lt_mask = tf.less(softmax_logits, threshold)
maxed = tf.where (gt_mask, tf.ones_like(softmax_logits), softmax_logits)
predictions = tf.where (lt_mask, tf.zeros_like(maxed), maxed)
argmax = tf.reduce_max(predictions, axis=1)
prediction = tf.cast(argmax, dtype = tf.int64)
'''
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
    save_dir = FLAGS.train_dir
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

#############OPERATIONS ON THE TRAINING SET#############################################################################
# NUM_TRAIN_IMAGES = 100
# im_batch, labels_batch, im_displayed = id.get_seg_data_gen_u_net_eval(im_dir_train, seg_dir_train
#                                                                          , mask_dir_train,NUM_TRAIN_IMAGES,
#                                                                                    pre_processed=True)
# best_threshold_rank = []
# worst_threshold_rank = []
# for i in range(0,NUM_TRAIN_IMAGES):
#     print(i)
#     #get predictions, prob_mao and labels from tensorflow graph into numpy format
#     feed_dict_train = {X: im_batch[i].reshape(1,img_height,img_width,1), y: labels_batch[i].reshape(1,img_height,img_width,1)}
#     pred, cp, label, prob_map = session.run([prediction, correct_prediction, y, probability_map], feed_dict=feed_dict_train)
#
#     best_threshold, worst_threshold = h.eval_plot(im_batch[i], labels_batch[i].reshape(img_height * img_width), pred, prob_map[:, 1],
#                                                   logging_dir+"train/", im_displayed[i], img_width, img_height)
#     best_threshold_rank.append(best_threshold[1])
#     worst_threshold_rank.append(worst_threshold[1])

############OPERATIONS ON THE VALIDATION SET##########################################################################
NUM_IMAGES=10
im_batch, labels_batch, im_displayed = id.get_clinic_eval_data(im_dir, seg_dir, img_height,img_width)
print("Images loaded are {}".format(im_displayed))
# Put the batch into a dict with the proper names for placeholder variables in the TensorFlow graph.
best_threshold_rank = []
worst_threshold_rank = []
for i in range(0,NUM_IMAGES):
    print(i)
    #get predictions, prob_mao and labels from tensorflow graph into numpy format
    feed_dict_train = {X: im_batch[i].reshape(1,img_height,img_width,1), y: labels_batch[i].reshape(1,img_height,img_width,1)}
    pred, cp, label, prob_map = session.run([prediction, correct_prediction, y, probability_map], feed_dict=feed_dict_train)
    #derive the best thresholds for the validation set
    best_threshold, worst_threshold = h.eval_plot(im_batch[i], labels_batch[i].reshape(img_height*img_width), pred
                                                  , prob_map[:,1],result_logging,im_displayed[i]
                                                  ,img_width, img_height)
    best_threshold_rank.append(best_threshold)
    worst_threshold_rank.append(worst_threshold)
#get the optimal weighted threshold
new_weighted_threshold = get_best_weighted_threshold(best_threshold_rank, worst_threshold_rank)

print("THE NEW WEIGHTED THRESHOLD IS {}".format(new_weighted_threshold))
#Go through all the images again and make classification based on best threshold
for i in range(0, NUM_IMAGES):
    feed_dict_train = {X: im_batch[i].reshape(1, img_height, img_width, 1), y: labels_batch[i].reshape(1, img_height, img_width, 1)}
    # print("mask shape is: {}".format(masks_batch.shape))
    # Run the optimizer using this batch of training data.
    # TensorFlow assigns the variables in feed_dict_train
    # to the placeholder variables and then runs the optimizer.
    pred, cp, label, prob_map = session.run([prediction, correct_prediction, y, probability_map],
                                            feed_dict=feed_dict_train)

    print("the network predicts {}".format(np.unique(pred,return_counts=True)))

    h.eval_plot_threshold(im_batch[i], labels_batch[i].reshape(img_height * img_width), prob_map[:,1], im_displayed[i],
                          result_logging, new_weighted_threshold, img_width, img_height)

# # ################
clinic_images,clinic_images_names = id.load_clinic_images(im_dir, img_width, img_height
                                                                              ,pre_processed=False)
#new_weighted_threshold = 0.8
NUM_CLINICAL_IMAGES= len(os.listdir(clinic_data_dir))
for i in range(0,NUM_CLINICAL_IMAGES):
    feed_dict_train = {X: clinic_images[i].reshape(1, img_height, img_width, 1)}
    # print("mask shape is: {}".format(masks_batch.shape))
    # Run the optimizer using this batch of training data.
    # TensorFlow assigns the variables in feed_dict_train
    # to the placeholder variables and then runs the optimizer.
    pred, prob_map = session.run([prediction,probability_map],
                                            feed_dict=feed_dict_train)

    print("the network predicts {}".format(np.unique(pred,return_counts=True)))

    h.eval_plot_clinic_images(clinic_images[i], prob_map[:,1], clinic_images_names[i],
                              result_logging, new_weighted_threshold, img_width, img_height)