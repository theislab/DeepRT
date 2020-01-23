import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import input_data as id
import helper as h
import model as m
import os

'''
DEBUG:
look at one pic
plot all in tensorboard
should be able to learn training set perfectly
'''
#write no pyc files

parser = m.parser

parser.add_argument('--train_dir', type=str, default='/home/icb/olle.holmberg/projects/OCT_segmentation/logging_ce_gen_u_net',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--pre_trained_dir', type=str, default='./output/pre_weights',
                    help='Directory where to write event logs and checkpoint.')
parser.add_argument('--max_steps', type=int, default=100000,
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

#data path
seg_dir = '/home/icb/olle.holmberg/projects/OCT_segmentation/data/train/Y/'
im_dir = '/home/icb/olle.holmberg/projects/OCT_segmentation/data/train/X/'
mask_dir = '/home/icb/olle.holmberg/projects/OCT_segmentation/data/train/masks/'
logging_dir = '/home/icb/olle.holmberg/projects/OCT_segmentation/logging_ce_gen_u_net'
#load data
x_batch, y_batch, im_names, masks_batch = id.get_seg_data_gen_u_net(im_dir, seg_dir, mask_dir\
                                                                            ,batch_size=batch_size, pre_processed=True)

print(("Size of:"))
print("- Training-labels:{}".format(y_batch.shape))
print("- Training-images:{}".format(x_batch.shape))

print(("Type of:"))
print("- Training-labels:{}".format(x_batch.dtype))
print("- Training-images:{}".format(y_batch.dtype))

#DATA DIMENSIONS

# We know that MNIST images are 28 pixels in each dimension.# We kn
img_height = 320
img_width = 960
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
#h.plot_images(X=images, y=labels, names=im_names)

#input placeholders
X = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name='X')
y = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name='y')
tf.summary.image('images', X)
tf.summary.image('ground_truth', y)

#border pixel mask
pixel_mask = tf.placeholder(tf.float32, shape=[None, img_height,img_width, None], name="pixel_mask")
#class weights of shape [2,1]
weights = tf.constant([[0.30770313], [0.69229688]], dtype=tf.float32) #inverse class ratio for 1's = 0.69229688
#model
logits = m.generalized_u_net(X)
print("logits shape is {}".format(logits.get_shape()))
#loss
loss = m.loss(logits, labels=y, weight_decay_factor=0.0005, mask=pixel_mask, class_weights=weights)

#optimization method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
#create session
session = tf.Session()
#init global vars
init = tf.global_variables_initializer()
#preidction
prediction  = tf.argmax(tf.nn.softmax(logits), axis=1)
#logging predictions to debugg
#tf.summary.image('prediction', tf.cast(tf.reshape(prediction, [batch_size,img_height,img_width,1]), tf.float32))
tf.summary.histogram('logits', logits)


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

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    avg_acc = []

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        #Get a batch of training examples.
        #x_batch now holds a batch of images and
        #y_true_batch are the true labels for those images.
        x_batch, y_batch, im_names, masks_batch = id.get_seg_data_gen_u_net(im_dir, seg_dir, mask_dir\
                                                                            ,batch_size=batch_size, pre_processed=True)
        #print("labels loaded are {}".format(y_batch))
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {X: x_batch, y: y_batch, pixel_mask: masks_batch}
        #print("mask shape is: {}".format(masks_batch.shape))
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run([optimizer], feed_dict=feed_dict_train)
        #print(centropy, ce_mask)
        # Print status every 100 iterations.
        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
            acc, l = session.run([accuracy_c ,loss], feed_dict=feed_dict_train)
            avg_acc.append(acc)
            # Message for printing.
            msg = "Optimization Iteration:{0:>6}, Training Accuracy:{1:>6.1%}"

            #save variables
            summary_str = session.run(summary_op, feed_dict=feed_dict_train)
            summary_writer.add_summary(summary_str, i)
            #set paths and saving ops for the full and sub_network
            checkpoint_path = os.path.join(logging_dir, 'model.ckpt')
            saver.save(session, checkpoint_path, global_step=i)
            #print accuracy on image
            print(msg.format(i+1, acc))
            print("loss is {}".format(l))
            print("The average accuracy over all bathches are{}".format(np.mean(np.asarray(avg_acc))))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=100000)
