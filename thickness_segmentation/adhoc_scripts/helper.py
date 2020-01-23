import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import numpy as np
import operator

def get_best_threshold(img_height, img_width,prob_map, labels):
    '''
    :param prob_map: the probabilities for each pixel, numpy array
    :param labels: labels for each pixel, numpy array
    :return: the worst and best accuracy and the thresholds that generated them, Dict
    '''
    im = prob_map
    im_final = np.copy(im)
    labels = labels.flatten()
    #set dict to hold all values for the thresholds
    accuracies = {}
    #generate range of thresholds
    threshholds = np.arange(0.5,1,0.01)
    for thresh in threshholds:
        for idx, val in enumerate(im):
            if val >= thresh:
                im_final[idx]=1
            else:
                im_final[idx]=0
        #derive and append thresholds spec accuracy
        asc = accuracy_score(im_final, labels)
        accuracies[str(thresh)]=asc

    best_acc = max(accuracies.items(), key=operator.itemgetter(1))
    worst_acc = min(accuracies.items(), key=operator.itemgetter(1))
    #print(best_acc, worst_acc)
    return(best_acc, worst_acc)

def get_best_boundry_seperating_threshold(img_height, img_width, prob_map, labels):
    '''
    :param prob_map: the probabilities for each pixel, numpy array
    :param labels: labels for each pixel, numpy array
    :return: the worst and best accuracy and the thresholds that generated them, Dict
    '''
    im = prob_map
    im_final = np.copy(im)
    labels = labels.flatten()
    #set dict to hold all values for the thresholds
    deviances = {}
    #generate range of thresholds
    threshholds = np.arange(0.5,1,0.01)
    for thresh in threshholds:
        for idx, val in enumerate(im):
            if val >= thresh:
                im_final[idx]=1
            else:
                im_final[idx]=0
        #derive and append thresholds spec boundry distance
        deviance = boundry_distance(labels, im_final, img_height, img_width)
        deviances[str(thresh)]=deviance

    #convert dict to numpy, select just the values and remove all none
    deviances = np.array(deviances.items())#select just the values
    deviances = deviances[deviances[:,1] != np.array(None)] # remove None values
    max_dev = deviances[deviances[:,1]==max(deviances[:,1])][0][0]#max(deviances.items(), key=operator.itemgetter(1))
    min_dev = deviances[deviances[:,1]==min(deviances[:,1])][0][0]#min(filter(None,deviances.items()), key=operator.itemgetter(1))
    return(min_dev, max_dev)

def get_boundries(labels,img_height, img_width):
    '''
    :param labels: numpy array: labels
    :param img_height: numpy scalar: height dimension
    :param img_width: numpy scalar: width dimension
    :return:
    '''
    l = labels.reshape(img_height,img_width)
    upper_bound = []
    lower_bound = []
    k = 0
    for x in range(0, l.shape[1]):
        for y in range(0,l.shape[0]):
            if l[y,x] == 1:
                upper_bound.append((y,x))
                k +=1
                break
        for y in range(l.shape[0]-1,0,-1):
            if l[y,x] == 1:
                lower_bound.append((y,x))
                #print(l[y,x])
                k +=1
                break
    upper_bound_array = np.asarray(upper_bound)
    lower_bound_array = np.asarray(lower_bound)
    return(upper_bound_array,lower_bound_array)

def boundry_distance(im1, im2,img_height, img_width):
    '''
    :param im1: numpy array: ground truth
    :param im2: numpy array: predictions
    :return: scalar, deviance btw boundries
    '''
    b11,b12 = get_boundries(im1,img_height, img_width)
    b21,b22 = get_boundries(im2,img_height, img_width)
    #in case thresholding m
    if b11.size != 0 and b21.size != 0 and b12.size != 0 and b22.size != 0:
        upper_diff = np.ediff1d(b11[:,0],b21[:,0])
        lower_diff = np.ediff1d(b12[:,0], b22[:,0])
        total_boundry_deviance = np.sum(np.abs(upper_diff)) + np.sum(np.abs(lower_diff))
    else:
        total_boundry_deviance = None

    return(total_boundry_deviance)

def get_predictions(prob_map, threshhold):
    im = prob_map
    im_final = np.copy(im)
    #set dict to hold all values for the thresholds
    accuracies = {}
    #generate range of thresholds
    for idx, val in enumerate(im):
        if val >= threshhold:
            im_final[idx]=1
        else:
            im_final[idx]=0
    #derive and append thresholds spec accuracy
    print("Im final is {}".format(np.unique(im_final,return_counts=True)))
    return(im_final.astype(int))


def unpadding(im, orig_shape, new_shape):
    '''
    :param im:
    :param orig_shape:
    :param new_shape:
    :return:
    '''
    im = im.reshape(orig_shape)
    result = np.zeros(new_shape)

    # print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((orig_shape[0] - new_shape[0]) / 2)  # 0 would be what you wanted
    y_offset = int((orig_shape[1] - new_shape[1]) / 2)  # 0 in your case

    result = im[x_offset:x_offset+result.shape[0], y_offset:y_offset+result.shape[1]]
    return (result)

def plot_images(X, y, names, cls_pred=None):
    w = 20
    h = 20
    y_1 = y[1, :, :].reshape(160, 400)
    x_1 = X[1, :, :].reshape(160, 400)
    y_2 = y[2, :, :].reshape(160, 400)
    x_2 = X[2, :, :].reshape(160, 400)

    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 2, figsize=(w, h))
    axarr[0, 0].imshow(x_1)
    axarr[0, 0].set_title(names[1])
    axarr[0, 1].imshow(y_1)
    axarr[0, 1].set_title(names[1])
    axarr[1, 0].imshow(x_2)
    axarr[1, 0].set_title(names[2])
    axarr[1, 1].imshow(y_2)
    axarr[1, 1].set_title(names[2])
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.show()


def eval_plot(X, y, predictions, prob_map, logg_dir,names, img_width, img_height, cls_pred=None):
    w = 20
    h = 15

    accuracy = accuracy_score(y, predictions, normalize=True)
    #get threshold with highest accuracy for this image
    best_threshold, worst_threshold = get_best_boundry_seperating_threshold(img_height, img_width, prob_map, labels=y)

    x = X.reshape(img_height,img_width)
    y = y.reshape(img_height,img_width)
    prob_map = prob_map.reshape(1,img_height,img_width)
    p = predictions.reshape(img_height,img_width)

    # unpad the images
    x = unpadding(x, [img_height, img_width], [160, 400])
    y = unpadding(y, [img_height, img_width], [160, 400])
    p = unpadding(p, [img_height, img_width], [160, 400])
    #save prob map

    # Four axes, returned as a 2-d array
    # f, axarr = plt.subplots(1, 3, figsize=(w, h))
    # axarr[0].imshow(x)
    # axarr[0].set_title("Orig image is: " + str(names))
    # axarr[1].imshow(y)
    # axarr[1].set_title("Ground truth is: " + str(names))
    # axarr[2].imshow(p)
    # axarr[2].set_title("Accuracy is of predictions are: "+str(accuracy))
    # plt.subplots_adjust(wspace=0.5, hspace=0)
    save_file = os.path.join(logg_dir,'evaluation', str(names))
    print(save_file)
    plt.savefig(save_file)
    return(best_threshold, worst_threshold)

def eval_plot_threshold(X, y, prob_map, names, logg_dir, threshhold, img_width, img_height,  cls_pred=None):
    w = 20
    h = 15
    #get predictions after threshold
    predictions = get_predictions(prob_map=prob_map, threshhold=threshhold)
    print(np.unique(predictions,return_counts=True), predictions.shape)
    accuracy = accuracy_score(y, predictions, normalize=True)

    x = X.reshape(img_height,img_width)
    y = y.reshape(img_height,img_width)
    p = predictions.reshape(img_height,img_width)

    #unpad the images
    x = unpadding(x,[img_height,img_width], [160,400])
    y = unpadding(y,[img_height,img_width], [160,400])
    p = unpadding(p,[img_height,img_width], [160,400])
    print("the plotted predictions are {}".format(np.unique(p,return_counts=True)))

    #save prob map
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(1, 3, figsize=(w, h))
    axarr[0].imshow(x)
    axarr[0].set_title("Orig image is: " + str(names))
    axarr[1].imshow(y)
    axarr[1].set_title("Ground truth is: " + str(names))
    axarr[2].imshow(p)
    axarr[2].set_title("Accuracy is of predictions are: "+str(accuracy))
    plt.subplots_adjust(wspace=0.5, hspace=0)
    save_file = os.path.join(str(logg_dir),"evaluation", str(names)+"_boundary_thresh_opt")
    print(save_file)
    plt.savefig(save_file)

def eval_plot_clinic_images(X, prob_map, names, logg_dir, threshhold, img_width, img_height, cls_pred=None):
    w = 20
    h = 15
    # get predictions after threshold
    predictions = get_predictions(prob_map=prob_map, threshhold=threshhold)
    print(np.unique(predictions, return_counts=True), predictions.shape)

    x = X.reshape(img_height, img_width)
    p = predictions.reshape(img_height, img_width)

    # unpad the images
    #x = unpadding(x, [img_height, img_width], [400, 160])
    #p = unpadding(p, [img_height, img_width], [400, 160])
    print("the plotted predictions are {}".format(np.unique(p, return_counts=True)))
    names = names.replace('.tif', '')
    # save prob map
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(1,2, figsize=(w, h))
    axarr[0].imshow(x)
    axarr[0].set_title("Orig image is: " + str(names))
    axarr[1].imshow(p)
    axarr[1].set_title("segmentation")
    plt.subplots_adjust(wspace=0.5, hspace=0)
    save_file = os.path.join("/home/olle/PycharmProjects/segmentation_OCT/data/clinic_train", \
                             "evaluation", str(names) + "boundary_thresh_opt_clinic.png")
    print(save_file)
    plt.savefig(save_file)

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
    return (new_weighted_threshold)