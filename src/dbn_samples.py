"""
Generate and save samples from a dbn trained on mnist

Created Decemeber 2013

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os.path

import cloud

from deep_learning.rbm_label import train_rbm

from deep_learning.logistic_sgd import load_data

def train_and_sample_from_dbn(random_seed=1,
                              dataset='bucket/mnist.pkl.gz',
                              epochs=15,
                              architecture=[500,500,2000],
                              samples=1,
                              plot_every=1000):
    # Debug
    random_seed=1
    dataset='bucket/mnist.pkl.gz'
    epochs=1
    architecture=[500,500,2000]
    samples=1
    plot_every=1
    # Setup
    rbms = []
    original_dataset='bucket/mnist.pkl.gz'
    # Pretraining loop
    for (i, n_hidden) in enumerate(architecture):
        # Train
        print 'Training rbm %d' % (i+1)
        (rbm, train_set_x, train_set_y, test_set_x, test_set_y) = train_rbm(learning_rate=0.1, training_epochs=epochs,
                                                                            n_hidden=n_hidden,
                                                                            dataset=dataset,
                                                                            random_seed=random_seed,
                                                                            augment_with_labels=(i==len(architecture)-1))
        rbms.append(rbm)
        if i < len(architecture) - 1:
            print 'Passing data through rbm %d' % (i+1)
            # Pass data through rbm
            # First reload data to get correct object types
            datasets = load_data(original_dataset)
            pseudo_train_set_x, pseudo_train_set_y = datasets[0]
            pseudo_test_set_x, pseudo_test_set_y = datasets[2]
            x_train_array = train_set_x.get_value()
            x_test_array = test_set_x.get_value()
            pseudo_x_train_array = np.zeros((x_train_array.shape[0], architecture[0]))
            pseudo_x_test_array = np.zeros((x_test_array.shape[0], architecture[0]))
            W = rbm.W.get_value()
            bias = np.tile(rbm.hbias.get_value(), (x_train_array.shape[0],1))
            #### TODO - should I be using mean activations or random activations?
            #### Currently using mean activations
            print 'Computing training features'
            pre_sigmoid_activation = np.dot(x_train_array, W) + bias
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            pseudo_x_train_array = hid_prob
            bias = np.tile(rbm.hbias.get_value(), (x_test_array.shape[0],1))
            print 'Computing testing features'
            pre_sigmoid_activation = np.dot(x_test_array, W) + bias
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            pseudo_x_test_array = hid_prob
            pseudo_train_set_x.set_value(pseudo_x_train_array)
            pseudo_test_set_x.set_value(pseudo_x_test_array)
            dataset = (pseudo_train_set_x, pseudo_train_set_y, pseudo_test_set_x, pseudo_test_set_y)
    print 'Pretraining complete'
    # Reload original data
    datasets = load_data(original_dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    # Sampling
    rng = np.random.RandomState(random_seed)

    images = np.zeros((0,28*28))
    labels = np.zeros((0,1))

    number_of_train_samples = train_set_x.get_value(borrow=True).shape[0]

    count = 0

    print 'Sampling images'

    while count < samples:

        # Pick random test example, with which to initialize the persistent chain
        train_idx = rng.randint(number_of_train_samples - 1)
        starting_image = np.asarray(train_set_x.get_value(borrow=True)[train_idx:train_idx+1])

        vis = starting_image
        # Propogate image up the rbms
        for rbm in rbms[:-1]:
            pre_sigmoid_activation = np.dot(vis, rbm.W.get_value()) + rbm.hbias.get_value()
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
        # Append label
        y_list = train_set_y.owner.inputs[0].get_value()
        y_ind = np.zeros((1, 10))
        y_ind[0,y_list[train_idx]] = 1
        vis = np.hstack((vis, y_ind))

        W = rbms[-1].W.get_value()
        h_bias = rbms[-1].hbias.get_value()
        v_bias = rbms[-1].vbias.get_value()
        # Gibbs sample in the autoassociative memory
        for dummy in range(plot_every):
            pre_sigmoid_activation = np.dot(vis, W) + h_bias
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            hid = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            pre_sigmoid_activation = np.dot(hid, W.T) + v_bias
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis.shape[1])) * 1
            # Clamp
            vis[0,-10:] = y_ind
        # Propogate the image down the rbms
        vis = vis[:,:-10]
        for rbm in reversed(rbms[:-1]):
            pre_sigmoid_activation = np.dot(vis, rbm.W.get_value().T) + rbm.vbias.get_value()
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis_prob.shape[1])) * 1

        vis_image = vis_prob

        images = np.vstack((images, vis_image))
        labels = np.vstack((labels, y_list[train_idx]))
        #np.savetxt('images.csv', images, delimiter=',')
        #np.savetxt('labels.csv', labels, delimiter=',')
        count += 1
        print 'Sampled %d images' % count

    return (rbms, images, labels)

def train_and_sample(random_seed):
    (rbms, images, labels) = train_and_sample_from_dbn(random_seed=random_seed)
    return (images, labels)

def main(n_rbms=5, save_folder='../data/mnist/many-rbm-samples/default', cloud_simulation=True):
    pass
    execfile('picloud_misc_credentials.py')
    if cloud_simulation:
        cloud.start_simulator()

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    seeds = [np.random.randint(2**31) for dummy in range(n_rbms)]
    print 'Sending jobs'
    job_ids = cloud.map(train_and_sample, seeds, _type='f2', _cores=1)
    print 'Jobs sent'
    images = np.zeros((0,28*28))
    labels = np.zeros((0,1))
    count = 1
    for (some_images, some_labels) in cloud.iresult(job_ids):
        print 'Job %d of %d complete' % (count, n_rbms)
        count += 1
        images = np.vstack((images, some_images))
        labels = np.vstack((labels, some_labels))
        np.savetxt(os.path.join(save_folder, 'images.csv'), images, delimiter=',')
        np.savetxt(os.path.join(save_folder, 'labels.csv'), labels, delimiter=',')
    return (images, labels)