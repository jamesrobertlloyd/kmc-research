"""
Generate and save samples from an rbm trained on digit data
Or maybe several rbms?

Created Docemeber 2013

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os.path

import cloud

from deep_learning.rbm_label import train_rbm

def train_rbm_on_mnist(random_seed=1):
    (rbm, train_set_x, train_set_y, test_set_x, test_set_y) = train_rbm(learning_rate=0.1, training_epochs=15,
                                                                        n_hidden = 500,
                                                                        dataset='bucket/mnist.pkl.gz',
                                                                        random_seed=random_seed)
    return (rbm, train_set_x, train_set_y, test_set_x, test_set_y)

def sample_from_rbm(rbm, train_set_x, train_set_y, test_set_x, test_set_y, samples=1, plot_every=1000, random_seed=1):
    rng = np.random.RandomState(random_seed)

    images = np.zeros((0,28*28))
    labels = np.zeros((0,1))

    n_chains = 1

    number_of_train_samples = train_set_x.get_value(borrow=True).shape[0]

    count = 0

    print 'Sampling images'

    while count < samples:

        # pick random test examples, with which to initialize the persistent chain
        train_idx = rng.randint(number_of_train_samples - n_chains)
        starting_image = np.asarray(train_set_x.get_value(borrow=True)[train_idx:train_idx + n_chains])

        vis = starting_image

        for dummy in range(plot_every):
            pre_sigmoid_activation = np.dot(vis, rbm.W.get_value()) + rbm.hbias.get_value()
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            hid = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            pre_sigmoid_activation = np.dot(hid, rbm.W.get_value().T) + rbm.vbias.get_value()
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis.shape[1])) * 1
            # Clamp
            vis[0,-10:] = starting_image[0,-10:]
        vis_image = vis_prob[0,0:(28*28)]

        images = np.vstack((images, vis_image))
        labels = np.vstack((labels, np.where(starting_image[0,-10:])[0][0]))
        #np.savetxt('images.csv', images, delimiter=',')
        #np.savetxt('labels.csv', labels, delimiter=',')
        count += 1
        print 'Sampled %d images' % count

    return (images, labels)

def train_and_sample(random_seed):
    result = train_rbm_on_mnist(random_seed=random_seed)
    (images, labels) = sample_from_rbm(*result, samples=1, plot_every=1000, random_seed=random_seed)
    return (images, labels)

def main(n_rbms=5, save_folder='../data/mnist/many-rbm-samples/default', cloud_simulation=True):
    execfile('picloud_misc_credentials.py')
    if cloud_simulation:
        cloud.start_simulator()

    #n_rbms = 4
    #save_folder = 'picloud_test'
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