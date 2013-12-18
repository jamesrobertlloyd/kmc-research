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

def train_dbn(random_seed=1,
              dataset='bucket/mnist.pkl.gz',
              epochs=15,
              architecture=[500,500]):
    # Debug stuff
    epochs = 1
    random_seed=1
    dataset='bucket/mnist.pkl.gz'
    architecture=[500,500]
    # Train initial rbm
    (rbm_1, train_set_x, train_set_y, test_set_x, test_set_y) = train_rbm(learning_rate=0.1, training_epochs=epochs,
                                                                          n_hidden=architecture[0],
                                                                          dataset=dataset,
                                                                          random_seed=random_seed,
                                                                          augment_with_labels=False)
    # Pass data through the rbm
    #### Load data to get correct format
    datasets = load_data(dataset)
    pseudo_train_set_x, pseudo_train_set_y = datasets[0]
    pseudo_test_set_x, pseudo_test_set_y = datasets[2]
    x_train_array = pseudo_train_set_x.get_value()
    x_test_array = pseudo_test_set_x.get_value()
    pseudo_x_train_array = np.zeros((x_train_array.shape[0], architecture[0]))
    pseudo_x_test_array = np.zeros((x_test_array.shape[0], architecture[0]))
    W = rbm_1.W.get_value()
    bias = np.tile(rbm_1.hbias.get_value(), (x_train_array.shape[0],1))
    #### TODO - should I be using mean activations or random activations?
    print 'Computing training features'
    pre_sigmoid_activation = np.dot(x_train_array, W) + bias
    hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
    pseudo_x_train_array = hid_prob
    bias = np.tile(rbm_1.hbias.get_value(), (x_test_array.shape[0],1))
    print 'Computing testing features'
    pre_sigmoid_activation = np.dot(x_test_array, W) + bias
    hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
    pseudo_x_test_array = hid_prob
    pseudo_train_set_x.set_value(pseudo_x_train_array)
    pseudo_test_set_x.set_value(pseudo_x_test_array)
    processed_dataset = (pseudo_train_set_x, pseudo_train_set_y, pseudo_test_set_x, pseudo_test_set_y)
    # Train the second rbm
    (rbm_2, train_set_x, train_set_y, test_set_x, test_set_y) = train_rbm(learning_rate=0.1, training_epochs=epochs,
                                                                          n_hidden=architecture[1],
                                                                          dataset=processed_dataset,
                                                                          random_seed=random_seed,
                                                                          augment_with_labels=True)
    # Return stuff
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    return (rbm_1, rbm_2, train_set_x, train_set_y, test_set_x, test_set_y)

def sample_from_dbn(rbm_1, rbm_2, train_set_x, train_set_y, test_set_x, test_set_y, samples=1, plot_every=1000, random_seed=1):
    # Debug stuff
    samples=1
    plot_every=1000
    random_seed=1
    # Let's a go
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
        pre_sigmoid_activation = np.dot(vis, rbm_1.W.get_value()) + rbm_1.hbias.get_value()
        hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
        vis = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
        # Append label
        y_list = train_set_y.owner.inputs[0].get_value()
        y_ind = np.zeros((1, 10))
        y_ind[0,y_list[train_idx]] = 1
        vis = np.hstack((vis, y_ind))

        W = rbm_2.W.get_value()
        h_bias = rbm_2.hbias.get_value()
        v_bias = rbm_2.vbias.get_value()

        for dummy in range(plot_every):
            pre_sigmoid_activation = np.dot(vis, W) + h_bias
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            hid = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            pre_sigmoid_activation = np.dot(hid, W.T) + v_bias
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis.shape[1])) * 1
            # Clamp
            vis[0,-10:] = starting_image[0,-10:]

        pre_sigmoid_activation = np.dot(vis[0,:-10], rbm_1.W.get_value().T) + rbm_1.vbias.get_value()
        vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))

        vis_image = vis_prob

        images = np.vstack((images, vis_image))
        labels = np.vstack((labels, np.where(starting_image[0,-10:])[0][0]))
        #np.savetxt('images.csv', images, delimiter=',')
        #np.savetxt('labels.csv', labels, delimiter=',')
        count += 1
        print 'Sampled %d images' % count

    # return (images, labels)

def debug():
    # Debug stuff
    epochs = 15
    random_seed=1
    dataset='bucket/mnist.pkl.gz'
    architecture=[500,500]
    # Train initial rbm
    (rbm_1, train_set_x, train_set_y, test_set_x, test_set_y) = train_rbm(learning_rate=0.1, training_epochs=epochs,
                                                                          n_hidden=architecture[0],
                                                                          dataset=dataset,
                                                                          random_seed=random_seed,
                                                                          augment_with_labels=False)
    # Pass data through the rbm
    #### Load data to get correct format
    datasets = load_data(dataset)
    pseudo_train_set_x, pseudo_train_set_y = datasets[0]
    pseudo_test_set_x, pseudo_test_set_y = datasets[2]
    x_train_array = pseudo_train_set_x.get_value()
    x_test_array = pseudo_test_set_x.get_value()
    pseudo_x_train_array = np.zeros((x_train_array.shape[0], architecture[0]))
    pseudo_x_test_array = np.zeros((x_test_array.shape[0], architecture[0]))
    W = rbm_1.W.get_value()
    bias = np.tile(rbm_1.hbias.get_value(), (x_train_array.shape[0],1))
    #### TODO - should I be using mean activations or random activations?
    print 'Computing training features'
    pre_sigmoid_activation = np.dot(x_train_array, W) + bias
    hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
    pseudo_x_train_array = hid_prob
    bias = np.tile(rbm_1.hbias.get_value(), (x_test_array.shape[0],1))
    print 'Computing testing features'
    pre_sigmoid_activation = np.dot(x_test_array, W) + bias
    hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
    pseudo_x_test_array = hid_prob
    pseudo_train_set_x.set_value(pseudo_x_train_array)
    pseudo_test_set_x.set_value(pseudo_x_test_array)
    processed_dataset = (pseudo_train_set_x, pseudo_train_set_y, pseudo_test_set_x, pseudo_test_set_y)
    # Train the second rbm
    (rbm_2, train_set_x, train_set_y, test_set_x, test_set_y) = train_rbm(learning_rate=0.1, training_epochs=epochs,
                                                                          n_hidden=architecture[1],
                                                                          dataset=processed_dataset,
                                                                          random_seed=random_seed,
                                                                          augment_with_labels=True)
    # Return stuff
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    # Debug stuff
    samples=5000
    plot_every=1000
    random_seed=1
    # Let's a go
    rng = np.random.RandomState(random_seed)

    images = np.zeros((0,28*28))
    labels = np.zeros((0,1))

    number_of_train_samples = train_set_x.get_value(borrow=True).shape[0]

    count = 0

    print 'Sampling images'

    while count < samples:

        # pick random test examples, with which to initialize the persistent chain
        train_idx = rng.randint(number_of_train_samples - n_chains)
        starting_image = np.asarray(train_set_x.get_value(borrow=True)[train_idx:train_idx+1])
        print y_list[train_idx]
        #plt.imshow(starting_image.reshape(28,28))
        #plt.show(block=True)

        vis = starting_image
        pre_sigmoid_activation = np.dot(vis, rbm_1.W.get_value()) + rbm_1.hbias.get_value()
        hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
        vis = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
        # Append label
        y_list = train_set_y.owner.inputs[0].get_value()
        y_ind = np.zeros((1, 10))
        y_ind[0,y_list[train_idx]] = 1
        vis = np.hstack((vis, y_ind))

        W = rbm_2.W.get_value()
        h_bias = rbm_2.hbias.get_value()
        v_bias = rbm_2.vbias.get_value()

        for dummy in range(plot_every):
            pre_sigmoid_activation = np.dot(vis, W) + h_bias
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            hid = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            pre_sigmoid_activation = np.dot(hid, W.T) + v_bias
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis.shape[1])) * 1
            # Clamp
            vis[0,-10:] = y_ind#starting_image[0,-10:]
            # Plot        
            #pre_sigmoid_activation = np.dot(vis[0,:-10], rbm_1.W.get_value().T) + rbm_1.vbias.get_value()
            #vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            #plt.imshow(vis_prob.reshape(28,28))
            #plt.show(block=True)

        pre_sigmoid_activation = np.dot(vis[0,:-10], rbm_1.W.get_value().T) + rbm_1.vbias.get_value()
        vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))

        vis_image = vis_prob

        #plt.imshow(vis_image.reshape(28,28))
        #plt.show(block=True)

        images = np.vstack((images, vis_image))
        labels = np.vstack((labels, y_list[train_idx]))
        np.savetxt('images.csv', images, delimiter=',')
        np.savetxt('labels.csv', labels, delimiter=',')
        count += 1
        print 'Sampled %d images' % count


def train_and_sample(random_seed):
    pass
    # result = train_rbm_on_mnist(random_seed=random_seed)
    # (images, labels) = sample_from_rbm(*result, samples=1, plot_every=1000, random_seed=random_seed)
    # return (images, labels)

def main(n_rbms=5, save_folder='../data/mnist/many-rbm-samples/default', cloud_simulation=True):
    pass
    # execfile('picloud_misc_credentials.py')
    # if cloud_simulation:
    #     cloud.start_simulator()

    # #n_rbms = 4
    # #save_folder = 'picloud_test'
    # if not os.path.isdir(save_folder):
    #     os.makedirs(save_folder)

    # seeds = [np.random.randint(2**31) for dummy in range(n_rbms)]
    # print 'Sending jobs'
    # job_ids = cloud.map(train_and_sample, seeds, _type='f2', _cores=1)
    # print 'Jobs sent'
    # images = np.zeros((0,28*28))
    # labels = np.zeros((0,1))
    # count = 1
    # for (some_images, some_labels) in cloud.iresult(job_ids):
    #     print 'Job %d of %d complete' % (count, n_rbms)
    #     count += 1
    #     images = np.vstack((images, some_images))
    #     labels = np.vstack((labels, some_labels))
    #     np.savetxt(os.path.join(save_folder, 'images.csv'), images, delimiter=',')
    #     np.savetxt(os.path.join(save_folder, 'labels.csv'), labels, delimiter=',')
    # return (images, labels)