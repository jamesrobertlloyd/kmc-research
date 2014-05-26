"""
Generate and save samples from a dbn trained on mnist

Created Decemeber 2013

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import pickle

from deep_learning.rbm_label import train_rbm

from deep_learning.logistic_sgd import load_data

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def fine_tune_dbn_layers(random_seed=1,
                         dataset='bucket/mnist.pkl.gz',
                         save_folder='./',
                         epochs=50,
                         rbms=[],
                         samples=3000,
                         plot_every=1000,
                         starting_learning_rate=0.002,
                         decay_learning_rate=True,
                         cd_iters=15):
    # Load data - pixels and targets (labels)
    original_dataset = dataset
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    train_digits = train_set_x.get_value()
    y_list = train_set_y.owner.inputs[0].get_value()
    train_labels = np.zeros((train_digits.shape[0], 10))
    for i in range(train_digits.shape[0]):
        train_labels[i,y_list[i]] = 1
    # Setup
    rec_biases = [np.atleast_2d(rbm.hbias.get_value()) for rbm in rbms[:-1]]
    gen_biases = [np.atleast_2d(rbm.vbias.get_value()) for rbm in rbms[:-1]]
    rec_W = [rbm.W.get_value() for rbm in rbms[:-1]]
    gen_W = [rbm.W.get_value().copy().T for rbm in rbms[:-1]]
    pentop = rbms[-1].W.get_value()[:-10,:]
    labtop = rbms[-1].W.get_value()[-10:,:]
    pengenbiases = np.atleast_2d(rbms[-1].vbias.get_value())[:,:-10];
    labgenbiases = np.atleast_2d(rbms[-1].vbias.get_value())[:,-10:];
    topbiases = np.atleast_2d(rbms[-1].hbias.get_value())
    wake_states   = [None] * (len(rbms) - 1)
    sleep_states  = [None] * (len(rbms) - 1)
    p_sleep_probs = [None] * (len(rbms) - 1)
    p_wake_probs  = [None] * (len(rbms) - 1)
    # Fine tuning loop
    for epoch in range(epochs):
        # Permute data
        perm = np.random.permutation(train_digits.shape[0])
        train_digits = train_digits[perm,:];
        train_labels = train_labels[perm,:];
        # Set learning rate
        if decay_learning_rate:
            learning_rate = starting_learning_rate / (epoch + 1);
        for data_iter in range(train_digits.shape[0]):
            # Tell us what's up
            print 'epoch %d of %d, iter %d of %d' % (epoch+1, epochs, data_iter+1, train_digits.shape[0])
            # Extract a digit
            data = train_digits[data_iter,:];
            targets = np.atleast_2d(train_labels[data_iter,:]);
            # PERFORM A BOTTOM-UP PASS TO GET WAKE/POSITIVE PHASE PROBABILITIES AND SAMPLE STATES
            state = data
            for i in range(len(rbms) - 1):
                pre_sigmoid_activation = np.dot(state, rec_W[i]) + rec_biases[i]
                hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
                state = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
                wake_states[i] = state
            pre_sigmoid_activation = np.dot(wake_states[-1], pentop) + np.dot(targets, labtop) + topbiases
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            waketopstates = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            # POSITIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
            poslabtopstatistics = np.dot(targets.T, waketopstates)
            pospentopstatistics = np.dot(wake_states[-1].T, waketopstates)
            # PERFORM GIBBS SAMPLING ITERATIONS USING THE TOP LEVEL UNDIRECTED ASSOCIATIVE MEMORY
            negtopstates = waketopstates
            for cd_iter in range(cd_iters):
                pre_sigmoid_activation = np.dot(negtopstates, pentop.T) + pengenbiases
                hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
                negpenstates = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1

                pre_sigmoid_activation = np.dot(negtopstates, labtop.T) + labgenbiases
                neglabprobs = softmax(pre_sigmoid_activation)

                pre_sigmoid_activation = np.dot(negpenstates, pentop) + np.dot(neglabprobs, labtop) + topbiases
                hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
                negtopstates = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            # NEGATIVE PHASE STATISTICS FOR CONTRASTIVE DIVERGENCE
            negpentopstatistics = np.dot(negpenstates.T, negtopstates)
            neglabtopstatistics = np.dot(neglabprobs.T, negtopstates)
            # STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE PROBABILITIES AND SAMPLE STATES
            sleep_states[-1] = negpenstates
            for i in reversed(range(0, len(rbms)-2)):
                pre_sigmoid_activation = np.dot(sleep_states[i+1], gen_W[i+1]) + gen_biases[i+1]
                hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
                state = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
                sleep_states[i] = state
            pre_sigmoid_activation = np.dot(sleep_states[0], gen_W[0]) + gen_biases[0]
            sleepvisprobs = 1 / (1 + np.exp(-pre_sigmoid_activation))
            # PREDICTIONS
            for i in reversed(range(len(rbms)-2)):
                pre_sigmoid_activation = np.dot(sleep_states[i], rec_W[i+1]) + rec_biases[i+1]
                hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
                p_sleep_probs[i+1] = hid_prob
            pre_sigmoid_activation = np.dot(sleepvisprobs, rec_W[0]) + rec_biases[0]
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            p_sleep_probs[0] = hid_prob
            for i in range(len(rbms)-1):
                pre_sigmoid_activation = np.dot(wake_states[i], gen_W[i]) + gen_biases[i]
                hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
                p_wake_probs[i] = hid_prob
            # UPDATES TO GENERATIVE PARAMETERS
            for i in range(len(rbms)-1):
                eff_data = data if i == 0 else wake_states[i-1]
                gen_W[i] = gen_W[i] + learning_rate * np.dot(wake_states[i].T, (eff_data - p_wake_probs[i]))
                gen_biases[i] = gen_biases[i] + learning_rate * (eff_data - p_wake_probs[i])
            # UPDATES TO TOP LEVEL ASSOCIATIVE MEMORY PARAMETERS
            labtop = labtop + learning_rate*(poslabtopstatistics-neglabtopstatistics);
            labgenbiases = labgenbiases + learning_rate*(targets - neglabprobs);
            pentop = pentop + learning_rate*(pospentopstatistics - negpentopstatistics);
            pengenbiases = pengenbiases + learning_rate*(wake_states[-1] - negpenstates);
            topbiases = topbiases + learning_rate*(waketopstates - negtopstates);
            # UPDATES TO RECOGNITION/INFERENCE APPROXIMATION PARAMETERS
            for i in reversed(range(len(rbms)-1)):
                eff_data = sleep_states[i-1] if i > 0 else sleepvisprobs
                rec_W[i] = rec_W[i] + learning_rate * np.dot(eff_data.T, sleep_states[i] - p_sleep_probs[i])
                rec_biases[i] = rec_biases[i] + learning_rate * (sleep_states[i] - p_sleep_probs[i])

    print 'Fine tuning complete'

    top_W = np.vstack((pentop, labtop))
    gen_biases.append(np.hstack((pengenbiases,labgenbiases)))

    with open(os.path.join(save_folder, 'dbn-ft.pkl'), 'w') as save_file:
        dbnft = {'rec_biases' : rec_biases,
                 'gen_biases' : gen_biases,
                 'rec_W' : rec_W,
                 'gen_W' : gen_W,
                 'pentop' : pentop,
                 'labtop' : labtop,
                 'pengenbiases' : pengenbiases,
                 'labgenbiases' : labgenbiases,
                 'topbiases' : topbiases,
                 'top_W' : top_W}
        pickle.dump(dbnft, save_file)

    # Reload original data
    datasets = load_data(original_dataset)
    train_set_x, train_set_y = datasets[0]
    # Sampling
    rng = np.random.RandomState(random_seed)

    images = np.zeros((0,28*28))
    labels = np.zeros((0,1))

    number_of_train_samples = train_set_x.get_value(borrow=True).shape[0]

    count = 0

    print 'Sampling images'

    while count < samples:

        train_idx = rng.randint(number_of_train_samples - 1)
        starting_image = np.asarray(train_set_x.get_value(borrow=True)[train_idx:train_idx+1])

        y_ind = np.zeros((1, 10))
        y_ind[0,y_list[train_idx]] = 1
        vis = np.hstack((np.random.rand(1, 500), y_ind))

        # Gibbs sample from random start with clamped labels

        for dummy in range(plot_every):
            pre_sigmoid_activation = np.dot(vis, top_W) + topbiases
            hid_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            hid = (hid_prob > np.random.rand(hid_prob.shape[0], hid_prob.shape[1])) * 1
            pre_sigmoid_activation = np.dot(hid, top_W.T) + gen_biases[-1]
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis.shape[1])) * 1
            # Clamp
            vis[0,-10:] = y_ind

        # Propogate down

        vis = vis[:,:-10]
        for (W, bias) in reversed(zip(gen_W, gen_biases)):
            pre_sigmoid_activation = np.dot(vis, W) + bias
            vis_prob = 1 / (1 + np.exp(-pre_sigmoid_activation))
            vis = (vis_prob > np.random.rand(vis_prob.shape[0], vis_prob.shape[1])) * 1

        vis_image = vis_prob # Use probabilities at final layer

        images = np.vstack((images, vis_image))
        labels = np.vstack((labels, y_list[train_idx]))
        np.savetxt(os.path.join(save_folder, 'images-ft.csv'), images, delimiter=',')
        np.savetxt(os.path.join(save_folder, 'labels-ft.csv'), labels, delimiter=',')
        count += 1
        print 'Sampled %d images' % count

def main(intermediate_layers=1):
    # Load pickled data and send to function above
    architecture = [500] * intermediate_layers + [2000]
    save_folder = '../data/mnist/dbn-layers-%s' % '-'.join('%d' % n_neurons for n_neurons in architecture)
    with open(os.path.join(save_folder, 'rbms.pkl'), 'r') as pickle_file:
        rbms = pickle.load(pickle_file)
    fine_tune_dbn_layers(random_seed=1,
                         dataset='bucket/mnist.pkl.gz',
                         save_folder=save_folder,
                         epochs=10,
                         rbms=rbms,
                         samples=3000,
                         plot_every=1000,
                         starting_learning_rate=0.002,
                         decay_learning_rate=True,
                         cd_iters=15)
    