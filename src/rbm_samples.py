"""
Generate and save samples from an rbm trained on digit data
Or maybe several rbms?

Created Docemeber 2013

@authors: James Robert Lloyd (jrl44@cam.ac.uk)
"""

from deep_learning.rbm_label import train_rbm

def train_rbm_on_mnist():
	rbm = train_rbm(learning_rate=0.1, training_epochs=15,
		            n_hidden = 500,
		            dataset='../data/mnist/mnist.pkl.gz')
