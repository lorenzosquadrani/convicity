from cBCM import cBCM
from plasticity.model.optimizer import SGD
from plasticity.model.weights import Normal
from plasticity.utils import view_weights

import pylab as plt
from sklearn.datasets import fetch_openml

#Download the MNIST dataset
X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

# normalize the sample into [0, 1]
X *= 1. / 255

print("FINO A QUA BUONO")
model = cBCM(	n_filters = 8, kernel_size = 5,
				num_epochs= 10, batch_size = 1000, activation = 'relu',
				optimizer = SGD(lr=4e-2), weight_init = Normal(), interaction_strength = 0.,
				random_state = 42, verbose = True)

model.fit()
print("OH YES I CANT BELIEVE IT")



