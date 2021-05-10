from plasticity.model import BCM
from sklearn.utils import check_array
import numpy as np

__author__  = ['Lorenzo Squadrani']
__email__ = ['lorenzo.squadrani@studio.unibo.it']

class cBCM (BCM):
    
    def __init__ (self, n_filters, kernel_size,
						num_epochs, batch_size, activation,
						optimizer, weights_init, interaction_strength = 0.,
						precision = 1e-30, random_state = 42, verbose = False):

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        
        outputs = n_filters

        super(cBCM, self).__init__(	outputs=outputs, num_epochs=num_epochs, 
									batch_size=batch_size, activation=activation,
                               		optimizer=optimizer,
                               		weights_init=weights_init,
                               		interaction_strength = interaction_strength,
                               		precision=precision,
                               		random_state=random_state, verbose=verbose)
    

    def fit(self, X, y = None):
        
        if y is not None:
            X = self._join_input_label(X=X, y=y)
            
        
        np.random.seed(self.random_state)
        
        N,W,H = X.shape
        SN,SW,SH = X.strides
        
        K = self.kernel_size
        
        view_shape = (N, W-K+1, H-K+1, K,K)
        view_stride = (SN, SW, SH, SW , SH)
        
    	# Se la batch è X=(N,W,H), il kernel è (K1,K2), lo stride è 1
    	# La shape dei sottosample dev'essere (num_samples, W-K1+1, H-K2+1, K, K)
    	# Lo stride dei sottosample dev'essere 
        # ( bytes * W * H , bytes*H * stride2   ,bytes*stride1 ,bytes * H, bytes)
    	# O, in termini dello stride di X = (SN,SW,SH):
    	# (SN, SW*stride1, SH*stride2,SW,SH)
		
        subs = np.lib.stride_tricks.as_strided(X, view_shape, strides=view_stride)
        
        super(cBCM, self).fit(subs.reshape(N*(W-K+1)*(W-K+1), K*K))

    
if __name__ == '__main__':

    from plasticity.model.optimizer import SGD
    from plasticity.model.weights import Normal
    from plasticity.utils import view_weights
    from sklearn.datasets import fetch_openml

    #Download the MNIST dataset
    print("Downloading the dataset...")
    X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

    # normalize the sample into [0, 1]
    X *= 1. / 255

    model = cBCM(	n_filters = 8, kernel_size = 5,
				    num_epochs= 10, batch_size = 1000, activation = 'relu',
				    optimizer = SGD(lr=4e-2), weights_init = Normal(), interaction_strength = 0.,
				    random_state = 42, verbose = True)

    print("Starting training...")
    model.fit(np.array(X).reshape(-1,28,28))

    view_weights(model.weights, (5,5))



