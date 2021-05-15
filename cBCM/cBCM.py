from plasticity.model import BCM
from sklearn.utils import check_array
import numpy as np

__author__  = ['Lorenzo Squadrani']
__email__ = ['lorenzo.squadrani@studio.unibo.it']

class cBCM (BCM):
    
    def __init__ (self, in_channels, out_channels, kernel_size,
						num_epochs, batch_size, activation,
						optimizer, weights_init, interaction_strength = 0.,
						precision = 1e-30, random_state = 42, verbose = False):

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        super(cBCM, self).__init__(	outputs= out_channels, num_epochs=num_epochs, 
									batch_size=batch_size, activation=activation,
                               		optimizer=optimizer,
                               		weights_init=weights_init,
                               		interaction_strength = interaction_strength,
                               		precision=precision,
                               		random_state=random_state, verbose=verbose)

    def _make_slices(self, X):

        N,C,W,H= X.shape
        SN,SC,SW,SH = X.strides
        
        K = self.kernel_size
        
        view_shape = (N, W-K+1, H-K+1, C, K, K)
        view_stride = (SN, SW, SH, SC, SW , SH)
        
        # Se la batch è X=(N,W,H), il kernel è (K1,K2), lo stride è 1
        # La shape dei sottosample dev'essere (num_samples, W-K1+1, H-K2+1, K, K)
        # Lo stride dei sottosample dev'essere 
        # ( bytes * W * H , bytes*H * stride2   ,bytes*stride1 ,bytes * H, bytes)
        # O, in termini dello stride di X = (SN,SW,SH):
        # (SN, SW*stride1, SH*stride2,SW,SH)
        
        subs = np.lib.stride_tricks.as_strided(X, view_shape, strides=view_stride)

        return subs
    
    def get_weights(self):
        return self.weights.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)


    def fit(self, X):

        if len(X.shape)==3:
            X = np.expand_dims(X, axis = 1)

        if X.shape[1] != self.in_channels:
            raise ValueError("The input images do not have the expected number of channels!")
                 
        
        np.random.seed(self.random_state)

        K, C = self.kernel_size, self.in_channels

        subs = self._make_slices(X)
        
        super(cBCM, self).fit(subs.reshape(-1, C*K*K))
 


    def predict(self, X):

        subs = self._make_slices(X)
        weights = self.get_weights()

        #The shape of the output should be (N, OUT_CHANNEL, W-K+1, H-K+1)
        #The shape of weights is (OUT_CHANNEL, IN_CHANNEL, KERNEL_SIZE, KERNEL_SIZE)
        #The shape of subs is (N, W-K+1, H-K+1, C, K , K)
        out = np.einsum('asdjkl,hjkl -> ahsd', subs, weights, optimize = True)

        return self.activation.activate(out)





    
if __name__ == '__main__':

    from plasticity.model.optimizer import SGD
    from plasticity.model.weights import Normal
    from convicity.utils import view_weights
    from sklearn.datasets import fetch_openml

    #Download the MNIST dataset
    print("Downloading the dataset...")
    X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

    # normalize the sample into [0, 1]
    X *= 1. / 255

    model = cBCM(	out_channels = 8, kernel_size = 5,
				    num_epochs= 10, batch_size = 1000, activation = 'relu',
				    optimizer = SGD(lr=4e-2), weights_init = Normal(), interaction_strength = 0.,
				    random_state = 42, verbose = True)

    print("Starting training...")
    model.fit(np.array(X).reshape(-1,28,28))

    view_weights(model.weights, (5,5))



