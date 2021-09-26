from plasticity.model import BCM
import numpy as np

__author__ = ['Lorenzo Squadrani']
__email__ = ['lorenzo.squadrani@studio.unibo.it']


class cBCM (BCM):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        super(cBCM, self).__init__(outputs=out_channels, **kwargs)

    def _make_slices(self, X):

        N, W, H, C = X.shape
        SN, SW, SH, SC = X.strides

        K = self.kernel_size

        view_shape = (N, W - K + 1, H - K + 1, K, K, C)
        view_stride = (SN, SW, SH, SW, SH, SC)

        # Se la batch è X=(N,W,H), il kernel è (K1,K2), lo stride è 1
        # La shape dei sottosample dev'essere (num_samples, W-K1+1, H-K2+1, K, K)
        # Lo stride dei sottosample dev'essere
        # ( bytes * W * H , bytes*H * stride2   ,bytes*stride1 ,bytes * H, bytes)
        # O, in termini dello stride di X = (SN,SW,SH):
        # (SN, SW*stride1, SH*stride2,SW,SH)

        subs = np.lib.stride_tricks.as_strided(X, view_shape, strides=view_stride)

        return subs

    def get_weights(self):
        return self.weights.reshape(self.out_channels, self.kernel_size, self.kernel_size, self.in_channels)

    def fit(self, X):
        '''
        Train the cBCM model on a given dataset of images.

        Parameters
        ----------

        X : numpy.array

            The dataset should be an array of floats with shape (num_samples, width, height, channels)
            or (num_samples, width, height).
        '''

        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        if X.shape[-1] != self.in_channels:
            raise ValueError("The input images do not have the expected number of channels! Got {} but expected {}."
                             .format(X.shape[-1], self.in_channels))

        np.random.seed(self.random_state)

        subs = self._make_slices(X)

        super(cBCM, self).fit(subs.reshape(subs.shape[0], -1))

    def predict(self, X):

        subs = self._make_slices(X)
        weights = self.get_weights()

        # The shape of the output should be (N, OUT_CHANNEL, W-K+1, H-K+1)
        # The shape of weights is (OUT_CHANNEL, IN_CHANNEL, KERNEL_SIZE, KERNEL_SIZE)
        # The shape of subs is (N, W-K+1, H-K+1, C, K , K)
        out = np.einsum('asdjkl,hjkl -> ahsd', subs, weights, optimize=True)

        return self.activation.activate(out)
