import numpy as np
import functools
from sklearn.cluster import KMeans

def construct_rbfn(samples, hidden_shape=None, basis_func="Gaussian"):
    n,d_=samples.shape
    x, y = samples[:, :-1], samples[:, -1]
    if hidden_shape is None:
        hidden_shape = int(np.sqrt(n))
    rbf = RBFN(hidden_shape, basis_func=basis_func)
    rbf.fit(x, y)
    return rbf

class RBFN:
    def __init__(
            self,
            hidden_shape: int,
            norm_func=functools.partial(np.linalg.norm, ord=2, axis=-1),
            basis_func="gaussian"
    ):
        """
        Args:
            hidden_shape: a positive integer, stands for the number of center point of the hidden layer
        """
        self.hidden_shape = hidden_shape
        self.norm_func = norm_func
        self.w = None # the weights of the full connect layer
        self.sigmas = None # sigmas[i] is the sigma of centers, it is the spread radius
        self.centers = None
        self.basis=basis_func
        """
        reference from : https://en.wikipedia.org/wiki/Radial_basis_function
        gaussian from :  MS-RV's RBF
        """
        if basis_func == "gaussian":
            self.basis_func = lambda r: np.exp(-0.5 * (r ** 2))
        elif basis_func == "quadratic":
            self.basis_func = lambda r: r ** 2 + 1.
        elif basis_func == "inquadratic":
            self.basis_func = lambda r: 1 / (r ** 2 + 1.)
        elif basis_func == "multiquadratic":
            self.basis_func = lambda r: np.sqrt(r ** 2 + 1.)
        elif basis_func == "inmultiquadratic":
            self.basis_func = lambda r: 1 / np.sqrt(r ** 2 + 1.)
        else:
            # 默认高斯核
            self.basis_func = lambda r: np.exp(-0.5 * (r ** 2))

    def _calc_sigmas(self) -> np.ndarray:
        """
        compute the hyperparameter `sigma` of kernel function
        """
        # equal to  c_=self.centers[:,np.newsaxis]
        c_ = np.expand_dims(self.centers, 1)
        ds = self.norm_func(c_ - self.centers)
        sigma = 2 * np.mean(ds, axis=1)
        sigma = np.sqrt(0.5) / sigma # if sigma is too large, overfitting
        return sigma

    def _calc_interpolation_mat(self, x: np.ndarray) -> np.ndarray:
        """
        This is the first layer of the radial basis function neural network
        """
        x_ = np.expand_dims(x, 1)
        r = self.norm_func(x_ - self.centers)
        r = r * self.sigmas
        return self.basis_func(r)

    def fit(self, x, y):
        """
            training the rbfn network
        Principle:
            1. kmeans: compute the first layer
            2. least squares method by pseudo-inverse: compute the second layer
        """
        self.centers = KMeans(n_clusters=self.hidden_shape,n_init=10).fit(x).cluster_centers_
        self.sigmas = self._calc_sigmas()
        tmp = self._calc_interpolation_mat(x)
        x_ = np.c_[np.ones(len(tmp)), tmp]
        y = y.reshape((-1, 1))
        # 而 np.linalg.pinv(x)是x的伪逆矩阵
        self.w = np.linalg.pinv(x_) @ y

    def predict(self, x):
        '''

        predict the x

        '''
        if x.ndim == 1:
            x = x.reshape((1, -1))
        tmp = self._calc_interpolation_mat(x)
        x_ = np.c_[np.ones(len(tmp)), tmp]
        y= x_ @ self.w
        return y.ravel()
