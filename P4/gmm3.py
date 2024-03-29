import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float)
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array)
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means, R, _ = KMeans(self.n_cluster, self.max_iter, self.e).fit(x)
            self.variances = np.zeros((self.n_cluster, D, D))
            self.pi_k = np.zeros(self.n_cluster)
            Nk = np.sum(np.identity(self.n_cluster)[R], axis=0)
            for k in range(self.n_cluster):
                x_mk = x - self.means[k]
                x_mk = x_mk[R == k]
                self.variances[k] = np.dot(np.transpose(x_mk), x_mk) / Nk[k]
            self.pi_k = Nk / N
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means = np.random.rand(self.n_cluster, D)
            self.variances = np.full((self.n_cluster, D, D), np.identity(D))
            self.pi_k = np.full(self.n_cluster, 1 / self.n_cluster)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int)
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        def guard_inverse(matrix, dim):
            temp = np.copy(matrix)
            while np.linalg.matrix_rank(temp) < dim:
                temp += 0.0001 * np.identity(dim)
            return temp

        def get_gamma(x, miu, var, pi, D):
            var_det = np.linalg.det(var)
            f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - miu, np.linalg.inv(var)), x - miu), axis=1)) / (
                np.sqrt((2 * np.pi) ** D * var_det))
            return pi * f

        lll = -np.inf
        gamma = np.zeros((N, self.n_cluster))
        for i in range(self.max_iter):
            for k in range(self.n_cluster):
                miu_k = self.means[k]
                var_k = guard_inverse(self.variances[k], D)
                gamma[:, k] = get_gamma(x, miu_k, var_k, self.pi_k[k], D)
            llli = np.sum(np.log(np.sum(gamma, axis=1)))
            gamma = (gamma.T / np.sum(gamma, axis=1)).T
            Nk = np.sum(gamma, axis=0)
            for k in range(self.n_cluster):
                self.means[k] = np.transpose(np.sum(gamma[:, k] * np.transpose(x), axis=1)) / Nk[k]
                self.variances[k] = np.dot(np.multiply(np.transpose(x - self.means[k]), gamma[:, k]),
                                           x - self.means[k]) / Nk[k]
            self.pi_k = Nk / N
            if (np.abs(lll - llli) <= self.e):
                return i + 1
            lll = llli
        return self.max_iter
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        _, D = self.means.shape
        samples = np.zeros((N, D))
        rand_k = np.random.choice(self.n_cluster, N, p=self.pi_k)
        for i in range(self.n_cluster):
            miu = self.means[rand_k[i]]
            var = self.variances[rand_k[i]]
            samples[i] = np.random.multivariate_normal(miu, var)
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k
            # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        def guard_inverse(matrix, dim):
            temp = np.copy(matrix)
            while np.linalg.matrix_rank(temp) < dim:
                temp += 0.0001 * np.identity(dim)
            return temp

        def get_gamma(x, miu, var, pi, D):
            var_det = np.linalg.det(var)
            f = np.exp(-0.5 * np.sum(np.multiply(np.dot(x - miu, np.linalg.inv(var)), x - miu), axis=1)) / (
                np.sqrt((2 * np.pi) ** D * var_det))
            return pi * f
        N, D = x.shape
        gamma = np.zeros((N, self.n_cluster))
        for k in range(self.n_cluster):
            miu_k = means[k]
            var_k = guard_inverse(variances[k], D)
            gamma[:, k] = get_gamma(x, miu_k, var_k, self.pi_k[k], D)
        log_likelihood = float(np.sum(np.log(np.sum(gamma, axis=1))))
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D, D = self.variance.shape
            self.variance = GMM.guard_inverse(variance, mean.shape[1])
            self.inv = np.linalg.inv(variance)
            self.c = np.pow(2 * np.pi, D) * np.linalg.det(variance)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self, x):
            '''
                Input:
                    x: a 1 X D numpy array representing a sample
                Output:
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint:
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            # TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            xm = x - self.mean
            p = np.exp(np.matmul(np.matmul(-0.5 * xm, self.inv), np.transpose(xm) / np.sqrt(self.c)))
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
