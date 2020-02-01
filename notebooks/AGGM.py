import pandas as pd
import os
from PIL import Image
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
from numpy import asarray
import numpy as np
import scipy
from scipy.special import gamma
from test_algorithms.test_kmean import KMeansPP
import pdb


class AGGM_EM(object):

    def __init__(self, centers, X):
        self.X = X
        self.cluster_num, self.dim = centers.shape
        self.mean = centers
        cov = [np.std(X[:, i]) for i in range(self.dim)]
        #         print(np.std(X[:,0]), np.std(X[:,1]), [np.std(X[:,i]) for i in range(self.dim)])
        self.sigma_l = self.sigma_r = np.array([cov for _ in range(self.cluster_num)])
        self.coef = np.ones(self.cluster_num) / 3
        self.beta = np.array([[2 for _ in range(self.dim)] for _ in range(self.cluster_num)])
        #         self.beta = np.ones(self.cluster_num) * 2
        self.params = np.hstack(
            (self.mean.ravel(),
             self.sigma_l.ravel(),
             self.sigma_r.ravel(),
             self.beta.ravel(),
             self.coef.ravel())
        )
        self.resp = np.zeros((len(X), self.cluster_num))

    #         print(self.X)
    #         print(self.beta.shape, self.beta)
    #         print(self.mean.shape)
    #         print(self.sigma_l.shape, self.sigma_l)
    #         print(self.sigma_l.ravel())
    #         print(self.resp)

    def A(self, beta):
        return (gamma(3 / beta) / gamma(1 / beta)) ** (beta / 2)

    def _asymmetric_generalized_gaussian(self, x, mean, sigma_l, sigma_r, beta):
        coefficient = (beta * ((gamma(3 / beta) / gamma(1 / beta)) ** 0.5)) / ((sigma_l + sigma_r) * gamma(1 / beta))

        def f(new_x, sigma, beta):
            likelihood = coefficient * np.exp(-self.A(beta) * ((new_x / sigma) ** beta))
            # print('----', new_x, likelihood)
            return likelihood

        return np.where(x - mean < 0, f(mean - x, sigma_l, beta), f(x - mean, sigma_r, beta))

    def _e_step(self):
        #         print(self.X.shape)
        for i in range(self.cluster_num):
            likelihoods = []
            print('e-step ---', self.mean[i], self.sigma_l[i],
                  self.sigma_r[i], self.beta[i])
            for j in range(self.X.shape[0]):
                likelihood = self._asymmetric_generalized_gaussian(self.X[j], self.mean[i], self.sigma_l[i],
                                                                   self.sigma_r[i], self.beta[i])
                temp_likelihood = likelihood[0] * likelihood[1]
                if temp_likelihood < 0:
                    print(self.X[j], self.mean[i], self.sigma_l[i],
                          self.sigma_r[i], self.beta[i])
                    pdb.set_trace()
                likelihoods.append(temp_likelihood)
            #                 print(likelihood[0] * likelihood[1], np.dot(likelihood , likelihood.T))
            #                 print(i, j, likelihood)

            #             print(likelihoods)
            #             print(np.array(likelihoods))
            self.resp[:, i] = self.coef[i] * np.array(likelihoods)
        # normalization for over all possible cluster assignments
        # resp = resp / resp.sum(axis = 1)[:,np.newaxis]
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=True)
        print('--resp', self.resp, self.resp.shape)

    def _m_step(self):
        # print(self.X, self.X[0:3])
        #         print('---', self.resp, self.beta.shape, self.mean.shape, self.resp[:,0])
        def f(Z, new_x, sigma, beta, debug=1):
            ret = debug * Z * new_x ** (beta - 1) / (sigma ** (beta))
            #             print('in f', ret)
            return ret

        def second_f(Z, new_x, sigma, beta):
            ret = Z * new_x ** (beta - 2) / (sigma ** (beta))
            return ret

        def sigma_f(A, Z, new_x, beta, sigma):
            ret = Z * A(beta) * beta / sigma * ((new_x / sigma) ** beta)
            print("sigma_f--", ret)
            return ret

        def sigma_second_f(A, Z, new_x, beta, sigma):
            ret = Z * A(beta) * beta * (beta + 1) / (sigma ** 2) * ((new_x / sigma) ** beta)
            return ret

        def beta_f(param):
            return scipy.special.digamma(param)

        def beta_sec_f(param):
            return scipy.special.polygamma(1, param)

        def beta_f2(A, Z, new_x, beta, sigma):
            #             if new_x/sigma < 0:
            #                 print('bug!!', new_x, sigma, new_x/sigma)
            # if ((new_x / sigma) < 0).any():
            #     pdb.set_trace()
            ret = Z * A(beta) * ((new_x / beta) ** beta)
            ret = ret * ((3 * beta_f(3 / beta) - beta_f(1 / beta)) / (2 * beta) - np.log(new_x / sigma))
            return ret

        def beta_sec_f2(Z, new_x, beta, sigma):
            ret = Z * ((new_x / sigma) ** beta) * ((9 * beta_sec_f(3 / beta) - beta_sec_f(1 / beta)) / (2 * (beta ** 3))
                                                   + (3 * beta_f(3 / beta) - beta_f(1 / beta)) / (2 * (beta ** 2))
                                                   + ((3 * beta_f(3 / beta) - beta_f(1 / beta)) / (2 * beta)
                                                      - np.log(new_x / sigma)) ** 2
                                                   )
            return ret

        #################### set the parameter of mean #########################
        for i in range(self.cluster_num):
            Z = self.resp[:, i]
            Z = Z[:, np.newaxis]

            ret = np.where(self.X - self.mean[i] >= 0, f(Z, self.X - self.mean[i], self.sigma_l[i], self.beta[i]),
                           f(Z, self.mean[i] - self.X, self.sigma_r[i], self.beta[i], -1))
            first_derivative = self.A(self.beta[i]) * self.beta[i] * np.sum(ret, axis=0)

            ret2 = np.where(self.X - self.mean[i] >= 0,
                            second_f(Z, self.X - self.mean[i], self.sigma_r[i], self.beta[i]),
                            second_f(Z, self.mean[i] - self.X, self.sigma_l[i], self.beta[i]))
            second_derivative = self.A(self.beta[i]) * self.beta[i] * (self.beta[i] - 1) * np.sum(ret2, axis=0)
            print('--mean', first_derivative, second_derivative, first_derivative / second_derivative)

            ################### set the parameter of left sigma ###################
            temp_Z = np.sum(Z / (self.sigma_l[i] + self.sigma_r[i]), axis=0)
            temp_second_Z = np.sum(Z / ((self.sigma_l[i] + self.sigma_r[i]) ** 2), axis=0)

            sigma_l_first = np.sum(np.where(self.X - self.mean[i] < 0,
                                            sigma_f(self.A, Z, self.mean[i] - self.X, self.beta[i], self.sigma_l[i]),
                                            0), axis=0)
            sigma_l_first = sigma_l_first - temp_Z
            print('sigma_l_first', sigma_l_first, temp_Z)

            sigma_l_second = np.where(self.X - self.mean[i] < 0,
                                      sigma_second_f(self.A, Z, self.mean[i] - self.X, self.beta[i], self.sigma_l[i])
                                      , 0)
            sigma_l_second = np.sum(sigma_l_second, axis=0) - temp_second_Z
            print('sigma_l_second', sigma_l_second, temp_second_Z)

            ################## set the parameter of right sigma ###################
            sigma_r_first = np.sum(np.where(self.X - self.mean[i] >= 0,
                                            sigma_f(self.A, Z, self.X - self.mean[i], self.beta[i], self.sigma_r[i]),
                                            0), axis=0)
            sigma_r_first = sigma_r_first - temp_Z
            print('sigma_r_first', sigma_r_first, temp_Z)

            sigma_r_second = np.where(self.X - self.mean[i] >= 0,
                                      sigma_second_f(self.A, Z, self.X - self.mean[i], self.beta[i], self.sigma_r[i])
                                      , 0)
            sigma_r_second = np.sum(sigma_r_second, axis=0) - temp_second_Z
            print('sigma_r_second', sigma_r_second, temp_second_Z)

            ################## set the parameter of beta #######################
            beta_first = 1 / self.beta[i] - 3 / 2 * (
                    (beta_f(3 / self.beta[i]) - beta_f(1 / self.beta[i])) / self.beta[i] ** 2)

            beta_first = np.sum(Z * beta_first, axis=0)

            #             print('beta first', np.where(self.X - self.mean[i] >= 0,
            #                 beta_f2(self.A, Z, self.X - self.mean[i], self.beta[i], self.sigma_r[i])
            #               , beta_f2(self.A, Z, self.mean[i] - self.X, self.beta[i], self.sigma_l[i])))

            tmp = np.sum(np.where(self.X - self.mean[i] >= 0,
                                  beta_f2(self.A, Z, self.X - self.mean[i], self.beta[i], self.sigma_r[i])
                                  , beta_f2(self.A, Z, self.mean[i] - self.X, self.beta[i], self.sigma_l[i])), axis=0)

            #             print('tmp', tmp)
            beta_first = beta_first + tmp
            print("beta_f", beta_first)

            beta_second = np.sum(Z * (1 / (self.beta[i] ** 2) +
                                      3 * beta_sec_f(1 / self.beta[i]) / 2 * (self.beta[i] ** 4) +
                                      3 * (beta_f(1 / self.beta[i]) - beta_f(3 / self.beta[i])) / (self.beta[i] ** 3)
                                      - 9 * beta_sec_f(3 / self.beta[i]) / (2 * (self.beta[i] ** 4))
                                      ), axis=0)

            tmp2 = self.A(self.beta[i]) * np.sum(np.where(self.X - self.mean[i] >= 0,
                                                          beta_sec_f2(Z, self.X - self.mean[i], self.beta[i],
                                                                      self.sigma_r[i]),
                                                          beta_sec_f2(Z, self.mean[i] - self.X, self.beta[i],
                                                                      self.sigma_l[i]),
                                                          ), axis=0)
            beta_second = beta_second + tmp2
            print("beta_sec", beta_second, beta_first / beta_second)

            print('before - m-step ---', self.mean[i], self.sigma_l[i],
                  self.sigma_r[i], self.beta[i])

            self.mean[i] = self.mean[i] + first_derivative / second_derivative
            self.sigma_l[i] = self.sigma_l[i] + sigma_l_first / sigma_l_second
            self.sigma_r[i] = self.sigma_r[i] + sigma_r_first / sigma_r_second
            self.beta[i] = self.beta[i] + beta_first / beta_second

            print('m-step ---', self.mean[i], self.sigma_l[i],
                  self.sigma_r[i], self.beta[i])

        Nk = np.sum(self.resp, axis=0)
        # Nk/N
        self.coef = Nk / len(X)
        print(Nk, self.coef)

    def em(self):
        while True:
            self._e_step()
            self._m_step()
            new_params = np.hstack(
                (self.mean.ravel(),
                 self.sigma_l.ravel(),
                 self.sigma_r.ravel(),
                 self.beta.ravel(),
                 self.coef.ravel())
            )
            if np.allclose(self.params, new_params):
                break
            else:
                self.params = new_params
            print('-[params]>', self.params, '--')


n_samples = 300

# generate random sample, two cluster
np.random.seed(0)
# generate spherical data
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([5, 3])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# the other one
stretched_shifted_gaussian = np.dot(np.random.randn(n_samples, 2), np.array([[0., -0.7], [-3.5, .7]])) + np.array(
    [0, 5])

# concatenate the two datasets into the final training set
X = np.vstack([shifted_gaussian, stretched_gaussian, stretched_shifted_gaussian])

kpp = KMeansPP(3)
kpp.fit(X)
cluster = kpp.predict(X)

# init with the standard deviation
aggm = AGGM_EM(kpp.centers, X)
aggm.em()
# aggm._e_step()
# aggm._m_step()
# aggm._e_step()
