import random
import unittest

import logging
import numpy as np
import matplotlib.pyplot as plt
import sys


class KMeans(object):
    """
    Various schemes have been proposed for speeding up the K-means algorithm, some of which are based on precomputing a data structure such as a tree such that nearby points are in the same subtree (Ramasubramanian and Paliwal, 1990; Moore, 2000). Other approaches make use of the triangle inequality for distances, thereby avoiding unnecessary dis- tance calculations (Hodgson, 1998; Elkan, 2003).

    But how to overcome Random initialization trap/local optima?
    Use k-means++
    """

    def __init__(self, k, log):
        """
        :param k: suppose for the moment that the value of K is given
        """
        self.cluster_nums = k
        self.output = log
        self.debug = self.output.debug
        self.centers = []

    def center_initialization(self, train_data):
        self.debug('random initialization')
        return train_data[np.random.choice(len(train_data), self.cluster_nums, replace=False)]

    def fit(self, train_data, iter_max=100):
        I = np.eye(self.cluster_nums)
        # self.debug(I)
        centers = self.center_initialization(train_data)

        for i in range(iter_max):
            prev_centers = np.copy(centers)
            # m * cluster_numbers
            euclidean_dist = self.cal_euclidean_distance(centers, train_data)
            # find cluster index by euclidean distance E-step
            # m * 1
            cluster_index = np.argmin(euclidean_dist, axis=1)
            # binary indicator variables -> Rnk
            # m * 3
            cluster_index = I[cluster_index]
            # M-step Rnk * Train / Rnk
            # cluster_index m*3*1  train_data m*1*2 => m * 3 * 2 => sum axis=0
            # => 3*2 => / sum(m*3*1, axis=0) 3*1 => 3*2
            centers = np.sum(cluster_index[:, :, np.newaxis] * train_data[:, np.newaxis, :], axis=0) / np.sum(
                cluster_index, axis=0)[:, np.newaxis]
            if np.allclose(prev_centers, centers):
                break
        self.centers = centers
        self.debug(self.centers)

    def predict(self, X):
        euclidean_dist = self.cal_euclidean_distance(self.centers, X)
        return np.argmin(euclidean_dist, axis=1)

    @staticmethod
    def cal_euclidean_distance(centers, X):
        euclidean_dist = []
        for row in range(len(centers)):
            # calculate the euclidean distance between train data to each center
            # (a, b) (c, d) -> ((a-b)^2 + (c-d)^2)^(1/2)
            p = np.power(X - centers[row], 2)
            euclidean_dist.append(np.power(np.sum(p, 1), 1 / 2))
        euclidean_dist = np.vstack(euclidean_dist).T
        # or use euclidean_dist = cdist(train_data, centers)
        return euclidean_dist


class KMeansPP(KMeans):
    """
    KMeansPP means k-means++
    it can eliminate the local optima by choosing the initial centers.
    """

    def center_initialization(self, train_data):
        first_center = train_data[np.random.choice(len(train_data), 1, replace=False)]
        centers = [first_center[0]]
        while len(centers) < self.cluster_nums:
            # np.power(np.sum(np.power(train_data[0] - centers[0], 2)), 1/2)
            # (a, b) (c, d)  ((a-c)^2 + (b-d)^2)1/2
            D2 = np.array([min([np.linalg.norm(x - c) ** 2 for c in centers]) for x in train_data])
            cumprobs = (D2 / D2.sum()).cumsum()
            print(D2.shape, train_data.shape, cumprobs.shape)
            r = random.random()
            chosen = train_data[cumprobs >= r][0]
            centers.append(chosen)
        # print(centers)
        return centers


class TestKMeans(unittest.TestCase):
    def test_k_means(self):
        log = logging.getLogger("K-Means")
        x1 = np.random.normal(size=(100, 2))
        x1 += np.array([-5, -5])
        x2 = np.random.normal(size=(100, 2))
        x2 += np.array([5, -5])
        x3 = np.random.normal(size=(100, 2))
        x3 += np.array([0, 5])
        x_train = np.vstack((x1, x2, x3))
        log.debug('initialize data successful')
        kmeans = KMeans(3, log)
        kmeans.fit(x_train)
        cluster = kmeans.predict(x_train)
        log.debug(cluster)
        log.debug('->begin to draw gaussian models')
        plt.scatter(x_train[:, 0], x_train[:, 1], c=cluster)
        plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=200, marker='X', lw=2, c=['purple', 'cyan', 'yellow'],
                    edgecolor="white")
        plt.show()


class TestKmeanPP(unittest.TestCase):
    def runTest(self):
        log = logging.getLogger("TestKmeanPP")
        x1 = np.random.normal(size=(100, 2))
        x1 += np.array([-5, -5])
        x2 = np.random.normal(size=(100, 2))
        x2 += np.array([5, -5])
        x3 = np.random.normal(size=(100, 2))
        x3 += np.array([0, 5])
        x_train = np.vstack((x1, x2, x3))
        kpp = KMeansPP(3, log)
        kpp.fit(x_train)
        cluster = kpp.predict(x_train)
        log.debug('->begin to draw gaussian models')
        plt.scatter(x_train[:, 0], x_train[:, 1], c=cluster)
        plt.scatter(kpp.centers[:, 0], kpp.centers[:, 1], s=200, marker='X', lw=2, c=['purple', 'cyan', 'yellow'],
                    edgecolor="white")
        plt.show()

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.TextTestRunner().run(TestKmeanPP())
    # unittest.main()
