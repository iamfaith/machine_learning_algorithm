import unittest

import logging
import numpy as np
import matplotlib.pyplot as plt
import sys


class KMeans(object):
    def __init__(self, k, log):
        """
        :param k: suppose for the moment that the value of K is given
        """
        self.cluster_nums = k
        self.output = log
        self.debug = self.output.debug
        self.centers = []

    def fit(self, train_data, iter_max=100):
        I = np.eye(self.cluster_nums)
        # self.debug(I)
        centers = train_data[np.random.choice(len(train_data), self.cluster_nums, replace=False)]

        for i in range(iter_max):
            prev_centers = np.copy(centers)
            euclidean_dist = self.cal_euclidean_distance(centers, train_data)

            # find cluster index by euclidean distance E-step
            cluster_index = np.argmin(euclidean_dist, axis=1)
            # binary indicator variables -> Rnk
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
        kmeans = KMeans(3, log)
        kmeans.fit(x_train)
        cluster = kmeans.predict(x_train)
        log.debug(cluster)
        log.debug('->begin to draw gaussian models')
        plt.scatter(x_train[:, 0], x_train[:, 1], c=cluster)
        plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=200, marker='X', lw=2, c=['purple', 'cyan', 'yellow'],
                    edgecolor="white")
        plt.show()


# class TestAlgorithm(unittest.TestCase):
#     def runTest(self):
#         log = logging.getLogger("TestAlgorithm")
#         log.debug("hello world")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    # unittest.TextTestRunner().run(TestAlgorithm())
    unittest.main()
