import random
from collections import Counter
import math


def frequency(data):
    """频率"""
    counter = Counter(data)
    ret = []
    for point in counter.most_common():
        ret.append((point[0], point[1] / len(data)))
    return ret


def mode(data):
    """
        众数 出现的次数
    """
    counter = Counter(data)
    if counter.most_common()[0][1] == 1:
        return None, None
    count = counter.most_common()[0][1]
    ret = []
    for point in counter.most_common():
        if point[1] == count:
            ret.append(point[0])
        else:
            break
    return ret, count


def median(data):
    """中位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)

    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2


def mean(data):
    return sum(data) / len(data)


def range(data):
    """极差"""
    return max(data) - min(data)


def quartile(data):
    """四分数"""
    n = len(data)
    q1, q2, q3 = None, None, None
    if n >= 4:
        sorted_data = sorted(data)
        q2 = median(sorted_data)
        q1 = median(sorted_data[:n // 2])
        if n % 2 == 1:
            q3 = median(sorted_data[n // 2 + 1:])
        else:
            q3 = median(sorted_data[n // 2:])
    return q1, q2, q3


def variance(data):
    """方差"""
    n = len(data)
    if n <= 1:
        return None
    mean_val = mean(data)
    # 样本方差用n-1 总体数据用n
    return sum((e - mean_val) ** 2 for e in data) / (n - 1)


def std(data):
    """标准差"""
    return math.sqrt(variance(data))


# plt.scatter plt.plot plt.bar plt.hist  plt.boxplot
def plot():
    data = [random.randint(66, 166) for _ in range(200)]


if __name__ == '__main__':
    data = [1, 2, 2, 1]
    counter = Counter(data)
    print(counter.most_common())
    print(counter.most_common()[0][1])
    # print(10 / 2, 19 // 2)
    # print(data[1:])
    # print(4 ** 0.5)
