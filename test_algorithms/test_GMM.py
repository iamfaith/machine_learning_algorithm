


import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np

x = np.linspace(0, 10, 1000)
y1 = [ss.norm.pdf(v, loc=5, scale=1) for v in x]
y2 = [ss.norm.pdf(v, loc=1, scale=1.3) for v in x]
y3 = [ss.norm.pdf(v, loc=9, scale=1.3) for v in x]
y = np.sum([y1, y2, y3], axis=0) / 3

plt.plot(x, y, '-')
plt.xlabel('$x$')
plt.ylabel('$P(x)$')