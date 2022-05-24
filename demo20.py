from matplotlib import pyplot as plt
import numpy as np

weights = [0.25, 0.5, 1.0, 2.0, 4.0]
label_template = "weight=%.2f"

x = np.arange(-10, 10, 0.1)
for w in weights:
    f = 1 / (1 + np.exp(-x * w))
    plt.plot(x, f, label=label_template % w)
plt.legend(loc=2)
plt.show()