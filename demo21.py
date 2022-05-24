from matplotlib import pyplot as plt
import numpy as np

w1 = 3.0
b = [-8, -4, 0, 4, 8]
label_template = "b=%.1f"

x = np.arange(-10, 10, 0.1)
for b1 in b:
    f = 1 / (1 + np.exp(-(w1 * x + b1))) #改變x軸位移
    plt.plot(x, f, label=label_template % b1)
plt.axvline(0, color='black')
plt.axhline(0.5, color='black')
plt.legend()
plt.show()