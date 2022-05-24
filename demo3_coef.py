from matplotlib import pyplot as plt
import numpy as np

b = 5
a = np.linspace(3, -3, 10)
x = np.arange(-5, 5, 0.1)
print(f"b={b}")
print(f"x={x}")
for a1 in a:
    y = a1 * x + b
    plt.plot(x, y, label=f"y={a1}x+{b:.1f}")
    plt.legend(loc=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.axhline(5, color='purple')
plt.show()