from matplotlib import pyplot as plt
import numpy
from sklearn import datasets

data1 = datasets.make_regression(100, 1, noise=5)
plt.scatter(data1[0], data1[1], c='red', marker='^')
init_m = 10
init_b = 10
learning_rate = 0.1
range1 = [-5, 5]
plt.plot(range1, init_m * numpy.array(range1) + init_b)
# plt.show()

def cost(m, b, X, Y):
    cost = 0
    for i in range(len(X)):
        cost += abs(Y[i] - (m * X[i] + b))
    return cost


init_cost = cost(init_m, init_b, data1[0], data1[1])
print(f"m={init_m}, b={init_b}, init_cost={init_cost}")


def update_weight(m, b, X, Y, learninig_rate):
    m_deriv = 0
    b_deriv = 0
    for i in range(len(X)):
        m_deriv += -2 * X[i] * (Y[i] - (m * X[i] + b))
        b_deriv += -2 * (Y[i] - (m * X[i] + b))
    m -= (m_deriv / len(X)) * learninig_rate
    b -= (b_deriv / len(X)) * learninig_rate
    return m, b


current_m = init_m
current_b = init_b
for _ in range(20):
    new_m, new_b = update_weight(current_m, current_b,
                                 data1[0], data1[1], learning_rate)
    range1 = [-5, 5]
    plt.plot(range1, new_m * numpy.array(range1) + new_b)
    plt.scatter(data1[0], data1[1], c='red', marker='^')
    new_cost = cost(new_m, new_b, data1[0], data1[1])
    print(f"m={new_m[0]}, b={new_b[0]}, init_cost={new_cost}")
    plt.figure()
    current_m = new_m
    current_b = new_b
plt.show()