import numpy as np
import matplotlib.pyplot as plt


# сигмоідальна функція втрат
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


# похідна сигмоідальної функції втрат по вектору w
def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y


#  дані для навчання с трьома ознаками (константа +1)
x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70],
           [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x +[1] for x in x_train]
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train)  # розмір навчальної вибірки
w = [0.0, 0.0, 0.0]  # початкові вагові коефіцієнти
step = 0.0005  # крок SGD
lm = 0.01  # швидкість забування для Q
N = 500  # кількість ітерацій

Q = np.mean([loss(x, w, y) for x, y in zip(x_train, y_train)])  # показник якості
Q_plot = [Q]

for i in range(N):
    k = np.random.randint(0, n_train - 1)  # випадковий індекс
    ek = loss(w, x_train[k], y_train[k])  # розрахунок втрат для випадкового вектору
    w = w - step * df(w, x_train[k], y_train[k])  # корекція ваг по SGD
    Q = lm * ek + (1 - lm) * Q  # перерахунок ваг
    Q_plot.append(Q)

print(w)
print(Q_plot)

line_x = list(range(max(x_train[:, 0])))  # формування відокремлюючої лінії
line_y = [-x * w[0] / w[1] - w[2] for x in line_x]

x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel('довжина')
plt.xlabel('ширина')
plt.grid(True)
plt.show()



