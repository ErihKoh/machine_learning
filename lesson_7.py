import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані
x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70],
           [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]

# Додаємо зміщення (w2)
x_train = [x + [1] for x in x_train]
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

# Обчислення ваг
#  Loss function: L_i(w) = (1 - y_i * (w^T * x_i))^2
pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)
xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)
w = np.dot(pt, np.linalg.pinv(xxt))  # Використовуємо псевдообернену матрицю

print("Фінальні ваги:", w)

# Формуємо графік розділяючої лінії
line_x = np.linspace(0, 45, 100)
line_y = - (w[0] * line_x + w[2]) / w[1]

# Формуємо точки для класів
x_0 = x_train[y_train == 1, :2]
x_1 = x_train[y_train == -1, :2]

# Візуалізація
plt.scatter(x_0[:, 0], x_0[:, 1], color='red', label='Клас 1')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue', label='Клас -1')
plt.plot(line_x, line_y, color='green', label='Розділяюча лінія')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.xlabel("ширина")
plt.ylabel("довжина")
plt.grid(True)
plt.legend()
plt.show()