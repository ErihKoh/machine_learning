import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані
x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70],
                    [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

# Початкові ваги
w = np.array([0.0, -1.0])


# Функція визначення класу
def a(x):
    return np.sign(np.dot(x, w))


# Налаштування алгоритму
N = 50  # Кількість ітерацій
L = 0.1  # Крок оновлення ваг

# Навчання (Перцептрон Розенблатта)
for n in range(N):
    errors = 0  # Лічильник помилок
    for i in range(len(x_train)):
        if y_train[i] * a(x_train[i]) < 0:  # Помилка класифікації
            w += L * y_train[i] * x_train[i]  # Оновлення ваг
            errors += 1  # Підраховуємо помилки
    if errors == 0:  # Якщо немає помилок, зупиняємо навчання
        break

print("Фінальні ваги:", w)

# Побудова лінії розділу
line_x = np.linspace(0, 45, 100)
line_y = - (w[0] * line_x) / w[1]  # Рівняння лінії w0*x + w1*y = 0

# Відокремлення точок за класами
x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

# Графік
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
