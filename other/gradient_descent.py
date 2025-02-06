import time
import numpy as np
import matplotlib.pyplot as plt


# def f(x):
#     return x * x - 5 * x + 5
#
#
# def df(x):
#     return 2*x - 5  # похідна

def f(x):
    return np.sin(x) + 0.5*x


def df(x):
    return np.cos(x) + 0.5  # похідна


N = 20  # кількість ітерацій
xx = 0  # початкове значення
lmd = 0.1  # крок

# x_plt = np.arange(0, 5.0, 0.1)
x_plt = np.arange(-5.0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion()  # інтерактивний режим відображення графіка
fig, ax = plt.subplots()  # створення вікна та ось для графіка
ax.grid(True)  # сітка на графіку

ax.plot(x_plt, f_plt)  # парабола
point = ax.scatter(xx, f(xx), c='red')  # точка червоним кольором

#  початока роботи алгоритму градієртного спуску
mn = 100
for i in range(N):
    lmd = 1 / min(i + 1, mn)
    xx = xx - lmd*np.sign(df(xx))  # зміна аргументу на поточній ітерації

    point.set_offsets([xx, f(xx)])  # відображення нового положення точки

    # перемалювання графіка з затримкою 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)

plt.ioff()
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()
