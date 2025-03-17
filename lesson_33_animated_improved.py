import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Исходные данные
# x = np.array([(126, 63), (101, 100), (80, 160), (88, 208), (89, 282), (88, 362), (94, 406), (149, 377), (147, 304),
#               (147, 235), (146, 152), (160, 103), (174, 142), (169, 184), (170, 241), (169, 293), (185, 376),
#               (178, 422), (116, 353), (124, 194), (273, 69), (277, 112), (260, 150), (265, 185), (270, 235),
#               (265, 295), (281, 351), (285, 416), (321, 404), (316, 366), (306, 304), (309, 254), (309, 207),
#               (327, 161), (318, 108), (306, 66), (425, 66), (418, 135), (411, 183), (413, 243), (414, 285),
#               (407, 333), (411, 385), (443, 387), (455, 330), (441, 252), (457, 207), (453, 149), (455, 90),
#               (455, 56), (439, 102), (431, 162), (431, 193), (426, 236), (427, 281), (438, 323), (419, 379),
#               (425, 389), (422, 349), (451, 275), (441, 222), (297, 145), (284, 195), (288, 237), (292, 282),
#               (288, 313), (303, 356), (293, 395), (274, 268), (280, 344), (303, 187), (114, 247), (131, 270),
#               (144, 215), (124, 219), (98, 240), (96, 281), (146, 267), (136, 221), (123, 166), (101, 185),
#               (152, 184), (104, 283), (74, 239), (107, 287), (118, 335), (89, 336), (91, 315), (151, 340),
#               (131, 373), (108, 133), (134, 130), (94, 260), (113, 193)])

x = np.array([(98, 62), (80, 95), (71, 130), (89, 164), (137, 115), (107, 155), (109, 105), (174, 62), (183, 115), (164, 153),
     (142, 174), (140, 80), (308, 123), (229, 171), (195, 237), (180, 298), (179, 340), (251, 262), (300, 176),
     (346, 178), (311, 237), (291, 283), (254, 340), (215, 308), (239, 223), (281, 207), (283, 156)])

# Параметры
K = 3  # Число кластеров
max_iter = 10  # Максимальное число итераций
COLORS = ('green', 'blue', 'brown', 'black', 'purple', 'orange')
assert K <= len(COLORS), "Увеличьте количество цветов для отображения кластеров"

# Инициализация центроид
np.random.seed(42)
M, D = np.mean(x, axis=0), np.var(x, axis=0)
centroids = np.random.normal(M, np.sqrt(D / 10), (K, 2))


# Функция расчета расстояния (векторизованная)
def assign_clusters(x, centroids):
    distances = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


# Функция обновления центроид
def update_centroids(x, labels, K):
    return np.array([np.mean(x[labels == k], axis=0) if np.any(labels == k) else centroids[k] for k in range(K)])


# Анимация алгоритма
def animate(i):
    global centroids
    plt.clf()
    labels = assign_clusters(x, centroids)
    centroids = update_centroids(x, labels, K)

    for k in range(K):
        plt.scatter(x[labels == k, 0], x[labels == k, 1], s=10, color=COLORS[k])
        plt.scatter(*centroids[k], s=100, color='red', marker='X')

    plt.title(f'Итерация {i + 1}')
    plt.grid()


time.sleep(1)
fig = plt.figure(figsize=(6, 6))
ani = FuncAnimation(fig, animate, frames=max_iter, repeat=False)
plt.show()
