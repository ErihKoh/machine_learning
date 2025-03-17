import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Данные
P = np.array([
    (64, 150), (84, 112), (106, 90), (154, 64), (192, 62), (220, 82), (244, 92), (271, 111),
    (275, 137), (286, 161), (56, 178), (80, 156), (101, 131), (123, 104), (155, 94), (191, 100),
    (242, 70), (231, 114), (272, 95), (261, 131), (299, 136), (308, 124), (128, 78), (47, 128),
    (47, 159), (137, 186), (166, 228), (171, 250), (194, 272), (221, 287), (253, 292), (308, 293),
    (332, 280), (385, 256), (398, 237), (413, 205), (435, 166), (447, 137), (422, 126), (400, 154),
    (389, 183), (374, 214), (358, 235), (321, 250), (274, 263), (249, 263), (208, 230), (192, 204),
    (182, 174), (147, 205), (136, 246), (147, 255), (182, 282), (204, 298), (252, 316), (312, 321),
    (349, 313), (393, 288), (417, 259), (434, 222), (443, 187), (463, 174)
])

# Параметры DBSCAN
eps = 40  # Радиус окрестности
min_samples = 5  # Минимальное количество точек для кластера

# Запуск DBSCAN
db = DBSCAN(eps=eps, min_samples=min_samples).fit(P)
labels = db.labels_

# Визуализация кластеров
unique_labels = set(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))  # Разные цвета для кластеров

plt.figure(figsize=(8, 6))
for label in unique_labels:
    mask = labels == label
    color = "gray" if label == -1 else colors(label)  # Шумовые точки серые
    plt.scatter(P[mask, 0], P[mask, 1], c=[color], label=f"Cluster {label}" if label != -1 else "Noise")

plt.legend()
plt.show()