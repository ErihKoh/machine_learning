import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Исходные данные
x = np.arange(0, 10, 0.1)
x_est = np.linspace(0, 10, 1000)  # Более гладкая сетка
N = len(x)
y_sin = np.sin(x)
y = y_sin + np.random.normal(0, 0.5, N)  # Добавление шума

# Определение ядер
kernels = {
    "gaussian": lambda r: np.exp(-2 * r ** 2),
    "triangular": lambda r: np.maximum(1 - np.abs(r), 0),
    "rectangular": lambda r: (np.abs(r) <= 1).astype(float)
}


# Функция ядерного сглаживания
def nadaraya_watson(x, y, x_est, h, kernel):
    """Аппроксимация методом Надарая-Ватсона."""
    xx, xi = np.meshgrid(x_est, x)
    weights = kernel(np.abs(xx - xi) / h)
    weights /= np.sum(weights, axis=0)  # Нормализация весов
    return np.dot(weights.T, y)


# Кросс-валидация для подбора h
def optimize_bandwidth(x, y, kernel, h_values):
    best_h, best_mse = None, float('inf')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for h in h_values:
        mse_values = []
        for train_idx, test_idx in kf.split(x):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            y_pred = nadaraya_watson(x_train, y_train, x_test, h, kernel)
            mse_values.append(np.mean((y_test - y_pred) ** 2))

        mse = np.mean(mse_values)
        if mse < best_mse:
            best_mse, best_h = mse, h

    return best_h


# Выбор ядра и оптимального h
kernel_type = "triangular"
kernel = kernels[kernel_type]
h_values = np.linspace(0.1, 2, 10)
best_h = optimize_bandwidth(x, y, kernel, h_values)

# Визуализация
plt.figure(figsize=(7, 5))
y_est = nadaraya_watson(x, y, x_est, best_h, kernel)
plt.scatter(x, y, color='black', s=10, label='Данные')
plt.plot(x, y_sin, color='blue', label='Исходная функция')
plt.plot(x_est, y_est, color='red', label=f'Ядерное сглаживание (h={best_h:.2f})')
plt.title(f"Ядерное сглаживание ({kernel_type} ядро)")
plt.legend()
plt.grid()
plt.show()
