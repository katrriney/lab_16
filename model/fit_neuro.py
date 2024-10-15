import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[1.6, 1.8, 3],
              [1.8, 1.9, 5],
              [1.4, 1.7, 3],
              [1.5, 1.6, 3],
              [1.8, 2, 5],
              [1.9, 2.1, 5]])
y = np.array([1, 0, 1, 1, 0, 0])
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.0001)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')