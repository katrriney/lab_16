import tensorflow as tf

import numpy as np

X_class = np.array([[1.6, 1.8, 3],
              [1.8, 1.9, 5],
              [1.4, 1.7, 3],
              [1.5, 1.6, 3],
              [1.8, 2, 5],
              [1.9, 2.1, 5]])
y_class = np.array([1, 0, 1, 1, 0, 0])

# Создание модели для классификации
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

model_class.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение модели
model_class.fit(X_class, y_class, epochs=100, batch_size=32)

# Прогноз
test_data = np.array([[1.6, 1.8, 3]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'papilioniadae', 'papilio podalirius'))

test_data = np.array([[1.8, 1.9, 5]])
y_pred_class = model_class.predict(test_data)
print("Предсказанные значения:", y_pred_class, *np.where(y_pred_class >= 0.5, 'papilioniadae', 'papilio podalirius'))
# Сохранение модели для классификации
model_class.save('my_model.keras.h5')