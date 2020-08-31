import keras.layers as kl
import keras.models as km
import tensorflow as tf
import numpy as np

print("Creating neural network")

X = np.random.randn(100, 16)
y = np.random.randn(100, 1)

input_tensor = kl.Input(shape=(16,))
h1 = kl.Dense(units=5, activation="relu")(input_tensor)
h2 = kl.Dense(units=1, activation=None)(h1)
output_tensor = kl.Lambda(lambda x: tf.sigmoid(x))(h2)

print("Compiling model")
model = km.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

print("Training model")
model.fit(X, y, epochs=50, batch_size=10)

print("Final MSE")
_, mse = model.evaluate(X, y)
print(mse)
