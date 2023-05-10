from pathlib import Path
import tensorflow.keras as keras


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Rescale the images from [0,255] to the [0.0, 1.0] range.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

bn_momentum = 0.9
epoch_num = 100
model = keras.models.Sequential()

model.add(keras.layers.Input(shape=[784]))

model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=16, activation="relu"))
model.add(keras.layers.Dense(units=10, activation=None))

model.add(keras.layers.Activation("softmax"))

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer="adam", metrics=["accuracy"])

print(model.summary())

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=epoch_num)

model.save(Path(__file__).parent / "model")

# save the model without the last layer
new_model = keras.models.Sequential(model.layers[:-1])
new_model.build(input_shape=model.input_shape)
new_model.save(Path(__file__).parent / "model_without_softmax")
