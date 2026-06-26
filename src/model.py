from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def build_cnn(input_shape, n_classes=2, lr=1e-3):
    model = Sequential()
    model.add(Conv2D(64, (4, 4), padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.33))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.33))
    model.add(Flatten())
    for units in (64, 64, 32, 32, 16, 4):
        model.add(Dense(units, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model
