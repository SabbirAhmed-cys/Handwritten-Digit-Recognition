"""Train the 128x128 handwritten digit CNN used by app.py."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)

print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_val, y_val = x_train[-6000:], y_train[-6000:]
x_train, y_train = x_train[:-6000], y_train[:-6000]

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, -1)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bilinear")
    image = image / 255.0
    return image, label

def make_dataset(images, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(10000, seed=SEED)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(x_train, y_train, training=True)
val_ds = make_dataset(x_val, y_val)
test_ds = make_dataset(x_test, y_test)

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "digit_model_best.keras", monitor="val_accuracy", save_best_only=True
    ),
]
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)
loss, accuracy = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")
model.save("digit_model.keras")
print("Saved digit_model.keras")
